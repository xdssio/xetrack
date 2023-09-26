import contextlib
import os
import typing
from typing import List
from uuid import uuid4
from datetime import datetime as dt
import logging
import multiprocessing
import time
import psutil
import re

from xetrack.stats import Stats
from xetrack.connection import DuckDBConnection
from xetrack.constants import TABLE, TRACK_ID

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method('fork')

default_logger = logging.getLogger(__name__)
ALPHANUMERIC = re.compile(r'[^a-zA-Z0-9_]')


class Tracker(DuckDBConnection):
    """
    Tracker class for tracking experiments, benchmarks, and other events.
    You can set params which are always attached to every event, and then track any parameters you want.
    """
    FUNCTION_NAME = 'function_name'
    FUNCTION_TIME = 'function_time'
    FUNCTION_RESULT = 'function_result'
    ARGS = 'args'
    KWARGS = 'kwargs'
    ERROR = 'error'    
    TIMESTAMP = 'timestamp'

    def __init__(self, db: str = 'track.db',
                 params=None,
                 reset: bool = False,
                 log_system_params: bool = True,
                 log_network_params: bool = True,
                 raise_on_error: bool = True,
                 measurement_interval: float = 1,
                 logger=None
                 ):
        """
        Initializes the class instance.

        :param db: The duckdb database file to use or ":memory:" for in-memory database.
                   Default is "track.db".
        :param params: A dictionary of default parameters to attach to every event. This can be changed later.
        :param reset: If True, the events table will be dropped and recreated. Default is False.
        :param log_system_params: If True, system parameters will be logged. Default is True.
        :param log_network_params: If True, network parameters will be logged. Default is True.
        :param raise_on_error: If True, an error will be raised on failure. Default is True.
        :param measurement_interval: The interval in seconds at which to measure events. Default is 1.
        :param logger: The logger to use for logging. If not provided, no logging are printed.
        """
        super().__init__(db=db)
        if params is None:
            params = {}
        self.params = params
        self.track_id = self.generate_uuid4()
        self._columns = set()
        self._create_events_table(reset=reset)
        self.log_system_params = log_system_params
        self.log_network_params = log_network_params
        self.raise_on_error = raise_on_error
        self.measurement_interval = measurement_interval
        self.logger = logger
        self.latest = None

    @staticmethod
    def _is_primitive(value: object) -> bool:
        """
        Check if the given value is a primitive data type.

        Args:
            value: The value to be checked.

        Returns:
            bool: True if the value is a primitive data type, False otherwise.
        """
        return type(value) in (int, float, bool, str, bytes, bytearray)

    def _create_events_table(self, reset: bool = False):
        """
        Creates the events table in the database.

        Parameters:
            reset (bool): If True, drops the table if it already exists and recreates it. Defaults to False.

        Returns:
            None
        """
        if reset:
            self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ({Tracker.TIMESTAMP} VARCHAR PRIMARY KEY, {TRACK_ID} VARCHAR)")
        self.conn.execute("SHOW TABLES").fetchall()
        self._columns = set(self.dtypes.keys())
        for key, value in self.params.items():
            self.add_column(key, value, self.to_py_type(value))
        self._columns = set(self.dtypes.keys())

    @staticmethod
    def generate_uuid4():
        return str(uuid4())

    @staticmethod
    def to_py_type(value):
        value_type = type(value)
        if value_type == bytearray:
            return 'BLOB'
        if value_type == int:
            if value.bit_length() > 64:
                return 'BIGINT'
            else:
                return 'INTEGER'
        if value_type == float:
            return 'FLOAT'
        if value_type == bool:
            return 'BOOLEAN'
        return 'VARCHAR'

    def _drop_table(self):
        return self.conn.execute(f"DROP TABLE {TABLE}")

    @property
    def _len(self) -> int:
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {TRACK_ID} = '{self.track_id}'").fetchall()[
            0][0]

    def __len__(self):
        return self._len

    @staticmethod
    def get_timestamp():
        return dt.now().strftime('%d-%m-%Y %H:%M:%S.%f')

    # TODO make more efficient
    def track_batch(self, data=List[dict]):
        if len(data) == 0:
            if self.logger:
                self.logger.warning('No values to track')
            return data
        for values in data:
            self.log(**values)

    @staticmethod
    def to_mb(bytes: int):
        """
        Convert a given number of bytes to megabytes.

        Args:
            bytes (int): The number of bytes to convert.

        Returns:
            float: The equivalent number of megabytes.
        """
        return bytes / (1024 * 1024)

    def _to_send_recv(self, net_io_before, net_io_after):
        """
        Calculates the amount of data sent and received during a network IO operation.

        Args:
            net_io_before (NetIOStats): The network IO statistics before the operation.
            net_io_after (NetIOStats): The network IO statistics after the operation.

        Returns:
            Tuple[float, float]: A tuple containing the amount of data sent and received in megabytes.
        """
        sleep_bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent
        sleep_bytes_recv = net_io_after.bytes_recv - net_io_before.bytes_recv
        return self.to_mb(sleep_bytes_sent), self.to_mb(sleep_bytes_recv)

    def validate_result(self, data: dict, result: typing.Union[dict, typing.Any]):
        """
        Validate the result of a function execution.

        Args:
            data (dict): The original data dictionary.
            result (Union[dict, Any]): The result of the function execution.

        Returns:
            dict: The updated data dictionary.
        """
        if isinstance(result, dict):
            for value in (Tracker.FUNCTION_TIME, Tracker.FUNCTION_RESULT, Tracker.FUNCTION_NAME):
                if value in result and self.logger:  # we don't want to override the time
                    self.logger.warning(f"Overriding {value}={result[value]}-> {data[value]}")
            data.update(result)
        else:
            if not self._is_primitive(result):
                result = str(result)
            data[Tracker.FUNCTION_RESULT] = result
        return data

    def _run_func(self, func: callable, *args, **kwargs):
        """
        Runs a given function with the provided arguments and keyword arguments.

        Args:
            func (callable): The function to be executed.
            *args: Variable length argument list to be passed to the function.
            **kwargs: Arbitrary keyword arguments to be passed to the function.

        Returns:
            tuple: A tuple containing the following elements:
                - data (dict): A dictionary containing information about the function execution, including:
                    - function_name (str): The name of the function.
                    - args (str): A string representation of the arguments passed to the function.
                    - kwargs (str): A string representation of the keyword arguments passed to the function.
                    - error (str): The error message, if any, raised during the execution of the function.
                    - time (float): The time taken to execute the function.
                - result: The result of the function execution.
                - exception: The exception, if any, raised during the execution of the function.
        """
        func_name, error, exception, result = func.__name__, '', None, None
        if self.logger:
            self.logger.debug(f"Running {func_name} with {args} and {kwargs}")
        start_func_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
            error = str(e)
        func_time = time.time() - start_func_time
        data = {'function_name': func_name,
                'args': str(list(args)),
                'kwargs': str(kwargs),
                'error': error,
                'function_time': func_time}
        data = self.validate_result(data, result)
        return data, result, exception

    def wrap(self, params: dict = {}):
        """
        Generates a decorator that tracks the execution of a function.

        Args:
            params (dict, optional): Additional parameters to pass to the tracker. Defaults to {}.

        Returns:
            TrackDecorator: A decorator object that tracks the execution of a function.
        """
        parent_tracker = self

        class TrackDecorator:
            def __init__(self, func):
                self.func = func
                self.params = params
                self.tracker = parent_tracker

            def __call__(self, *args, **kwargs):
                return self.tracker.track(self.func, params=self.params, args=args, kwargs=kwargs)

        return TrackDecorator

    def track(self, func: callable, params: dict = None, args: list = None, kwargs: dict = None):
        """
        Executes a function and logs the execution data.

        Args:
            func (callable): The function to be executed.
            params (dict, optional): Additional parameters to log along with the execution data. Defaults to None.
            args (list, optional): Positional arguments to pass to the function. Defaults to None.
            kwargs (dict, optional): Keyword arguments to pass to the function. Defaults to None.

        Returns:
            Any: The return value of the executed function.

        """
        if params is None:
            params = {}
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if self.log_network_params:
            net_io_before = psutil.net_io_counters()
        if self.log_system_params:
            process = psutil.Process(os.getpid())
            manager = multiprocessing.Manager()
            stop_event = multiprocessing.Event()
            stats = Stats(process, stats=manager.dict(), interval=self.measurement_interval)
            stats_process = multiprocessing.Process(target=stats.collect_stats, args=(stop_event,))
            stats_process.start()
        data, result, exception = self._run_func(func, *args, **kwargs)
        data.update(params)
        if self.log_system_params:
            stop_event.set()
            data.update(stats.get_average_stats())
        if self.log_network_params:
            bytes_sent, bytes_recv = self._to_send_recv(net_io_before, psutil.net_io_counters())
            data.update({'bytes_sent': bytes_sent, 'bytes_recv': bytes_recv})
        self.log(**data)
        if exception is not None and self.raise_on_error:
            raise exception
        return result

    def add_column(self, key, value, dtype: type):
        if key not in self._columns:
            self.conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {key} {dtype}")
            self._columns.add(key)
        elif self.logger:
            self.logger.warning(f'Column {key} already exists')
        return value

    def _to_valid_key(self, key):
        key = str(key)
        key = ALPHANUMERIC.sub('_', key)

        if ALPHANUMERIC.match(key[0]) is not None:
            key = '_' + key

        # Truncate the name if needed
        max_length = 64
        key = key[:max_length]
        return key

    def _validate_data(self, data: dict):
        if Tracker.TIMESTAMP in data and self.logger:
            self.logger.warning(f"Overriding {Tracker.TIMESTAMP} - please use another key")
        if TRACK_ID in data and self.logger:
            self.logger.warning(f"Overriding {TRACK_ID} - please use another key")
        data = {self._to_valid_key(key): value for key, value in data.items()}
        new_columns = set(data.keys()) - self._columns
        for key in new_columns:
            self.conn.execute(
                f"ALTER TABLE {TABLE} ADD COLUMN {key} {self.to_py_type(data[key])}")
            self._columns.add(key)

        for key, value in self.params.items():
            if key not in data:
                data[key] = value
            elif self.logger:
                self.logger.warning(f'Overriding the {key} parameter')
        data[Tracker.TIMESTAMP] = self.get_timestamp()
        data[TRACK_ID] = self.track_id
        return data

    def _to_key_values(self, data: dict):
        keys, values = [], []
        for i, (key, value) in enumerate(data.items()):
            keys.append(key)
            values.append(value)
        return keys, values, i + 1

    def log(self, **data: dict) -> int:
        data_size = len(data)
        if data_size == 0:
            if self.logger:
                self.logger.warning('No values to track')
            return data_size
        data = self._validate_data(data)
        keys, values, size = self._to_key_values(data)
        self.conn.execute(
            f"INSERT INTO {TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
            list(values))
        if self.logger:
            message = '\t'.join([f'{key}={value}' for key, value in data.items()])
            self.logger.info(f"{message}")
        self.latest = data
        return data

    def __repr__(self):
        return self.head().__repr__()

    def to_df(self, all: bool = False):
        query = f"SELECT * FROM {TABLE}"
        if not all:
            query += f" WHERE {TRACK_ID} = '{self.track_id}'"
        return self.conn.execute(query).df()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.conn.execute(f"SELECT {item} FROM {TABLE} WHERE {TRACK_ID} = '{self.track_id}'").df()[
                item]

        elif isinstance(item, int):
            results = self.conn.execute(f"SELECT * FROM {TABLE} WHERE {Tracker.ID} = {item}").df()
            if len(results) == 0:
                return None
            return results.iloc[0].to_dict()
        raise ValueError(f"Invalid type: {type(item)}")

    def set_params(self, params: dict):
        for key, value in params.items():
            self.set_param(key, value)

    def set_value(self, key, value):
        if key not in self._columns:
            self.add_column(key, value, self.to_py_type(value))
        self.conn.execute(
            f"UPDATE {TABLE} SET {key} = '{value}' WHERE {TRACK_ID} = '{self.track_id}'")

    def set_param(self, key, value):
        self.params[key] = value
        self.add_column(key, value, self.to_py_type(value))
        return key

    def head(self, n: int = 5):
        return self.conn.execute(
            f"SELECT * FROM {TABLE} WHERE {TRACK_ID} = '{self.track_id}' LIMIT {n}").df()

    def tail(self, n: int = 5):
        return self.conn.execute(
            f"SELECT * FROM {TABLE} WHERE {TRACK_ID} = '{self.track_id}' ORDER BY {Tracker.TIMESTAMP} DESC LIMIT {n}").df()

    def count_all(self):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {TABLE}").fetchall()[
            0][0]

    def count_run(self):
        return self._len

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def to_csv(self, path: str, all: bool = False):
        return self.to_df(all).to_csv(path, index=False)

    def to_parquet(self, path: str, all: bool = False):
        return self.to_df(all).to_parquet(path, index=False)
