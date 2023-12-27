import contextlib
import os
import typing
from typing import Any, Dict, List, Optional, Union
from datetime import datetime as dt
import logging
import multiprocessing
import time
import psutil
import re
from coolname import generate_slug
import random
from duckdb import ConstraintException
from xetrack.stats import Stats
from xetrack.connection import DuckDBConnection
from xetrack.config import CONSTANTS, SCHEMA_PARAMS, TRACKER_CONSTANTS
from xetrack.logging import Logger
from xetrack.assets import AssetsManager

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method('fork')

default_logger = logging.getLogger(__name__)
ALPHANUMERIC = re.compile(r'[^a-zA-Z0-9_]')


class Tracker(DuckDBConnection):
    """
    Tracker class for tracking experiments, benchmarks, and other events.
    You can set params which are always attached to every event, and then track any parameters you want.
    """
    IN_MEMROY: str = ':memory:'
    SKIP_INSERT: str = 'SKIP_INSERT'

    def __init__(self, db: str = 'track.db',
                 params=None,
                 reset: bool = False,
                 log_system_params: bool = True,
                 log_network_params: bool = True,
                 raise_on_error: bool = True,
                 measurement_interval: float = 1,
                 track_id: Optional[str] = None,
                 logs_path: Optional[str] = None,
                 logs_file_format: Optional[str] = None,
                 logs_stdout: bool = False,
                 compress: bool = False
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
        :param track_id: The track id to use. If None, a random track id will be generated.
        :param logs_path: The path to the logs directory. If given, logs will be printed to that file.
        :param logs_file_format: The format of the logs file. Default is "{time:YYYY-MM-DD}.log" (daily). Only apply if logs_path is given.
        :param logs_stdout: If True, logs will be printed to stdout. Default is False.
        """
        self.skip_insert = False
        if db == Tracker.SKIP_INSERT:
            self.skip_insert = True
            db = Tracker.IN_MEMROY
        super().__init__(db=db)
        if params is None:
            params = {}
        self.params = params
        self.track_id = track_id or self.generate_track_id()
        self.logger = self._build_logger(
            logs_stdout, logs_path, logs_file_format)
        self._columns = set()
        self._create_events_table(reset=reset)
        self.log_system_params = log_system_params
        self.log_network_params = log_network_params
        self.raise_on_error = raise_on_error
        self.measurement_interval = measurement_interval
        self.latest = {}
        self.assets = AssetsManager(
            path=db, compress=compress, autocommit=True)

    def _build_logger(self, stdout: bool = False, logs_path: Optional[str] = None, logs_file_format: Optional[str] = None) -> Optional[Logger]:
        """
        Builds a logger object.
        """
        if stdout or logs_path:
            return Logger(
                stdout=stdout, logs_path=logs_path, file_format=logs_file_format)
        return None

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
            self.conn.execute(f"DROP TABLE IF EXISTS {SCHEMA_PARAMS.TABLE}")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {SCHEMA_PARAMS.TABLE} ("
            f"{TRACKER_CONSTANTS.TIMESTAMP} VARCHAR, "
            f"{SCHEMA_PARAMS.TRACK_ID} VARCHAR,  "
            f"PRIMARY KEY ({TRACKER_CONSTANTS.TIMESTAMP}, {SCHEMA_PARAMS.TRACK_ID}))")
        self.conn.execute("SHOW TABLES").fetchall()
        self._columns = set(self.dtypes.keys())
        for key, value in self.params.items():
            self.add_column(key, value)
        self._columns = set(self.dtypes.keys())

    @staticmethod
    def generate_track_id():
        return f'{generate_slug(2)}-{str(random.randint(0, 9999)).zfill(4)}'

    def _drop_table(self):
        return self.conn.execute(f"DROP TABLE {SCHEMA_PARAMS.TABLE}")

    @property
    def _len(self) -> int:
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {SCHEMA_PARAMS.TABLE} WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}'").fetchall()[
            0][0]

    def __len__(self):
        return self._len

    @staticmethod
    def get_timestamp():
        return dt.now().strftime(CONSTANTS.TIMESTAMP_FORMAT)

    def get(self, key: str):
        return self.assets.get(key)

    def log_batch(self, data=List[dict]) -> Dict[str, Any]:
        """
        Track a batch of data.

        Parameters:
            data (List[dict]): The list of dictionaries containing the data to be tracked.

        Returns:
            List[dict]: The updated list of dictionaries after tracking the data.

        This function takes a batch of data in the form of a list of dictionaries and tracks it. If the data is empty, a warning message is logged. The function then iterates over each event in the data and performs the following steps:
        1. Validates the event data.
        2. Converts the event data into keys, values, and size.
        3. Attempts to insert the event data into the database table.
        4. Logs the event data.
        Finally, the function commits the transaction and updates the latest event data.

        """
        if not data:
            if self.logger:
                self.logger.warning('No values to track')
            return {}
        event_data = {}

        # first we commit the assets
        raw = []
        for event in data:  # type: ignore
            event_data = self._validate_data(event)
            raw.append(self._to_key_values(event_data))
            if self.logger:
                self.logger.track(event_data)

        self.conn.execute("BEGIN TRANSACTION")
        for keys, values, size in raw:  # type: ignore
            with contextlib.suppress(ConstraintException):
                self.conn.execute(
                    f"INSERT INTO {SCHEMA_PARAMS.TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                    values)
        self.conn.execute("COMMIT TRANSACTION")
        self.latest = event_data
        return event_data

    def remove_asset(self, hash_value: str, column: Optional[str]):
        """Removes the asset stored in the database with the given hash. If a column is given, the value of that column will be set to None
        Args:
            hash_value (str): The hash of the asset to remove
            column (Optional[str], optional): The column to set to None. Defaults to None.

        Note:
            A model removed is deleted from all runs - model assets are global
        """
        if self.assets.remove_hash(hash_value, remove_keys=True):
            if column:
                SQL = f"""UPDATE {SCHEMA_PARAMS.TABLE} SET {column} = NULL WHERE {column} = '{hash_value}'"""
                self.conn.execute(SQL)
            return True
        return False

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
            for value in (TRACKER_CONSTANTS.FUNCTION_TIME, TRACKER_CONSTANTS.FUNCTION_RESULT, TRACKER_CONSTANTS.FUNCTION_NAME):
                if value in result and self.logger:  # we don't want to override the time
                    self.logger.warning(
                        f"Overriding {value}={result[value]} -> {data[value]}")
            data |= result
        else:
            if not self._is_primitive(result):
                result = str(result)
            data[TRACKER_CONSTANTS.FUNCTION_RESULT] = result
        return data

    def _run_func(self, func: callable, *args, **kwargs):  # type: ignore
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

    def wrap(self, params: Optional[dict] = None):
        """
        Generates a decorator that tracks the execution of a function.

        Args:
            params (dict, optional): Additional parameters to pass to the tracker. Defaults to {}.

        Returns:
            TrackDecorator: A decorator object that tracks the execution of a function.
        """
        if params is None:
            params = {}
        parent_tracker = self

        class TrackDecorator:
            def __init__(self, func):
                self.func = func
                self.params = params
                self.tracker = parent_tracker

            def __call__(self, *args, **kwargs):
                return self.tracker.track(self.func, params=self.params, args=args, kwargs=kwargs)

        return TrackDecorator

    def track(self, func: callable, params: Optional[dict] = None,
              args: Optional[list] = None,
              kwargs: Optional[dict] = None):
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
            stats = Stats(process, stats=manager.dict(),
                          interval=self.measurement_interval)
            stats_process = multiprocessing.Process(
                target=stats.collect_stats, args=(stop_event,))
            stats_process.start()
        data, result, exception = self._run_func(func, *args, **kwargs)
        data.update(params)
        if self.log_system_params:
            stop_event.set()  # type: ignore
            data.update(stats.get_average_stats())  # type: ignore
        if self.log_network_params:
            bytes_sent, bytes_recv = self._to_send_recv(
                net_io_before, psutil.net_io_counters())  # type: ignore
            data.update({'bytes_sent': bytes_sent, 'bytes_recv': bytes_recv})
        self.log(data)
        if exception is not None and self.raise_on_error:
            raise exception
        return result

    def add_column(self, key, value):
        dtype = self.to_sql_type(value)
        if key not in self._columns:
            self.conn.execute(
                f"ALTER TABLE {SCHEMA_PARAMS.TABLE} ADD COLUMN {key} {dtype}")
            self._columns.add(key)
        elif self.logger:
            self.logger.warning(f'Column {key} already exists')
        return value

    def _validate_key(self, key):
        is_number = isinstance(key, (int, float))
        key = str(key)
        key = ALPHANUMERIC.sub('_', key)

        if ALPHANUMERIC.match(key[0]) is not None:
            key = f'_{key}'

        # Truncate the name if needed
        max_length = 64
        key = key[:max_length]
        if is_number:
            key = f'"{key}"'
        return key

    def _validate_asset(self, key, value: Any) -> str:
        if self._is_primitive(value) or isinstance(value, (list, dict)):
            return value  # type: ignore
        return self.assets.insert(key, value)

    def _validate_data(self, data: Dict[Union[str, int, float, bool], Any]) -> Dict[str, Any]:
        if TRACKER_CONSTANTS.TIMESTAMP in data and self.logger:
            self.logger.warning(
                f"Overriding {TRACKER_CONSTANTS.TIMESTAMP} - please use another key")
        if SCHEMA_PARAMS.TRACK_ID in data and self.logger:
            self.logger.warning(
                f"Overriding {SCHEMA_PARAMS.TRACK_ID} - please use another key")
        data = {self._validate_key(key): self._validate_asset(
            key, value) for key, value in data.items()}
        new_columns = set(data.keys()) - self._columns
        for key in new_columns:
            self.conn.execute(
                f"ALTER TABLE {SCHEMA_PARAMS.TABLE} ADD COLUMN {key} {self.to_sql_type(data[key])}")
            self._columns.add(key)

        for key, value in self.params.items():
            if key not in data:
                data[key] = value
            elif self.logger:
                self.logger.warning(f'Overriding the {key} parameter')
        data[TRACKER_CONSTANTS.TIMESTAMP] = self.get_timestamp()
        data[SCHEMA_PARAMS.TRACK_ID] = self.track_id
        return data

    def _to_key_values(self, data: dict):
        keys, values, i = [], [], 0
        for i, (key, value) in enumerate(data.items()):
            keys.append(key)
            values.append(value)
        return keys, values, i + 1

    def log(self, data: dict) -> Dict[str, Any]:
        data_size = len(data)
        if data_size == 0:
            if self.logger:
                self.logger.warning('No values to track')
            return {}
        data = self._validate_data(data)  # type: ignore
        keys, values, size = self._to_key_values(data)
        if not self.skip_insert:
            self.conn.execute(
                f"INSERT INTO {SCHEMA_PARAMS.TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                list(values))
        if self.logger:
            self.logger.track(data)
        self.latest = data
        return data  # type: ignore

    def __repr__(self):
        return self.head().__repr__()

    def to_df(self, all: bool = False):
        query = f"SELECT * FROM {SCHEMA_PARAMS.TABLE}"
        if not all:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}'"
        return self.conn.execute(query).df()

    def __getitem__(self, item) -> Any:
        if isinstance(item, str):
            return self.conn.execute(f"SELECT {item} FROM {SCHEMA_PARAMS.TABLE} WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}'").df()[
                item]
        elif isinstance(item, int):
            results = self.conn.execute(
                f"SELECT * FROM {SCHEMA_PARAMS.TABLE} LIMIT 1 OFFSET {item-1}").df()
            return None if len(results) == 0 else results.iloc[0].to_dict()
        raise ValueError(f"Invalid type: {type(item)}")

    def set_params(self, params: dict):
        for key, value in params.items():
            self.set_param(key, value)

    def set_value(self, key, value):
        if key not in self._columns:
            self.add_column(key, value)
        self.conn.execute(
            f"UPDATE {SCHEMA_PARAMS.TABLE} SET {key} = '{value}' WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}'")

    def set_param(self, key, value):
        self.params[key] = value
        self.add_column(key, value)
        return key

    def head(self, n: int = 5):
        return self.conn.execute(
            f"SELECT * FROM {SCHEMA_PARAMS.TABLE} WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}' LIMIT {n}").df()

    def tail(self, n: int = 5):
        return self.conn.execute(
            f"SELECT * FROM {SCHEMA_PARAMS.TABLE} WHERE {SCHEMA_PARAMS.TRACK_ID} = '{self.track_id}' ORDER BY {TRACKER_CONSTANTS.TIMESTAMP} DESC LIMIT {n}").df()

    def count_all(self):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {SCHEMA_PARAMS.TABLE}").fetchall()[
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
