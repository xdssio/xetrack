import contextlib
import os
from typing import List
from uuid import uuid4
from datetime import datetime as dt
import logging
import multiprocessing
import time
import psutil

from xetrack.stats import Stats
from xetrack.connection import DuckDBConnection
from xetrack.constants import TRACK_ID, TABLE, TIMESTAMP


with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method('fork')
logger = logging.getLogger(__name__)


class Tracker(DuckDBConnection):
    """
    Tracker class for tracking experiments, benchmarks, and other events.
    You can set params which are always attached to every event, and then track any parameters you want.
    """

    def __init__(self, db: str = 'track.db',
                 params=None,
                 reset: bool = False,
                 verbose: bool = True,
                 log_system_params: bool = True,
                 log_network_params: bool = True,
                 raise_on_error: bool = True,
                 measurement_interval: float = 1,
                 logger = None
                 ):
        """
        :param db: The duckdb database file to use or ":memory:" for in-memory database - default is "tracker.db"
        :param params: A dictionary of default parameters to attach to every event - this can be changed later
        :param reset: If True, the events table will be dropped and recreated - default is False
        :param verbose: If True, log messages will be printed - default is True
        """
        super().__init__(db=db)
        if params is None:
            params = {}
        self.params = params
        self.verbose = verbose
        self.track_id = self.generate_uuid4()
        self._columns = set()
        self._create_events_table(reset=reset)
        self.log_system_params = log_system_params
        self.log_network_params = log_network_params
        self.raise_on_error = raise_on_error
        self.measurement_interval = measurement_interval
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def _create_events_table(self, reset: bool = False):
        if reset:
            self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ({TIMESTAMP} VARCHAR PRIMARY KEY, {TRACK_ID} VARCHAR)")
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



    def _drop_table(self, ):
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
        return dt.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]

    # TODO make more efficient
    def track_batch(self, data=List[dict]):
        if len(data) == 0:
            if self.verbose:
                self.logger.warning('No values to track')
            return data
        for values in data:
            self.track(**values)

    @staticmethod
    def to_mb(bytes: int):
        return bytes / (1024 * 1024)

    def _to_send_recv(self, net_io_before, net_io_after):
        sleep_bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent
        sleep_bytes_recv = net_io_after.bytes_recv - net_io_before.bytes_recv
        return self.to_mb(sleep_bytes_sent), self.to_mb(sleep_bytes_recv)

    def _run_func(self, func: callable, *args, **kwargs):
        name, error = func.__name__, ''
        if self.verbose:
            self.logger.info(f"Running {name} with {args} and {kwargs}")
        start_func_time = time.time()
        try:
            data = func(*args, **kwargs)
        except Exception as e:
            if self.raise_on_error:
                raise e
            error = f"Error running {name} with {args} and {kwargs} - {e}"
        if data is not None and type(data) != dict:
            raise ValueError(f'Function must return a dictionary or None, not {type(data)}')
        func_time = time.time() - start_func_time
        if data is None:
            data = {}
        data.update({'name': name, 'time': func_time, 'error': error})
        return data

    def track_function(self, func: callable, *args, **kwargs):
        """tracking a function which returns a dictionary or parameters to track"""
        stats_process, system_stats = None, []
        if self.log_network_params:
            net_io_before = psutil.net_io_counters()
        if self.log_system_params:
            process = psutil.Process(os.getpid())
            manager = multiprocessing.Manager()
            stop_event = multiprocessing.Event()
            stats = Stats(process, stats=manager.dict(), interval=self.measurement_interval)
            stats_process = multiprocessing.Process(target=stats.collect_stats, args=(stop_event,))
            stats_process.start()
        data = self._run_func(func, *args, **kwargs)

        if self.log_system_params:
            stop_event.set()
            data.update(stats.get_average_stats())
        if self.log_network_params:
            bytes_sent, bytes_recv = self._to_send_recv(net_io_before, psutil.net_io_counters())
            data.update({'bytes_sent': bytes_sent, 'bytes_recv': bytes_recv})
        self.track(**data)
        if self.verbose:
            self.logger.info(f"Tracked {data}")
        return data

    def add_column(self, key, value, dtype: type):
        if key not in self._columns:
            self.conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {key} {dtype}")
            self._columns.add(key)
        elif self.verbose:
            self.logger.warning(f'Column {key} already exists')
        return value

    def _validate_data(self, data: dict):

        if TIMESTAMP in data and self.verbose:
            self.logger.warning(f"Overriding {TIMESTAMP} - please use another key")
        new_columns = set(data.keys()) - self._columns
        for key in new_columns:
            self.conn.execute(
                f"ALTER TABLE {TABLE} ADD COLUMN {key} {self.to_py_type(data[key])}")
            self._columns.add(key)

        data['timestamp'] = self.get_timestamp()
        data['track_id'] = self.track_id
        for key, value in self.params.items():
            if key not in data:
                data[key] = value
            elif self.verbose:
                self.logger.warning(f'Overriding the {key} parameter')
        return data

    def _to_key_values(self, data: dict):
        keys, values = [], []
        for i, (key, value) in enumerate(data.items()):
            keys.append(key)
            values.append(value)
        return keys, values, i + 1

    def track(self, **data: dict) -> int:
        data_size = len(data)
        if data_size == 0:
            if self.verbose:
                self.logger.warning('No values to track')
            return data_size
        data = self._validate_data(data)
        keys, values, size = self._to_key_values(data)
        self.conn.execute(
            f"INSERT INTO {TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
            list(values))
        if self.verbose:
            self.logger.info(f"Tracked {data}")
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
            f"SELECT * FROM {TABLE} WHERE {TRACK_ID} = '{self.track_id}' ORDER BY {TIMESTAMP} DESC LIMIT {n}").df()

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