import contextlib
import os
import subprocess
from typing import List
from uuid import uuid4
from datetime import datetime as dt
import duckdb
import pandas as pd
import logging
import multiprocessing
import time
import psutil

from xetrack.stats import Stats

_DTYPES_TO_PYTHON = {
    'BOOLEAN': bool,
    'TINYINT': int,
    'SMALLINT': int,
    'INTEGER': int,
    'BIGINT': int,
    'FLOAT': float,
    'DOUBLE': float,
    'VARCHAR': str,
    'CHAR': str,
    'BLOB': bytearray,
    'DATE': str,
    'TIME': str,
    'TIMESTAMP': str,
    'DECIMAL': float,
    'INTERVAL': str,
    'UUID': str
}
multiprocessing.set_start_method('fork')
logger = logging.getLogger(__name__)
EVENTS = 'events'


class Tracker:
    """
    Tracker class for tracking experiments, benchmarks, and other events.
    You can set params which are always attached to every event, and then track any parameters you want.
    """
    ID = '_id'
    TRACK_ID = 'track_id'
    DATE = 'date'

    def __init__(self, db: str = 'track.db',
                 params=None,
                 reset: bool = False,
                 verbose: bool = True,
                 log_system_params: bool = True,
                 log_network_params: bool = True,
                 raise_on_error: bool = True,
                 measurement_interval: float = 1,
                 ):
        """
        :param db: The duckdb database file to use or ":memory:" for in-memory database - default is "tracker.db"
        :param params: A dictionary of default parameters to attach to every event - this can be changed later
        :param reset: If True, the events table will be dropped and recreated - default is False
        :param verbose: If True, log messages will be printed - default is True
        """
        if params is None:
            params = {}
        self.db = db
        self.table_name = EVENTS
        self.params = params
        self.verbose = verbose
        self._columns = set()
        self.track_id = self.generate_uuid4()
        self.conn = duckdb.connect(self.db)
        self._create_events_table(reset=reset)
        self.log_system_params = log_system_params
        self.log_network_params = log_network_params
        self.raise_on_error = raise_on_error
        self.measurement_interval = measurement_interval

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

    @property
    def dtypes(self):
        return {column[0]: _DTYPES_TO_PYTHON.get(column[1]) for column in
                self.conn.execute(f"DESCRIBE {self.table_name}").fetchall()}

    def _create_events_table(self, reset: bool = False):
        if reset:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ({Tracker.ID} VARCHAR PRIMARY KEY,{Tracker.TRACK_ID} VARCHAR, {Tracker.DATE} VARCHAR)")
        self._columns = set(self.dtypes.keys())
        for key, value in self.params.items():
            self.add_column(key, value, self.to_py_type(value))
        self._columns = set(self.dtypes.keys())

    def _drop_table(self, ):
        return self.conn.execute(f"DROP TABLE {self.table_name}")

    @property
    def _len(self) -> int:
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}'").fetchall()[0][0]

    def __len__(self):
        return self._len

    @staticmethod
    def to_datetime(timestamp):
        return timestamp.strftime('%d-%m-%Y %H:%M:%S')

    @property
    def now(self):
        return dt.now()

    @property
    def _table(self):
        return self.conn.table(self.table_name)

    # TODO make more efficient
    def track_batch(self, data=List[dict]):
        if len(data) == 0:
            if self.verbose:
                logger.warning('No values to track')
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
            logger.info(f"Running {name} with {args} and {kwargs}")
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
        return data

    def add_column(self, key, value, dtype: type):
        if key not in self._columns:
            self.conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {key} {dtype}")
            self._columns.add(key)
        elif self.verbose:
            logger.warning(f'Column {key} already exists')
        return value

    def _validate_data(self, data: dict):

        if Tracker.ID in data and self.verbose:
            logger.warning('Overriding _id - please use another key')
        new_columns = set(data.keys()) - self._columns
        for key in new_columns:
            self.conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {key} {self.to_py_type(data[key])}")
            self._columns.add(key)
        data['_id'] = self.generate_uuid4()
        data['date'] = self.to_datetime(self.now)
        data['track_id'] = self.track_id
        for key, value in self.params.items():
            if key not in data:
                data[key] = value
            elif self.verbose:
                logger.warning(f'Overriding the {key} parameter')
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
                logger.warning('No values to track')
            return data_size
        data = self._validate_data(data)
        keys, values, size = self._to_key_values(data)
        self.conn.execute(
            f"INSERT INTO {self.table_name} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
            list(values))
        return data

    def __repr__(self):
        return self.head().__repr__()

    def to_df(self, all: bool = False):
        if all:
            return self._table.to_df()
        results = self.conn.execute(
            f"SELECT * FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}'").fetchall()
        return pd.DataFrame(results, columns=self._table.columns)

    def __getitem__(self, item):
        if isinstance(item, str):
            results = self.conn.execute(
                f"SELECT {item} FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}'").fetchall()
            return pd.Series([res[0] for res in results], name=item)
        elif isinstance(item, int):
            results = self.conn.execute(f"SELECT * FROM {self.table_name} WHERE _id = {item}").fetchall()
            if len(results) == 0:
                return None
            return {column: value for column, value in zip(self._table.columns, results[0])}
        raise ValueError(f"Invalid type: {type(item)}")

    def set_params(self, params: dict):
        for key, value in params.items():
            self.set_param(key, value)

    def set_value(self, key, value):
        if key not in self._columns:
            self.add_column(key, value, self.to_py_type(value))
        self.conn.execute(
            f"UPDATE {self.table_name} SET {key} = '{value}' WHERE {Tracker.TRACK_ID} = '{self.track_id}'")

    def set_param(self, key, value):
        self.params[key] = value
        self.add_column(key, value, self.to_py_type(value))
        return key

    def head(self, n: int = 5):
        results = self.conn.execute(
            f"SELECT * FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}' LIMIT {n}").fetchall()
        return pd.DataFrame(results, columns=self._table.columns)

    def tail(self, n: int = 5):
        result = self.conn.execute(
            f"SELECT * FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}' ORDER BY _id DESC LIMIT {n}").fetchall()
        return pd.DataFrame(reversed(result), columns=self._table.columns)

    def count_all(self):
        return len(self._table)

    def count_run(self):
        return self._len

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def to_csv(self, path: str, all: bool = False):
        return self.to_df(all).to_csv(path, index=False)

    def to_parquet(self, path: str, all: bool = False):
        return self.to_df(all).to_parquet(path, index=False)


def copy(source: str,
         target: str,
         handle_duplicate: str = "IGNORE",
         ):
    """
    Copies the data from one tracker to another
    :param source: The source database file
    :param target: The target database file
    :param handle_duplicate: How to handle duplicate columns - IGNORE or REPLACE 
    """
    if handle_duplicate not in ('IGNORE', 'REPLACE'):
        raise ValueError(f"Invalid handle_duplicate: {handle_duplicate} - Must be either IGNORE or REPLACE")
    source = Tracker(db=source, )
    target = Tracker(db=target, verbose=False)
    results = source.conn.execute(f"SELECT * FROM {source.table_name}").fetchall()
    if len(results) == 0:
        print('No data to copy')
        return
    new_column_count = 0
    for column, value in source.dtypes.items():
        if column not in target._columns:
            new_column_count += 1
            target.add_column(column, value, source.to_py_type(value))
    keys = source._table.columns
    size = len(keys)
    target.conn.execute("BEGIN TRANSACTION")
    for event in results:
        values = list(event)
        with contextlib.suppress(duckdb.ConstraintException):
            target.conn.execute(
                f"INSERT OR {handle_duplicate} INTO {target.table_name} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                values)
    target.conn.execute("COMMIT TRANSACTION")
    total = target.count_all()
    print(f"Copied {len(results)} events and {new_column_count} new columns. New total is {total} events")


"""
#TODO remove
# stop = count = cpu_usage_sum = ram_usage_sum = disk_usage_sum = 0
            # process = psutil.Process()
            # while stop == 0:
            #     count += 1
            #     cpu_usage_sum += process.cpu_percent(interval=self.measurement_interval)
            #     ram_usage_sum += process.memory_percent()
            #     disk_usage_sum += psutil.disk_usage('/').percent
            #     data = self._run_func(func, *args, **kwargs)
            #     if data is not None:
            #         break
            # data['cpu_usage'] = cpu_usage_sum / count
            # data['ram_usage'] = ram_usage_sum / count
            # data['disk_usage'] = disk_usage_sum / count
"""
