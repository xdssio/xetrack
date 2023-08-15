import typing
from typing import List
from uuid import uuid4
from datetime import datetime as dt
import duckdb
import pandas as pd
import logging

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
                 table_name: str = EVENTS,
                 params=None,
                 reset: bool = False,
                 verbose: bool = True):
        """
        :param db: The duckdb database file to use or ":memory:" for in-memory database - default is "tracker.db"
        :param table_name: a table name to use for the events table - default is "events"
        :param params: A dictionary of default parameters to attach to every event - this can be changed later
        :param reset: If True, the events table will be dropped and recreated - default is False
        :param verbose: If True, log messages will be printed - default is True
        """
        if params is None:
            params = {}
        self.db = db
        self.table_name = table_name
        self.params = params
        self.verbose = verbose
        self._columns = set()
        self.track_id = self.generate_uuid4()
        self.conn = duckdb.connect(self.db)
        self._create_events_table(reset=reset)

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

    def __len__(self):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}'").fetchall()[0][0]

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
        return size

    def __repr__(self):
        return self.head().__repr__()

    def to_df(self, all: bool = False):
        if all:
            return self._table.to_df()
        return self.run()

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

    def __setitem__(self, key, value):
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

    def run(self):
        results = self.conn.execute(
            f"SELECT * FROM {self.table_name} WHERE {Tracker.TRACK_ID} = '{self.track_id}'").fetchall()
        return pd.DataFrame(results, columns=self._table.columns)

    def all(self):
        return self._table.to_df()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def to_csv(self, path: str, all: bool = False):
        return self.to_df(all).to_csv(path, index=False)

    def to_parquet(self, path: str, all: bool = False):
        return self.to_df(all).to_parquet(path, index=False)
