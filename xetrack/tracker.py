import contextlib
import os
import typing
from typing import Any, Dict, List, Optional, Union, Literal, TypeVar
from datetime import datetime as dt
import logging
import multiprocessing
import time
import psutil
import re
from coolname import generate_slug
import random

from xetrack.stats import Stats
from xetrack.engine import Engine, SqliteEngine
from xetrack.config import CONSTANTS, SCHEMA_PARAMS, TRACKER_CONSTANTS, LOGURU_PARAMS
from xetrack.logging import Logger
from xetrack.git import get_commit_hash

ConnT = TypeVar('ConnT')

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("fork")

default_logger = logging.getLogger(__name__)
ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9_]")


class Tracker:
    """
    Tracker class for tracking experiments, benchmarks, and other events.
    You can set params which are always attached to every event, and then track any parameters you want.
    """

    IN_MEMORY: str = CONSTANTS.IN_MEMORY_DB
    SKIP_INSERT: str = CONSTANTS.SKIP_INSERT
    
    def __init__(
        self,
        db: str = "track.db",
        params: Dict[str, Any] | None = None,
        reset: bool = False,
        log_system_params: bool = False,
        log_network_params: bool = False,
        raise_on_error: bool = True,
        measurement_interval: float = 1,
        track_id: Optional[str] = None,
        logs_path: Optional[str] = None,
        logs_file_format: Optional[str] = None,
        logs_stdout: bool = False,
        compress: bool = False,
        warnings: bool = True,
        git_root: Optional[str] = None,
        engine: Literal["duckdb", "sqlite"] = "sqlite",
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
    ):
        """
        Initializes the class instance.

        :param db: The database file to use or ":memory:" for in-memory database.
                   Default is "track.db".
        :param params: A dictionary of default parameters to attach to every event. This can be changed later.
        :param reset: If True, the events table will be dropped and recreated. Default is False.
        :param log_system_params: If True, system parameters will be logged. Default is False.
        :param log_network_params: If True, network parameters will be logged. Default is False.
        :param raise_on_error: If True, an error will be raised on failure. Default is True.
        :param measurement_interval: The interval in seconds at which to measure events. Default is 1.
        :param track_id: The track id to use. If None, a random track id will be generated.
        :param logs_path: The path to the logs directory. If given, logs will be printed to that file.
        :param logs_file_format: The format of the logs file. Default is "{time:YYYY-MM-DD}.log" (daily). Only apply if logs_path is given.
        :param logs_stdout: If True, logs will be printed to stdout. Default is False.
        :param compress: If True, the database will be compressed. Default is False.
        :param warnings: If True, warnings will be printed. Default is True.
        :param git_root: Directory to use for git operations. If provided, git commit hash will be tracked.
        :param engine: Database engine to use, either "duckdb" or "sqlite". Default is "sqlite".
        :param table_name: Name of the table to store events. Default is "default". Allows multiple experiment types.
        """
        self.skip_insert = False
        if db == Tracker.SKIP_INSERT:
            self.skip_insert = True
            db = Tracker.IN_MEMORY
        
        self.engine = self._get_engine(db, engine, compress, table_name)
            
        if params is None:
            params = {}
        self.params = params
        self.track_id = track_id or self.generate_track_id()
        self.logger = self._build_logger(logs_stdout, logs_path, logs_file_format)
        self.warnings = warnings and self.logger is not None
        self._columns = set()
        self._create_events_table(reset=reset)
        self.log_system_params = log_system_params
        self.log_network_params = log_network_params
        self.raise_on_error = raise_on_error
        self.measurement_interval = measurement_interval
        self.latest = {}
        self.git_root = git_root
        if git_root:
            self.set_param(
                TRACKER_CONSTANTS.GIT_COMMIT_KEY, get_commit_hash(git_root=git_root)
            )

    def _get_engine(
        self,
        db: str,
        engine: Literal["duckdb", "sqlite"],
        compress: bool = False,
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
    ) -> Engine[Any]:
        """
        Create and return the appropriate database connection implementation.
        
        Args:
            db: Database file path
            engine: Database engine, either 'duckdb' or 'sqlite'
            compress: Whether to compress the database
            table_name: Name of the table to store events
            
        Returns:
            A Connection implementation
            
        Raises:
            ImportError: If the duckdb engine is requested but not installed
        """
        if engine == "duckdb":
            try:
                from xetrack.duckdb import DuckDBEngine
                return DuckDBEngine(db=db, compress=compress, table_name=table_name)
            except ImportError:
                raise ImportError("DuckDB is not installed. Please install it with 'pip install duckdb'")
                
        return SqliteEngine(db=db, compress=compress, table_name=table_name)
    
    @property
    def conn(self):
        return self.engine.conn
        
    @property
    def db(self):
        return self.engine.db
        
    @property
    def table_name(self):
        return self.engine.table_name
        
    @property
    def assets(self):
        return self.engine.assets
        
    @property
    def dtypes(self):
        return self.engine.dtypes
        
    @property
    def columns(self):
        return self.engine.columns

    def to_sql_type(self, value: Any) -> str:
        return self.engine.to_sql_type(value)
        
    def _insert_raw(self, data: List[tuple[list[str], list[Any], int]]):
        return self.engine._insert_raw(data)
        
    def remove_asset(self, hash_value: str, column: Optional[str], remove_keys: bool = True) -> bool:
        return self.engine.remove_asset(hash_value, column, remove_keys)

    def _build_logger(
        self,
        stdout: bool = False,
        logs_path: Optional[str] = None,
        logs_file_format: Optional[str] = None,
    ) -> Optional[Logger]:
        """
        Builds a logger object.
        """
        if stdout or logs_path:
            return Logger(
                stdout=stdout, logs_path=logs_path, file_format=logs_file_format
            )
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
        Creates or resets the events table in the database.

        Parameters:
            reset (bool): If True, drops the table if it already exists and recreates it. Defaults to False.

        Returns:
            None
        """
        if reset:
            # Only drop the table if it exists and we're resetting
            # Use the engine's execute method which handles table name quoting
            self.engine.execute(f"DROP TABLE IF EXISTS {self.engine.table_name}")
            
            # Force table recreation by initializing the database structure
            self.engine._init_database()
        
        # The table is automatically created during connection initialization
        # if it doesn't already exist, so we just need to add any additional columns
        # for our parameters
        
        # Update column information
        self._columns = set(self.dtypes.keys())
        
        # Add columns for all parameters
        for key, value in self.params.items():
            self.add_column(key, value)
            
        # Refresh column information
        self._columns = set(self.dtypes.keys())

    @staticmethod
    def generate_track_id():
        return f"{generate_slug(2)}-{str(random.randint(0, 9999)).zfill(4)}"

    def _drop_table(self):
        return self.engine.execute(f"DROP TABLE IF EXISTS {SCHEMA_PARAMS.DUCKDB_TABLE}")

    @property
    def _len(self) -> int:
        """
        Get the number of records for this track_id.
        
        Returns:
            Number of records for this track_id
        """
        return self.engine.count_records(self.track_id)

    def __len__(self):
        return self._len

    @staticmethod
    def get_timestamp():
        return dt.now().strftime(CONSTANTS.TIMESTAMP_FORMAT)

    def get(self, key: str):
        """
        Get an asset by key.
        
        Args:
            key: The key of the asset to retrieve
            
        Returns:
            The asset if found, None otherwise
        """
        if self.assets is None:
            if self.warnings:
                self.logger.warning("Assets functionality disabled: sqlitedict not installed")
            return None
        return self.assets.get(key)

    def log_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track a batch of data.

        Parameters:
            data (List[dict]): The list of dictionaries containing the data to be tracked.

        Returns:
            Dict[str, Any]: The updated dictionary after tracking the data.
        """
        if not data:
            if self.warnings:
                self.logger.warning("No values to track")
            return {}
        event_data = {}

        # first we commit the assets
        raw: List[tuple[list[str], list[Any], int]] = []
        for event in data:  # type: ignore
            event_data = self._validate_data(event)
            raw.append(self._to_key_values(event_data))
            if self.logger:
                self.logger.track(event_data)

        self._insert_raw(raw)
        self.latest = event_data
        return event_data

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

    def validate_result(self, data: Dict[str, Any], result: typing.Union[Dict[str, Any], Any]):
        """
        Validate the result of a function execution.

        Args:
            data (dict): The original data dictionary.
            result (Union[dict, Any]): The result of the function execution.

        Returns:
            dict: The updated data dictionary.
        """
        if isinstance(result, dict):
            for value in (
                TRACKER_CONSTANTS.FUNCTION_TIME,
                TRACKER_CONSTANTS.FUNCTION_RESULT,
                TRACKER_CONSTANTS.FUNCTION_NAME,
            ):
                if (
                    value in result and self.warnings
                ):  # we don't want to override the time
                    self.logger.warning(
                        f"Overriding {value}={result[value]} -> {data[value]}"
                    )
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
        func_name, error, exception, result = func.__name__, "", None, None
        if self.logger:
            self.logger.debug(f"Running {func_name} with {args} and {kwargs}")
        start_func_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
            error = str(e)
        func_time = time.time() - start_func_time
        data = {
            "function_name": func_name,
            "args": str(list(args)),
            "kwargs": str(kwargs),
            "error": error,
            "function_time": func_time,
        }
        data = self.validate_result(data, result)
        return data, result, exception

    def wrap(self, params: Optional[Dict[str, Any]] = None):
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
            def __init__(self, func: typing.Callable):
                self.func = func
                self.params = params
                self.tracker = parent_tracker

            def __call__(self, *args: Any, **kwargs: Any):
                return self.tracker.track(
                    self.func, params=self.params, args=list(args), kwargs=kwargs
                )

        return TrackDecorator

    def track(
        self,
        func: typing.Callable,
        params: Optional[Dict[str, Any]] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
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
            stats = Stats(
                process, stats=manager.dict(), interval=self.measurement_interval
            )
            stats_process = multiprocessing.Process(
                target=stats.collect_stats, args=(stop_event,)
            )
            stats_process.start()
        data, result, exception = self._run_func(func, *args, **kwargs)
        data.update(params)
        if self.log_system_params:
            stop_event.set()  # type: ignore
            data.update(stats.get_average_stats())  # type: ignore
        if self.log_network_params:
            bytes_sent, bytes_recv = self._to_send_recv(
                net_io_before, psutil.net_io_counters() # type: ignore
            )  
            data.update({"bytes_sent": bytes_sent, "bytes_recv": bytes_recv})
        self.log(data)
        if exception is not None and self.raise_on_error:
            raise exception
        return result

    def add_column(self, key: str, value: Any):
        """
        Add a column to the events table.
        
        Args:
            key: The name of the column.
            value: The value of the column, used to determine its type.
            
        Returns:
            The value that was passed in.
        """
        dtype = self.to_sql_type(value)
        if key not in self._columns:
            # Use the engine's add_column method instead of direct SQL execution
            # This ensures proper handling of table names for each database engine
            self.engine.add_column(self.engine.table_name, key, dtype)
            self._columns.add(key)
        elif self.warnings and self.db != Tracker.IN_MEMORY:
            self.logger.warning(f"Column {key} already exists")
        return value

    def set_value(self, key: str, value: Any):
        """
        Sets the value of a specific key in the database table for all records with this track_id.
        
        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
        """
        if key not in self._columns:
            self.add_column(key, value)
                
        self.engine.set_value(key, value, track_id=self.track_id)
        
    def set_where(
        self,
        key: str,
        value: Any,
        where_key: str,
        where_value: Any,
    ):
        """
        Sets the value of a specific key in the database table where another column matches a value.
        Limited to records with this track_id.
        
        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
            where_key: The column to match for the WHERE clause.
            where_value: The value to match in the WHERE clause.
        """
        self.engine.set_where(key, value, where_key, where_value, track_id=self.track_id)
        
    def set_param(self, key: str, value: Any) -> str:
        """
        Set a parameter that will be included in all subsequent tracked events.
        
        Args:
            key: The parameter name.
            value: The parameter value.
            
        Returns:
            The parameter key that was set.
        """
        self.params[key] = value
        self.add_column(key, value)
        return key

    def _validate_key(self, key: Union[str, int, float, bool]) -> str:
        """
        Validate and normalize a key name for use in the database.
        
        Args:
            key: The key to validate, can be a string, number, or boolean.
            
        Returns:
            A valid string key name for the database.
        """
        is_number = isinstance(key, (int, float))
        key = str(key)
        key = ALPHANUMERIC.sub("_", key)

        if ALPHANUMERIC.match(key[0]) is not None:
            key = f"_{key}"

        # Truncate the name if needed
        max_length = 64
        key = key[:max_length]
        if is_number:
            key = f'"{key}"'
        return key

    def _validate_asset(self, key, value: Any) -> str:
        if self._is_primitive(value) or isinstance(value, (list, dict)):
            return value  # type: ignore
        
        if self.assets is None:
            if self.warnings:
                self.logger.warning("Assets functionality disabled: sqlitedict not installed")
            return str(value)
            
        return self.assets.insert(key, value)

    def _validate_data(
        self, data: Dict[Union[str, int, float, bool], Any]
    ) -> Dict[str, Any]:
        """
        Validates the input data, ensuring all keys are valid and all values are
        serializable.

        Args:
            data: The data to validate.

        Returns:
            The validated data, with all keys converted to strings.
        """
        if self.skip_insert:
            if self.logger:
                # When skipping insert, add only the specific data items
                # plus track_id and timestamp to avoid duplicates
                log_data = {}
                for key, value in data.items():
                    valid_key = self._validate_key(key)
                    # Convert complex types to strings for logging
                    if isinstance(value, (dict, list, tuple)):
                        log_data[valid_key] = str(value)
                    elif self._is_primitive(value):
                        log_data[valid_key] = value
                    else:
                        # For complex objects, just use a string representation
                        log_data[valid_key] = str(value)
                
                # Add required tracking fields
                log_data[SCHEMA_PARAMS.TRACK_ID] = self.track_id
                log_data[TRACKER_CONSTANTS.TIMESTAMP] = self.get_timestamp()
                
                # Pass the data directly to the log method
                self.logger.log(level=LOGURU_PARAMS.TRACKING, data=log_data)
            return {}

        if not data:
            if self.warnings:
                default_logger.warning("No data provided")
            return {
                SCHEMA_PARAMS.TRACK_ID: self.track_id,
                TRACKER_CONSTANTS.TIMESTAMP: self.get_timestamp()
            }

        # Include default params
        result = self.params.copy()

        # Process each key-value pair
        for key, value in data.items():
            valid_key = self._validate_key(key)
                
            # Add the column if it doesn't exist
            self.add_column(valid_key, value)

            # Store the value
            if self._is_primitive(value):
                result[valid_key] = value
            elif isinstance(value, (dict, list, tuple)):
                # Convert complex data structures to strings
                result[valid_key] = str(value)
            else:
                # Handle complex objects by converting to string/asset
                asset_hash = self._validate_asset(valid_key, value)
                result[valid_key] = asset_hash

        # Add the timestamp and track ID
        result[TRACKER_CONSTANTS.TIMESTAMP] = self.get_timestamp()
        result[SCHEMA_PARAMS.TRACK_ID] = self.track_id

        # Apply system and network params if enabled
        if self.log_system_params or self.log_network_params:
            self._apply_system_params(result)

        return result

    def _apply_system_params(self, data: Dict[str, Any]) -> None:
        """
        Apply system and network parameters to the data dictionary if enabled.
        
        Args:
            data: The data dictionary to update with system parameters
        """
        if not (self.log_system_params or self.log_network_params):
            return
            
        # Add system parameters if enabled
        if self.log_system_params:
            # Add CPU and memory info columns first
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory()
            process = psutil.Process()
            disk_usage = psutil.disk_usage('/')
            
            # Ensure columns exist
            self.add_column("cpu_percent", cpu_percent)
            self.add_column("ram_percent", memory_usage.percent)
            self.add_column("p_memory_percent", process.memory_percent())
            self.add_column("disk_percent", disk_usage.percent)
            
            # Add stats to data
            data["cpu_percent"] = cpu_percent
            data["ram_percent"] = memory_usage.percent
            data["p_memory_percent"] = process.memory_percent()
            data["disk_percent"] = disk_usage.percent
            
        # Add network parameters if enabled
        if self.log_network_params:
            # Get network stats
            net_io = psutil.net_io_counters()
            
            # Ensure columns exist
            self.add_column("bytes_sent", net_io.bytes_sent)
            self.add_column("bytes_recv", net_io.bytes_recv)
            
            # Add to data dict
            data["bytes_sent"] = net_io.bytes_sent
            data["bytes_recv"] = net_io.bytes_recv

    def _to_key_values(self, data: dict) -> tuple[list[str], list[Any], int]:
        keys, values, i = [], [], 0
        for i, (key, value) in enumerate(data.items()):
            keys.append(key)
            values.append(value)
        return keys, values, i + 1

    def log(self, data: dict) -> Dict[str, Any]:
        data_size = len(data)
        if data_size == 0:
            if self.warnings:
                self.logger.warning("No values to track")
            return {}
        data = self._validate_data(data)  # type: ignore
        keys, values, size = self._to_key_values(data)
        if not self.skip_insert:
            # Use the engine's execute method to handle table names correctly
            placeholders = ', '.join(['?' for _ in range(size)])
            query = f"INSERT INTO {self.engine.table_name} ({', '.join(keys)}) VALUES ({placeholders})"
            self.engine.execute(query, values)
        if self.logger:
            self.logger.track(data)
        self.latest = data
        return data  # type: ignore

    def __repr__(self):
        return self.head().__repr__()

    def to_df(self, all: bool = False):
        query = f"SELECT * FROM {self.engine.table_name}"
        if not all:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            cursor = self.engine.execute(query, [self.track_id])
        else:
            cursor = self.engine.execute(query)
            
        # For SQLite connections, convert cursor results to pandas DataFrame
        # since SQLite cursor doesn't have a df() method like DuckDB
        if hasattr(cursor, 'df'):
            return cursor.df()
        else:
            import pandas as pd
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else []
            # Fetch all rows
            rows = cursor.fetchall()
            # Convert to DataFrame
            return pd.DataFrame.from_records(rows, columns=columns)

    def __getitem__(self, item: Union[str, int]) -> Any:
        if isinstance(item, str):
            query = f"SELECT {item} FROM {self.engine.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            cursor = self.engine.execute(query, [self.track_id])
            
            # Convert cursor results to a DataFrame
            if hasattr(cursor, 'df'):
                return cursor.df()[item]
            else:
                import pandas as pd
                # Get column names from cursor description
                columns = [col[0] for col in cursor.description] if cursor.description else []
                # Fetch all rows
                rows = cursor.fetchall()
                # Convert to DataFrame
                df = pd.DataFrame.from_records(rows, columns=columns)
                return df[item]
            
        elif isinstance(item, int):
            query = f"SELECT * FROM {self.engine.table_name} LIMIT 1 OFFSET ?"
            cursor = self.engine.execute(query, [item-1])
            
            # Convert cursor results to a DataFrame
            if hasattr(cursor, 'df'):
                results = cursor.df()
            else:
                import pandas as pd
                # Get column names from cursor description
                columns = [col[0] for col in cursor.description] if cursor.description else []
                # Fetch all rows
                rows = cursor.fetchall()
                # Convert to DataFrame
                results = pd.DataFrame.from_records(rows, columns=columns)
            
            return None if len(results) == 0 else results.iloc[0].to_dict()
        raise ValueError(f"Invalid type: {type(item)}")

    def set_params(self, params: Dict[str, Any]):
        """
        Set multiple parameters at once.
        
        Args:
            params: Dictionary of parameter names and values to set.
        """
        for key, value in params.items():
            self.set_param(key, value)

    def head(self, n: int = 5):
        query = f"SELECT * FROM {self.engine.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ? LIMIT ?"
        cursor = self.engine.execute(query, [self.track_id, n])
        
        # Convert cursor results to a DataFrame
        if hasattr(cursor, 'df'):
            return cursor.df()
        else:
            import pandas as pd
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else []
            # Fetch all rows
            rows = cursor.fetchall()
            # Convert to DataFrame
            return pd.DataFrame.from_records(rows, columns=columns)

    def tail(self, n: int = 5):
        query = f"SELECT * FROM {self.engine.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ? ORDER BY {TRACKER_CONSTANTS.TIMESTAMP} DESC LIMIT ?"
        cursor = self.engine.execute(query, [self.track_id, n])
        
        # Convert cursor results to a DataFrame
        if hasattr(cursor, 'df'):
            return cursor.df()
        else:
            import pandas as pd
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else []
            # Fetch all rows
            rows = cursor.fetchall()
            # Convert to DataFrame
            return pd.DataFrame.from_records(rows, columns=columns)

    def count_all(self):
        """
        Count all records in the events table.
        
        Returns:
            Total number of records in the events table
        """
        return self.engine.count_records()

    def count_run(self):
        """
        Count records for this track_id.
        
        Returns:
            Number of records for this track_id
        """
        return self._len

    def __del__(self):
        if hasattr(self, "engine") and hasattr(self.engine, "conn"):
            self.engine.conn.close()

    def to_csv(self, path: str, all: bool = False):
        return self.to_df(all).to_csv(path, index=False)

    def to_parquet(self, path: str, all: bool = False):
        return self.to_df(all).to_parquet(path, index=False)
