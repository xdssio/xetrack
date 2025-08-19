import logging
from typing import Any, Optional, List, TypeVar, Generic, Dict, Set
from xetrack.config import SCHEMA_PARAMS, CONSTANTS, DEFAULTS, TRACKER_CONSTANTS
import sqlite3
from abc import ABC, abstractmethod
import os
import pandas as pd


logger = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = ['Engine', 'SqliteEngine']


class Engine(ABC, Generic[T]):
    def __init__(
        self,
        db: str = DEFAULTS.DB,
        compress: bool = False,
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
    ):
        """
        Abstract base class for database connections.
        """
        logger.debug(f"Connecting to {db}")
        self.db = db
        self.table_name = table_name
        self.conn: T = self._init_connection()
        self.assets = None
        try:
            from xetrack.assets import AssetsManager
            self.assets = AssetsManager(path=db, compress=compress, autocommit=True)
        except ImportError:
            logger.debug("sqlitedict not installed, assets functionality will be disabled")
        self._columns: Set[str] = set()                
        self._init_database()

    @abstractmethod
    def _init_connection(self) -> T:
        """Initialize the database connection."""
        pass

    @property
    @abstractmethod
    def columns(self) -> set[str]:
        """Get the set of column names in the events table."""
        pass

    @property
    @abstractmethod
    def dtypes(self) -> dict[str, Any]:
        """Get the data types of columns in the events table."""
        pass

    @staticmethod
    @abstractmethod
    def to_sql_type(value: Any) -> str:
        """
        Convert a primitive Python value to its corresponding SQL type.

        Parameters:
            value (Any): The value to be converted.

        Returns:
            str: The SQL type corresponding to the given value.
        """
        pass
    
    @abstractmethod
    def execute(self, query: str, params: Optional[List[Any]] = None) -> Any:
        """
        Execute a raw SQL query with optional parameters.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            The result of the query execution
        """
        pass
    
    @abstractmethod
    def add_column(self, table: str, column: str, dtype: str) -> None:
        """
        Add a column to a table if it doesn't exist.
        
        Args:
            table: The table name
            column: The column name
            dtype: The SQL data type for the column
        """
        pass
    
    @abstractmethod
    def create_table(self, table: str, columns: Dict[str, str], primary_key: Optional[List[str]] = None) -> None:
        """
        Create a table if it doesn't exist.
        
        Args:
            table: The table name
            columns: Dictionary mapping column names to their SQL data types
            primary_key: Optional list of column names to use as primary key
        """
        pass

    @abstractmethod
    def set_value(self, key: str, value: Any, track_id: Optional[str] = None) -> None:
        """
        Sets the value of a specific key in the database table.

        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
            track_id: The track ID used to identify the specific record in the table.
        """
        pass

    @abstractmethod
    def set_where(
        self,
        key: str,
        value: Any,
        where_key: str,
        where_value: Any,
        track_id: Optional[str] = None,
    ) -> None:
        """
        Sets the value of a specific key in the database table given a where key value pair.
        
        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
            where_key: The column to check in the WHERE clause.
            where_value: The value to match in the WHERE clause.
            track_id: Optional track ID to restrict updates to.
        """
        pass

    @abstractmethod
    def remove_asset(
        self, hash_value: str, column: Optional[str], remove_keys: bool = True
    ) -> bool:
        """
        Removes the asset stored in the database with the given hash.
        
        Args:
            hash_value: The hash of the asset to remove.
            column: The column to set to NULL if the asset is removed.
            remove_keys: Whether to remove the keys associated with the asset.
            
        Returns:
            True if the asset was removed, False otherwise.
        """
        pass
    
    @abstractmethod
    def _insert_raw(self, data: List[tuple[list[str], list[Any], int]]) -> None:
        """
        Inserts new rows into the database table.
        
        Args:
            data: List of (keys, values, size) tuples to insert.
        """
        pass

    @abstractmethod
    def count_records(self, track_id: Optional[str] = None) -> int:
        """
        Count records in the events table, optionally filtered by track_id.
        
        Args:
            track_id: Optional track ID to filter by
            
        Returns:
            Number of records matching the criteria
        """
        pass

    @abstractmethod
    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return a pandas DataFrame with the results.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            pandas.DataFrame: A DataFrame containing the query results
        """
        pass

    def _init_database(self) -> None:
        """
        Initialize the database structure (tables, indices, etc.)
        """
        self._ensure_events_table()
    
    def _ensure_events_table(self) -> None:
        """
        Ensure the events table exists in the database.
        """
        columns = {
            SCHEMA_PARAMS.TRACK_ID: "VARCHAR",
            TRACKER_CONSTANTS.TIMESTAMP: "VARCHAR"
        }
        primary_key = [SCHEMA_PARAMS.TRACK_ID, TRACKER_CONSTANTS.TIMESTAMP]
        self.create_table(self.table_name, columns, primary_key)

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback): # type: ignore
        self.close()


class SqliteEngine(Engine[sqlite3.Connection]):
    def _init_connection(self) -> sqlite3.Connection:
        """
        Initialize the SQLite database connection.
        """
        if self.db != ":memory:":
            db_dir = os.path.dirname(self.db)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        
        return conn
    
    def execute(self, query: str, params: Optional[List[Any]] = None) -> Any:
        cursor = self.conn.cursor()
        try:
            fixed_query = query
            # Handle table names in queries - both DuckDB-style and quote reserved keywords
            if self.table_name in query and f'"{self._strip_db_prefix(self.table_name)}"' not in query:
                if "." in self.table_name:
                    # Handle DuckDB-style table names by stripping db. prefix for SQLite
                    sqlite_table_name = self._strip_db_prefix(self.table_name)
                    quoted_table_name = self._quote_table_name(self.table_name)
                    fixed_query = query.replace(self.table_name, quoted_table_name)
                else:
                    # Quote table name to handle reserved keywords like 'default'
                    quoted_table_name = self._quote_table_name(self.table_name)
                    fixed_query = query.replace(self.table_name, quoted_table_name)
            
            if params:
                cursor.execute(fixed_query, params)
            else:
                cursor.execute(fixed_query)
            self.conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            logger.debug(f"SQLite error: {str(e)}, query: {query}")
            
            # Fallback: try stripping db. prefix if table not found
            if "no such table" in str(e) and "." in self.table_name and self.table_name in query:
                fixed_query = query.replace(self.table_name, self._strip_db_prefix(self.table_name))
                if fixed_query != query:
                    if params:
                        cursor.execute(fixed_query, params)
                    else:
                        cursor.execute(fixed_query)
                    self.conn.commit()
                    return cursor
            raise
    
    def _strip_db_prefix(self, table: str) -> str:
        """
        Strip the database prefix from a table name for SQLite.
        """
        if "." in table:
            return table.split(".")[-1]
        return table
    
    def _quote_table_name(self, table: str) -> str:
        """
        Quote table name to handle reserved keywords like 'default'.
        """
        table_name = self._strip_db_prefix(table)
        return f'"{table_name}"'
    
    def add_column(self, table: str, column: str, dtype: str) -> None:
        quoted_table_name = self._quote_table_name(table)
        self.execute(f"ALTER TABLE {quoted_table_name} ADD COLUMN {column} {dtype}")
        self._columns.add(column)
    
    def create_table(self, table: str, columns: Dict[str, str], primary_key: Optional[List[str]] = None) -> None:
        quoted_table_name = self._quote_table_name(table)
            
        column_defs = [f"{col} {dtype}" for col, dtype in columns.items()]
        if primary_key:
            column_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")
        
        query = f"CREATE TABLE IF NOT EXISTS {quoted_table_name} ({', '.join(column_defs)})"
        self.execute(query)

    @property
    def columns(self) -> set[str]:
        return set(self.dtypes.keys())

    @property
    def _sqlite_types(self):
        cursor = self.conn.cursor()
        quoted_table_name = self._quote_table_name(self.table_name)
        cursor.execute(f"PRAGMA table_info({quoted_table_name})")
        return [(row['name'], row['type']) for row in cursor.fetchall()]

    @property
    def dtypes(self) -> dict[str, Any]:
        return {
            column[0]: CONSTANTS.DTYPES_TO_PYTHON.get(column[1])
            for column in self._sqlite_types
        }

    @staticmethod
    def to_sql_type(value: Any) -> str:
        """
        Convert a primitive Python value to its corresponding SQL type.
        """
        value_type = type(value)
        if value_type == bytearray:
            return "BLOB"
        if value_type == int:
            return "INTEGER"
        if value_type == float:
            return "REAL"
        return "INTEGER" if value_type == bool else "TEXT"

    def set_value(self, key: str, value: Any, track_id: Optional[str] = None) -> None:
        if key not in self.columns:
            self.add_column(self.table_name, key, self.to_sql_type(value))

        quoted_table_name = self._quote_table_name(self.table_name)
        query = f"UPDATE {quoted_table_name} SET {key} = ?"
        params = [value]
        
        if track_id is not None:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            params.append(track_id)

        self.execute(query, params)

    def set_where(
        self,
        key: str,
        value: Any,
        where_key: str,
        where_value: Any,
        track_id: Optional[str] = None,
    ) -> None:
        params = [value, where_value]
        
        quoted_table_name = self._quote_table_name(self.table_name)
        sql = f"UPDATE {quoted_table_name} SET {key} = ? WHERE {where_key} = ?"
        
        if track_id is not None:
            sql += f" AND {SCHEMA_PARAMS.TRACK_ID} = ?"
            params.append(track_id)
            
        self.execute(sql, params)

    def remove_asset(
        self, hash_value: str, column: Optional[str], remove_keys: bool = True
    ) -> bool:
        if self.assets is None:
            logger.warning("Assets functionality disabled: sqlitedict not installed")
            return False
            
        if self.assets.remove_hash(hash_value, remove_keys=remove_keys):
            if column:
                quoted_table_name = self._quote_table_name(self.table_name)
                sql = f"UPDATE {quoted_table_name} SET {column} = NULL WHERE {column} = ?"
                self.execute(sql, [hash_value])
            return True
        return False
    
    def _insert_raw(self, data: List[tuple[list[str], list[Any], int]]) -> None:
        try:
            self.conn.execute("BEGIN TRANSACTION")

            for keys, values, size in data:
                sanitized_values: list[Any] = []
                for key, value in zip(keys, values):
                    # Convert complex types to string
                    if isinstance(value, (dict, list, tuple)):
                        sanitized_values.append(str(value)) # type: ignore
                    # Coerce str to int/float if needed based on schema
                    elif isinstance(value, str):
                        expected_type = self.dtypes.get(key)
                        if expected_type == int:
                            try:
                                sanitized_values.append(int(value))
                            except ValueError:
                                sanitized_values.append(value)
                        elif expected_type == float:
                            try:
                                sanitized_values.append(float(value))
                            except ValueError:
                                sanitized_values.append(value)
                        else:
                            sanitized_values.append(value)
                    else:
                        sanitized_values.append(value)

                placeholders = ', '.join(['?' for _ in range(size)])
                quoted_table_name = self._quote_table_name(self.table_name)
                query = f"INSERT INTO {quoted_table_name} ({', '.join(keys)}) VALUES ({placeholders})"

                self.conn.execute(query, sanitized_values)

            self.conn.commit()
        except sqlite3.IntegrityError:
            self.conn.rollback()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error inserting data: {str(e)}")
            raise

    def _ensure_events_table(self) -> None:
        events_table = self._strip_db_prefix(self.table_name)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (events_table,))
        
        if not cursor.fetchone():
            columns = {
                SCHEMA_PARAMS.TRACK_ID: "TEXT",
                TRACKER_CONSTANTS.TIMESTAMP: "TEXT"
            }
            primary_key = [SCHEMA_PARAMS.TRACK_ID, TRACKER_CONSTANTS.TIMESTAMP]
            self.create_table(self.table_name, columns, primary_key)
            logger.info(f"Created events table: {events_table}")
        else:
            logger.debug(f"Events table already exists: {events_table}")
            
        self._columns = set(self.dtypes.keys())

    def count_records(self, track_id: Optional[str] = None) -> int:
        quoted_table_name = self._quote_table_name(self.table_name)
        
        if track_id is not None:
            query = f"SELECT COUNT(*) FROM {quoted_table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            result = self.execute(query, [track_id]).fetchall()
        else:
            query = f"SELECT COUNT(*) FROM {quoted_table_name}"
            result = self.execute(query).fetchall()
        
        return result[0][0]

    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        cursor = self.execute(query, params)
                
        columns = [col[0] for col in cursor.description] if cursor.description else []        
        rows = cursor.fetchall()        
        return pd.DataFrame.from_records(rows, columns=columns)
    
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.debug("SQLite connection closed.")

try:
    from xetrack.duckdb import DuckDBEngine
    __all__.append('DuckDBEngine')
except ImportError:
    DuckDBEngine = None