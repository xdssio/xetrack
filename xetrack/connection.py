import logging
from typing import Any, Optional, List, TypeVar, Generic, Dict, Set
from xetrack.assets import AssetsManager
from xetrack.config import SCHEMA_PARAMS, CONSTANTS, DEFAULTS, TRACKER_CONSTANTS
import sqlite3
from abc import ABC, abstractmethod
import os
import pandas as pd


logger = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = ['Connection', 'SqliteConnection']


class Connection(ABC, Generic[T]):
    def __init__(self, db: str = DEFAULTS.DB, compress: bool = False):
        """
        Abstract base class for database connections.
        """
        logger.debug(f"Connecting to {db}")
        self.db = db
        self.table_name = SCHEMA_PARAMS.DUCKDB_TABLE
        self.conn: T = self._init_connection()
        self.assets = AssetsManager(path=db, compress=compress, autocommit=True)
        self._columns: Set[str] = set()
        
        # Initialize the database structure
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
        
        This is called after connection initialization to ensure the database
        has all required tables and structures.
        """
        # Create the events table if it doesn't exist
        self._ensure_events_table()
    
    def _ensure_events_table(self) -> None:
        """
        Ensure the events table exists in the database.
        
        Override this in subclasses if special handling is needed.
        """
        # Create the events table with timestamp and track_id columns
        columns = {
            SCHEMA_PARAMS.TRACK_ID: "VARCHAR",
            TRACKER_CONSTANTS.TIMESTAMP: "VARCHAR"
        }
        primary_key = [SCHEMA_PARAMS.TRACK_ID, TRACKER_CONSTANTS.TIMESTAMP]
        self.create_table(SCHEMA_PARAMS.DUCKDB_TABLE, columns, primary_key)


class SqliteConnection(Connection[sqlite3.Connection]):
    def _init_connection(self) -> sqlite3.Connection:
        """
        Initialize the SQLite database connection.
        
        Creates the database file if it doesn't exist and establishes a connection.
        """
        # SQLite will create the file if it doesn't exist
        # Skip the file creation if using in-memory database
        if self.db != ":memory:":
            db_dir = os.path.dirname(self.db)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        
        # Enable foreign keys support
        conn.execute("PRAGMA foreign_keys = ON")
        
        return conn
    
    def execute(self, query: str, params: Optional[List[Any]] = None) -> Any:
        cursor = self.conn.cursor()
        try:
            # For SQLite, we need to strip the database prefix from table names
            # If we see a query with "db.events", we'll convert it to just "events"
            fixed_query = query
            if SCHEMA_PARAMS.DUCKDB_TABLE in query:
                table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
                fixed_query = query.replace(SCHEMA_PARAMS.DUCKDB_TABLE, table_name)
            
            if params:
                cursor.execute(fixed_query, params)
            else:
                cursor.execute(fixed_query)
            self.conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            # Log the error for debugging
            logger.debug(f"SQLite error: {str(e)}, query: {query}")
            
            # If the error is about an unknown database, try to fix the query
            if "no such table" in str(e) and SCHEMA_PARAMS.DUCKDB_TABLE in query:
                # For SQLite, we don't need to prefix table names with database name
                # So if we see a query with "db.table_name", we'll convert it to just "table_name"
                fixed_query = query.replace(f"{SCHEMA_PARAMS.DUCKDB_TABLE}", self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE))
                if fixed_query != query:
                    if params:
                        cursor.execute(fixed_query, params)
                    else:
                        cursor.execute(fixed_query)
                    self.conn.commit()
                    return cursor
            # If we couldn't fix it or it's another error, re-raise
            raise
    
    def _strip_db_prefix(self, table: str) -> str:
        """
        Strip the database prefix from a table name for SQLite.
        
        Args:
            table: Table name, possibly with db prefix (e.g., "db.table")
            
        Returns:
            Table name without db prefix
        """
        if "." in table:
            return table.split(".")[-1]
        return table
    
    def add_column(self, table: str, column: str, dtype: str) -> None:
        """
        Add a column to a table if it doesn't exist.
        
        Args:
            table: The table name
            column: The column name
            dtype: The SQL data type for the column
        """
        table_name = self._strip_db_prefix(table)
        self.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {dtype}")
        self._columns.add(column)
    
    def create_table(self, table: str, columns: Dict[str, str], primary_key: Optional[List[str]] = None) -> None:
        """
        Create a table if it doesn't exist.
        
        Args:
            table: The table name
            columns: Dictionary mapping column names to their SQL data types
            primary_key: Optional list of column names to use as primary key
        """
        # Strip database prefix from table name for SQLite
        table_name = self._strip_db_prefix(table)
            
        column_defs = [f"{col} {dtype}" for col, dtype in columns.items()]
        if primary_key:
            column_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")
        
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
        self.execute(query)

    @property
    def columns(self) -> set[str]:
        return set(self.dtypes.keys())

    @property
    def _sqlite_types(self):
        cursor = self.conn.cursor()
        table_name = self._strip_db_prefix(SCHEMA_PARAMS.EVENTS_TABLE)
        cursor.execute(f"PRAGMA table_info({table_name})")
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

        Parameters:
            value (Any): The value to be converted.

        Returns:
            str: The SQL type corresponding to the given value.
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
        """
        Sets the value of a specific key in the database table.

        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
            track_id: The track ID used to identify the specific record in the table.
        """
        if key not in self.columns:
            self.add_column(SCHEMA_PARAMS.DUCKDB_TABLE, key, self.to_sql_type(value))

        table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
        query = f"UPDATE {table_name} SET {key} = ?"
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
        """
        Sets the value of a specific key in the database table given a where key value pair.
        """
        params = [value, where_value]
        
        table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
        sql = f"UPDATE {table_name} SET {key} = ? WHERE {where_key} = ?"
        
        if track_id is not None:
            sql += f" AND {SCHEMA_PARAMS.TRACK_ID} = ?"
            params.append(track_id)
            
        self.execute(sql, params)

    def remove_asset(
        self, hash_value: str, column: Optional[str], remove_keys: bool = True
    ) -> bool:
        """
        Removes the asset stored in the database with the given hash.
        """
        if self.assets.remove_hash(hash_value, remove_keys=remove_keys):
            if column:
                table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
                sql = f"UPDATE {table_name} SET {column} = NULL WHERE {column} = ?"
                self.execute(sql, [hash_value])
            return True
        return False
    
    def _insert_raw(self, data: List[tuple[list[str], list[Any], int]]) -> None:
        """
        Inserts new rows into the database table.
        """
        try:
            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            for keys, values, size in data:
                # Convert any complex types to strings
                sanitized_values = []
                for value in values:
                    if isinstance(value, (dict, list, tuple)):
                        # Convert complex types to strings for SQLite
                        sanitized_values.append(str(value))
                    else:
                        sanitized_values.append(value)
                        
                placeholders = ', '.join(['?' for _ in range(size)])
                table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
                query = f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders})"
                
                # Execute with sanitized values
                self.conn.execute(query, sanitized_values)
                
            # Commit transaction
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Rollback on integrity error
            self.conn.rollback()
            # Skip rows that would violate constraints
        except Exception as e:
            # Rollback on any other error
            self.conn.rollback()
            logger.error(f"Error inserting data: {str(e)}")
            raise

    def _ensure_events_table(self) -> None:
        """
        Ensure the events table exists in the SQLite database.
        
        SQLite requires special handling for table names with db prefixes.
        """
        # Get the table name without the db prefix
        events_table = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
        
        # Check if the table already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (events_table,))
        
        if not cursor.fetchone():
            # Create the events table with timestamp and track_id columns
            columns = {
                SCHEMA_PARAMS.TRACK_ID: "TEXT",
                TRACKER_CONSTANTS.TIMESTAMP: "TEXT"
            }
            primary_key = [SCHEMA_PARAMS.TRACK_ID, TRACKER_CONSTANTS.TIMESTAMP]
            self.create_table(SCHEMA_PARAMS.DUCKDB_TABLE, columns, primary_key)
            logger.info(f"Created events table: {events_table}")
        else:
            logger.debug(f"Events table already exists: {events_table}")
            
        # Populate the columns set with existing columns
        self._columns = set(self.dtypes.keys())

    def count_records(self, track_id: Optional[str] = None) -> int:
        """
        Count records in the events table, optionally filtered by track_id.
        
        Args:
            track_id: Optional track ID to filter by
            
        Returns:
            Number of records matching the criteria
        """
        table_name = self._strip_db_prefix(SCHEMA_PARAMS.DUCKDB_TABLE)
        
        if track_id is not None:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            result = self.execute(query, [track_id]).fetchall()
        else:
            query = f"SELECT COUNT(*) FROM {table_name}"
            result = self.execute(query).fetchall()
        
        return result[0][0]

    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return a pandas DataFrame with the results.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            pandas.DataFrame: A DataFrame containing the query results
        """
        cursor = self.execute(query, params)
                
        columns = [col[0] for col in cursor.description] if cursor.description else []        
        rows = cursor.fetchall()        
        return pd.DataFrame.from_records(rows, columns=columns)

try:
    from xetrack.duckdb import DuckDBConnection
    __all__.append('DuckDBConnection')
except ImportError:
    DuckDBConnection = None