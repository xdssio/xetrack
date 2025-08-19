import logging
import os
from typing import Any, Optional, List, Dict
from xetrack.engine import Engine
from xetrack.config import SCHEMA_PARAMS, CONSTANTS, TRACKER_CONSTANTS
import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class DuckDBEngine(Engine[duckdb.DuckDBPyConnection]):
    def __init__(
        self,
        db: str = "track.db",
        compress: bool = False,
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
    ):
        # For DuckDB, we need to ensure table names have the db. prefix
        if "." not in table_name:
            table_name = f"db.{table_name}"
        super().__init__(db, compress, table_name)
    
    def _init_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initialize a DuckDB connection and attach the SQLite database.
        
        Returns:
            A configured DuckDB connection
        """
        # Connect to DuckDB in-memory
        conn = duckdb.connect()
        
        # Install and load SQLite extension
        conn.execute("INSTALL sqlite;")
        conn.execute("LOAD sqlite;")

        # Get the database name from table schema
        db_name = self.table_name.split(".")[0] if "." in self.table_name else "db"
        
        # Check if the database is already attached
        is_attached = conn.execute(
            f"SELECT * FROM pragma_database_list WHERE name = '{db_name}'"
        ).fetchall()
        
        # Attach the SQLite database if not already attached
        if not is_attached:
            # Create directory for database file if needed
            if self.db != CONSTANTS.IN_MEMORY_DB:
                db_dir = os.path.dirname(self.db)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                    
            conn.execute(f"ATTACH '{self.db}' AS {db_name} (TYPE SQLITE)")
            
        # Use the attached database
        conn.execute(f"USE {db_name};")
        return conn
    
    def _ensure_events_table(self) -> None:
        """
        Ensure the events table exists in the DuckDB database.
        
        DuckDB works with the db prefix in table names.
        """
        # Check if the table already exists
        db_name = self.table_name.split(".")[0] if "." in self.table_name else "db"
        table_only_name = self.table_name.split(".")[-1]
        
        result = self.execute(
            f"""
            SELECT count(*) 
            FROM system.information_schema.tables 
            WHERE table_schema = '{db_name}' 
            AND table_name = '{table_only_name}'
            """
        ).fetchone()
        
        if not result or result[0] == 0:
            # Create the events table with timestamp and track_id columns
            columns = {
                SCHEMA_PARAMS.TRACK_ID: "VARCHAR",
                TRACKER_CONSTANTS.TIMESTAMP: "VARCHAR"
            }
            primary_key = [SCHEMA_PARAMS.TRACK_ID, TRACKER_CONSTANTS.TIMESTAMP]
            self.create_table(self.table_name, columns, primary_key)
            logger.info(f"Created events table: {self.table_name}")
        else:
            logger.debug(f"Events table already exists: {self.table_name}")
            
        # Populate the columns set with existing columns
        self._columns = set(self.dtypes.keys())
    
    def execute(self, query: str, params: Optional[List[Any]] = None) -> Any:
        """
        Execute a SQL query with optional parameters.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            The result of the query execution
        """
        try:
            if params:
                return self.conn.execute(query, params)
            return self.conn.execute(query)
        except Exception as e:
            # Log the error with details
            logger.error(f"Error executing query: {query} with params: {params} - {str(e)}")
            raise
    
    def add_column(self, table: str, column: str, dtype: str) -> None:
        """Add a column to a table if it doesn't exist."""
        self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
        self._columns.add(column)
    
    def create_table(self, table: str, columns: Dict[str, str], primary_key: Optional[List[str]] = None) -> None:
        """Create a table if it doesn't exist."""
        column_defs = [f"{col} {dtype}" for col, dtype in columns.items()]
        if primary_key:
            column_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")
        
        query = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(column_defs)})"
        self.execute(query)

    @property
    def columns(self) -> set[str]:
        return set(self.dtypes.keys())

    @property
    def _duckdb_types(self):
        table_only_name = self.table_name.split(".")[-1]
        return self.execute(
            f"""
            SELECT column_name, data_type
            FROM system.information_schema.columns
            WHERE table_schema = 'main'
            AND table_name = '{table_only_name}'
            """
        ).fetchall()

    @property
    def dtypes(self) -> dict[str, Any]:
        return {
            column[0]: CONSTANTS.DTYPES_TO_PYTHON.get(column[1])
            for column in self._duckdb_types
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
            return "BIGINT" if value.bit_length() > 64 else "INTEGER"
        if value_type == float:
            return "FLOAT"
        return "BOOLEAN" if value_type == bool else "VARCHAR"

    def set_value(self, key: str, value: Any, track_id: Optional[str] = None) -> None:
        """
        Sets the value of a specific key in the database table.

        Args:
            key: The key whose value needs to be set.
            value: The value to set for the specified key.
            track_id: The track ID used to identify the specific record in the table.
        """
        if key not in self.columns:
            self.add_column(self.table_name, key, self.to_sql_type(value))

        query = f"UPDATE {self.table_name} SET {key} = ?"
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
        """Sets the value of a specific key in the database table given a where key value pair."""
        params = [value, where_value]
        
        sql = f"UPDATE {self.table_name} SET {key} = ? WHERE {where_key} = ?"
        
        if track_id is not None:
            sql += f" AND {SCHEMA_PARAMS.TRACK_ID} = ?"
            params.append(track_id)
            
        self.execute(sql, params)

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
        if self.assets is None:
            logger.warning("Assets functionality disabled: sqlitedict not installed")
            return False
            
        if self.assets.remove_hash(hash_value, remove_keys=remove_keys):
            if column:
                sql = f"UPDATE {self.table_name} SET {column} = NULL WHERE {column} = ?"
                self.execute(sql, [hash_value])
            return True
        return False
    
    def _insert_raw(self, data: List[tuple[list[str], list[Any], int]]) -> None:
        """
        Inserts new rows into the database table.
        
        Args:
            data: List of (keys, values, size) tuples to insert.
        """
        self.execute("BEGIN TRANSACTION")
        for keys, values, size in data:
            try:
                # Convert values to appropriate types for DuckDB
                sanitized_values = []
                for i, val in enumerate(values):
                    # Special handling for timestamp column
                    if keys[i] == TRACKER_CONSTANTS.TIMESTAMP and isinstance(val, str):
                        # Keep timestamp as string
                        sanitized_values.append(val)
                    elif isinstance(val, (dict, list, tuple)):
                        # Convert complex types to strings
                        sanitized_values.append(str(val))
                    else:
                        sanitized_values.append(val)
                
                placeholders = ', '.join(['?' for _ in range(size)])
                query = f"INSERT INTO {self.table_name} ({', '.join(keys)}) VALUES ({placeholders})"
                self.execute(query, sanitized_values)
            except Exception as e:
                logger.error(f"Error inserting data: {str(e)}")
                # Continue with other rows
        self.execute("COMMIT TRANSACTION")

    def count_records(self, track_id: Optional[str] = None) -> int:
        """
        Count records in the events table, optionally filtered by track_id.
        
        Args:
            track_id: Optional track ID to filter by
            
        Returns:
            Number of records matching the criteria
        """
        if track_id is not None:
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            result = self.execute(query, [track_id]).fetchall()
        else:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
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
        result = self.execute(query, params)
        # DuckDB has native support for converting results to a DataFrame
        return result.df()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.debug("DuckDB connection closed.")
