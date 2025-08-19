import pandas as pd
from typing import Any, Optional, Literal

from xetrack.engine import DuckDBEngine, SqliteEngine
from xetrack.config import SCHEMA_PARAMS
from xetrack.logging import Logger


class Reader:
    def __init__(
        self,
        db: str,
        engine: Literal["duckdb", "sqlite"] = "sqlite",
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
    ):
        """
        Initialize a Reader with the specified database file and engine.
        
        Args:
            db: The database file path
            engine: The database engine to use, either "duckdb" or "sqlite". Default is "sqlite".
            table_name: Name of the table to read from. Default is "default". Allows reading different experiment types.
        """
        self.db = db
        self.table_name = table_name
        if engine == "sqlite":
            self.engine = SqliteEngine(db=db, table_name=table_name)
        else:
            self.engine = DuckDBEngine(db=db, table_name=table_name)
    
    @property
    def conn(self):
        return self.engine.conn
        
    @property
    def assets(self):
        return self.engine.assets
        
    @property
    def columns(self):
        return self.engine.columns
        
    @property
    def dtypes(self):
        return self.engine.dtypes

    def to_df(self, track_id: Optional[str] = None, head: Optional[int] = None, tail: Optional[int] = None):
        """
        Returns a pandas dataframe of the events table

        Args:
            track_id (Optional[str], optional): The track ID used to identify the specific record in the table. Defaults to None.
            head (Optional[int], optional): The number of rows to return from the head of the table. Defaults to None.
            tail (Optional[int], optional): The number of rows to return from the tail of the table. Defaults to None.        
        """
        query = f"SELECT * FROM {self.engine.table_name}"
        if track_id is not None:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
            cursor = self.engine.execute(query, [track_id])
        elif head is not None:
            query += f" LIMIT {head}"
            cursor = self.engine.execute(query)
        elif tail is not None:
            query += f" ORDER BY timestamp DESC LIMIT {tail}"
            cursor = self.engine.execute(query)
        else:
            cursor = self.engine.execute(query)
            
        # Convert cursor results to a DataFrame
        if hasattr(cursor, 'df'):
            results = cursor.df()
        else:
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else []
            # Fetch all rows
            rows = cursor.fetchall()
            # Convert to DataFrame
            results = pd.DataFrame.from_records(rows, columns=columns)
            
        return results.sort_values(by=['timestamp'])

    def latest(self):
        query = f"SELECT {SCHEMA_PARAMS.TRACK_ID} FROM {self.engine.table_name} ORDER BY {SCHEMA_PARAMS.TRACK_ID} DESC LIMIT 1"
        result = self.engine.execute(query).fetchone()
        
        if result is None:
            return pd.DataFrame()  # Return empty DataFrame if no results
            
        latest_track_id = result[0]
        query = f"SELECT * FROM {self.engine.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
        cursor = self.engine.execute(query, [latest_track_id])
        
        # Convert cursor results to a DataFrame
        if hasattr(cursor, 'df'):
            return cursor.df()
        else:
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else []
            # Fetch all rows
            rows = cursor.fetchall()
            # Convert to DataFrame
            return pd.DataFrame.from_records(rows, columns=columns)

    def delete_run(self, track_id: str):
        """
        Delete a run by track_id
        """
        query = f"DELETE FROM {self.engine.table_name} WHERE {SCHEMA_PARAMS.TRACK_ID} = ?"
        self.engine.execute(query, [track_id])
        return True

    def set_value(self, key: str, value: Any, track_id: Optional[str] = None) -> None:
        """Set a value in the database"""
        return self.engine.set_value(key, value, track_id)
        
    def set_where(self, key: str, value: Any, where_key: str, where_value: Any, track_id: Optional[str] = None) -> None:
        """
        Set a value in the database where a condition is met
        
        Args:
            key: The key whose value needs to be set
            value: The value to set
            where_key: The column to check in the WHERE clause
            where_value: The value to match in the WHERE clause
            track_id: Optional track ID to restrict updates to
        """
        # For SQLite connections, we need to handle string values differently
        # SQLite might store numeric values as strings, so we need to ensure the 
        # where_value comparison handles this correctly
        if isinstance(self.engine, SqliteEngine):
            # Get all records that match the criteria
            df = self.to_df()
            
            # For SQLite, convert values to string for comparison
            # This is important because SQLite might store numeric values as strings
            matching_rows = df[df[where_key].astype(str) == str(where_value)]
            
            # If track_id is provided, further filter the matching rows
            if track_id is not None:
                matching_rows = matching_rows[matching_rows[SCHEMA_PARAMS.TRACK_ID] == track_id]
                
            # For each matching row, update the value using track_id
            for _, row in matching_rows.iterrows():
                self.set_value(key, value, row[SCHEMA_PARAMS.TRACK_ID])
        else:
            # For DuckDB, use the engine's set_where method directly
            self.engine.set_where(key, value, where_key, where_value, track_id)
            
            # Ensure the database change is flushed and our in-memory representation is updated
            self.engine.conn.commit()
        
    def remove_asset(self, hash_value: str, column: Optional[str], remove_keys: bool = True) -> bool:
        """Remove an asset from the database"""
        return self.engine.remove_asset(hash_value, column, remove_keys)

    def __len__(self):
        """Get the number of records in the database"""
        return self.engine.count_records()

    @classmethod
    def read_logs(cls, path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Return a pandas dataframe of the logs in the given path"""
        helper = Logger()
        logs_df = pd.DataFrame(helper.read_logs(path, limit=limit))
        
        # Filter out rows where all columns are NaN
        if not logs_df.empty:
            logs_df = logs_df.dropna(how='all')
        
        return logs_df
