import logging
from typing import Any, Optional
from xetrack.assets import AssetsManager
from xetrack.config import SCHEMA_PARAMS, CONSTANTS
import duckdb


logger = logging.getLogger(__name__)


class DuckDBConnection:
    def __init__(self, db: str = 'track.db',
                 compress: bool = False):
        """
        Connect to a SQLlite database with DuckDB and setup the assets manager.

        """
        logger.debug(f"Connecting to {db}")
        self.db = db
        self.table_name = SCHEMA_PARAMS.TABLE
        self.conn = self._init_connection()
        self.assets = AssetsManager(
            path=db, compress=compress, autocommit=True)

    def _init_connection(self):
        conn = duckdb.connect()
        if 'sqlite_scanner' not in conn.execute("SELECT * FROM duckdb_extensions()").fetchall():
            conn.install_extension('sqlite')
        # conn.load_extension('sqlite')
        db = SCHEMA_PARAMS.TABLE.split('.')[0]
        is_attached = conn.execute(
            f"SELECT * FROM pragma_database_list WHERE name = '{db}'").fetchall()
        if not is_attached:
            conn.execute(f"ATTACH '{self.db}' AS {db} (TYPE SQLITE)")
        return conn

    @property
    def columns(self):
        return set(self.dtypes.keys())

    @property
    def dtypes(self):
        return {column[0]: CONSTANTS.DTYPES_TO_PYTHON.get(column[1]) for column in
                self.conn.execute(f"DESCRIBE {SCHEMA_PARAMS.TABLE}").fetchall()}

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
            return 'BLOB'
        if value_type == int:
            return 'BIGINT' if value.bit_length() > 64 else 'INTEGER'
        if value_type == float:
            return 'FLOAT'
        return 'BOOLEAN' if value_type == bool else 'VARCHAR'

    def set_value(self, key: str, value: Any, track_id: Optional[str] = None):
        """
        Sets the value of a specific key in the database table.

        Args:
            key (Any): The key whose value needs to be set.
            value (Any): The value to set for the specified key.
            track_id (Optional[str], optional): The track ID used to identify the specific record in the table. Defaults to None.

        Returns:
            None
        """
        if key not in self.columns:
            self.conn.execute(
                f"ALTER TABLE {SCHEMA_PARAMS.TABLE} ADD COLUMN {key} {self.to_sql_type(value)}")

        query = f"UPDATE {SCHEMA_PARAMS.TABLE} SET {key} = '{value}'"
        if track_id is not None:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = '{track_id}'"

        self.conn.execute(query)

    def set_where(self, key: str, value: Any, where_key: str, where_value: Any, track_id: Optional[str] = None):
        """Sets the value of a specific key in the database table given a where key value pair."""
        sql = f"""UPDATE {SCHEMA_PARAMS.TABLE} SET {key} = '{value}' WHERE {where_key} = '{where_value}'"""
        if track_id is not None:
            sql += f" AND {SCHEMA_PARAMS.TRACK_ID} = '{track_id}'"
        self.conn.execute(sql)

    def remove_asset(self, hash_value: str, column: Optional[str], remove_keys: bool = True):
        """Removes the asset stored in the database with the given hash. If a column is given, the value of that column will be set to None
        Args:
            hash_value (str): The hash of the asset to remove
            column (Optional[str], optional): The column to set to None. Defaults to None.
            remove_keys (bool, optional): Whether to remove the keys associated with the asset. Defaults to True.

        Note:
            A model removed is deleted from all runs - model assets are global
        """
        if self.assets.remove_hash(hash_value, remove_keys=remove_keys):
            if column:
                SQL = f"""UPDATE {SCHEMA_PARAMS.TABLE} SET {column} = NULL WHERE {column} = '{hash_value}'"""
                self.conn.execute(SQL)
            return True
        return False
