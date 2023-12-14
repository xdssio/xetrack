import logging
from xetrack.config import SCHEMA_PARAMS, CONSTANTS
import duckdb


logger = logging.getLogger(__name__)


class DuckDBConnection:
    def __init__(self, db: str = 'track.db'):
        logger.debug(f"Connecting to {db}")
        self.db = db
        self.table_name = SCHEMA_PARAMS.TABLE
        self.conn = self._init_connection()

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
        return {column[0]: CONSTANTS._DTYPES_TO_PYTHON.get(column[1]) for column in
                self.conn.execute(f"DESCRIBE {SCHEMA_PARAMS.TABLE}").fetchall()}

    @staticmethod
    def to_sql_type(value) -> str:
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
