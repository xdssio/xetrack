import logging
from xetrack.constants import _DTYPES_TO_PYTHON, TABLE
import duckdb

EVENTS = 'events'
DB = 'db'
TRACK_ID = 'track_id'

logger = logging.getLogger(__name__)


class DuckDBConnection:
    def __init__(self, db: str = 'track.db'):
        logger.debug(f"Connecting to {db}")
        self.db = db
        self.table_name = f"{DB}.{EVENTS}"
        self.conn = self._init_connection()

    def _init_connection(self):
        conn = duckdb.connect()
        if 'sqlite_scanner' not in conn.execute("SELECT * FROM duckdb_extensions()").fetchall():
            conn.install_extension('sqlite')
        # conn.load_extension('sqlite')
        is_attached = conn.execute(f"SELECT * FROM pragma_database_list WHERE name = '{DB}'").fetchall()
        if not is_attached:
            conn.execute(f"ATTACH '{self.db}' AS {DB} (TYPE SQLITE)")
        return conn

    @property
    def columns(self):
        return set(self.dtypes.keys())

    @property
    def dtypes(self):
        return {column[0]: _DTYPES_TO_PYTHON.get(column[1]) for column in
                self.conn.execute(f"DESCRIBE {TABLE}").fetchall()}
