import contextlib
import logging

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