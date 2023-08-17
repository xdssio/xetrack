import duckdb
import pandas as pd
from xetrack import DB, EVENTS,TRACK_ID


class Reader:

    def __init__(self, db: str = 'track.db',
                 verbose: bool = True,
                 ):
        """
        :param db: The duckdb database file to use or ":memory:" for in-memory database - default is "tracker.db"
        :param params: A dictionary of default parameters to attach to every event - this can be changed later
        :param reset: If True, the events table will be dropped and recreated - default is False
        :param verbose: If True, log messages will be printed - default is True
        """

        self.db = db
        self.table_name = EVENTS
        self.verbose = verbose
        self._columns = set()
        self.conn = self._init_connection()

    def _init_connection(self):
        conn = duckdb.connect()
        if 'sqlite_scanner' not in conn.execute("SELECT * FROM duckdb_extensions()").fetchall():
            conn.install_extension('sqlite')
        conn.load_extension('sqlite')
        dbs = conn.execute("PRAGMA database_list").fetchall()
        if len(dbs) < 2:
            conn.execute(f"ATTACH '{self.db}' AS {DB} (TYPE SQLITE)")
        return conn

    @property
    def _table(self):
        return self.conn.table(self.table_name)

    def to_df(self, track_id:int=None):
        if track_id:
            results = self.conn.execute(
                f"SELECT * FROM {DB}.{self.table_name} WHERE {TRACK_ID} = '{track_id}'").fetchall()
            return pd.DataFrame(results, columns=self._table.columns)
        return self._table.to_df()

self = Reader('/var/folders/gl/cklpy5415rzd6vb8y29rccpr0000gn/T/tmpwa4pnlyw/database.db')
self.to_df()
