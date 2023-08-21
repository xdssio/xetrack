from xetrack.connection import DuckDBConnection
from xetrack.constants import TRACK_ID, TABLE


class Reader(DuckDBConnection):

    def to_df(self, track_id: int = None):
        query = f"SELECT * FROM {TABLE}"
        if track_id is not None:
            query += f" WHERE {TRACK_ID} = '{track_id}'"
        return self.conn.execute(query).df()

    def latest(self):
        latest_track_id = self.conn.execute(f"SELECT track_id FROM {TABLE} ORDER BY track_id DESC LIMIT 1"). \
            fetchone()[0]
        return self.conn.execute(f"SELECT * FROM db.events WHERE track_id = '{latest_track_id}'").df()
