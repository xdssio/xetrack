from xetrack.connection import DuckDBConnection
from xetrack.constants import TRACK_ID, TABLE


class Reader(DuckDBConnection):

    def to_df(self, track_id: int = None):
        query = f"SELECT * FROM {TABLE}"
        if track_id is not None:
            query += f" WHERE {TRACK_ID} = '{track_id}'"
        return self.conn.execute(query).df()
