import datetime

import pandas as pd
from xetrack.connection import DuckDBConnection
from xetrack.config import SCHEMA_PARAMS
from xetrack.logging import Logger
from typing import Any, Optional
import sqlite3


class Reader(DuckDBConnection):

    def to_df(self, track_id: Optional[str] = None, head: Optional[int] = None, tail: Optional[int] = None):
        """
        Returns a pandas dataframe of the events table

        Args:
            track_id (Optional[str], optional): The track ID used to identify the specific record in the table. Defaults to None.
            head (Optional[int], optional): The number of rows to return from the head of the table. Defaults to None.
            tail (Optional[int], optional): The number of rows to return from the tail of the table. Defaults to None.        
        """
        query = f"SELECT * FROM {SCHEMA_PARAMS.TABLE}"
        if track_id is not None:
            query += f" WHERE {SCHEMA_PARAMS.TRACK_ID} = '{track_id}'"
        elif head is not None:
            query += f" LIMIT {head}"
        elif tail is not None:
            query += f" ORDER BY timestamp DESC LIMIT {tail}"
        results = self.conn.execute(query).df()
        return results.sort_values(by=['timestamp'])

    def latest(self):
        latest_track_id = self.conn.execute(f"SELECT track_id FROM {SCHEMA_PARAMS.TABLE} ORDER BY track_id DESC LIMIT 1"). \
            fetchone()[0]
        return self.conn.execute(f"SELECT * FROM db.events WHERE track_id = '{latest_track_id}'").df()

    def delete_run(self, track_id: str):
        """
        Using sqlite3 to remove rows
        * unclear why duckdb DELETE not working ):
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM events WHERE track_id = '{track_id}'")
        conn.commit()
        conn.close()
        return True

    @property
    def columns(self):
        return set(self.dtypes.keys())

    def __len__(self):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {SCHEMA_PARAMS.TABLE}").fetchall()[
            0][0]

    @classmethod
    def read_logs(cls, path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Return a pandas dataframe of the logs in the given path"""
        helper = Logger()
        return pd.DataFrame(helper.read_logs(path, limit=limit))
