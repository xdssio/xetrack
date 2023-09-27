from xetrack.connection import DuckDBConnection
from xetrack.constants import TRACK_ID, TABLE
from typing import Any
import sqlite3


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

    def set_value(self, key: str, value: Any, track_id: str):
        """
        Sets the value of a specific key in the database table.

        Args:
            key (Any): The key whose value needs to be set.
            value (Any): The value to set for the specified key.
            track_id (str): The track ID used to identify the specific record in the table.

        Returns:
            None
        """
        if key not in self.columns:
            self.conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {key} {self.to_sql_type(value)}")
        self.conn.execute(
            f"UPDATE {TABLE} SET {key} = '{value}' WHERE {TRACK_ID} = '{track_id}'")

    def set_where(self, key: str, value: Any, where_key: str, where_value: Any, track_id: str = None):
        SQL = f"""UPDATE {TABLE} SET {key} = '{value}' WHERE {where_key} = '{where_value}'"""
        if track_id is not None:
            SQL += f" AND {TRACK_ID} = '{track_id}'"
        self.conn.execute(SQL)

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
            f"SELECT COUNT(*) FROM {TABLE}").fetchall()[
            0][0]
