import contextlib

import duckdb

from .tracker import Tracker
from .reader import Reader
from .constants import TRACK_ID, TABLE


def copy(source: str,
         target: str,
         ):
    """
    Copies the data from one tracker to another
    :param source: The source database file
    :param target: The target database file
    """

    # if handle_duplicate not in ('IGNORE', 'REPLACE'):
    #     raise ValueError(f"Invalid handle_duplicate: {handle_duplicate} - Must be either IGNORE or REPLACE")
    source = Tracker(db=source)
    target = Tracker(db=target)
    results = source.conn.execute(f"SELECT * FROM {TABLE}").fetchall()
    if len(results) == 0:
        print('No data to copy')
        return
    new_column_count = 0
    for column, value in source.dtypes.items():
        if column not in target._columns:
            new_column_count += 1
            target.add_column(column, value, source.to_py_type(value))
    keys = [column[1] for column in source.conn.execute(f"PRAGMA table_info({TABLE})").fetchall()]
    size = len(keys)
    target.conn.execute("BEGIN TRANSACTION")
    for event in results:
        values = list(event)
        with contextlib.suppress(duckdb.ConstraintException):
            target.conn.execute(
                f"INSERT INTO {TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                values)
    target.conn.execute("COMMIT TRANSACTION")
    total = target.count_all()
    print(f"Copied {len(results)} events and {new_column_count} new columns. New total is {total} events")
