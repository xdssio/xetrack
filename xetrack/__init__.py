import contextlib

import duckdb

from .reader import Reader
from .tracker import Tracker
from .config import SCHEMA_PARAMS, TRACKER_CONSTANTS


def copy(source: str,
         target: str,
         assets: bool = True
         ):
    """
    Copies the data from one tracker to another
    :param source: The source database file
    :param target: The target database file
    """

    # if handle_duplicate not in ('IGNORE', 'REPLACE'):
    #     raise ValueError(f"Invalid handle_duplicate: {handle_duplicate} - Must be either IGNORE or REPLACE")
    source_tracker: Tracker = Tracker(db=source)
    target_tracker: Tracker = Tracker(db=target)
    ids = target_tracker.conn.execute(  # type: ignore
        "SELECT timestamp, track_id track_id FROM db.events").fetchall()  # type: ignore
    ids = {f"{id[0]}-{id[1]}" for id in ids}
    results = source_tracker.conn.execute(
        f"SELECT * FROM {SCHEMA_PARAMS.TABLE}").fetchall()
    if len(results) == 0:
        print('No data to copy')
        return
    new_column_count = 0
    for column, hash_value in source_tracker.dtypes.items():
        if column not in target_tracker._columns:
            new_column_count += 1
            target_tracker.add_column(column, hash_value)
    keys = [column[1] for column in source_tracker.conn.execute(
        f"PRAGMA table_info({SCHEMA_PARAMS.TABLE})").fetchall()]
    size = len(keys)
    timestamp_ix, track_ix = keys.index(
        TRACKER_CONSTANTS.TIMESTAMP), keys.index(SCHEMA_PARAMS.TRACK_ID)
    count = 0
    source_assets = source_tracker.assets
    target_assets = target_tracker.assets
    for key, hash_value in source_assets.keys.items():
        target_assets.keys[key] = hash_value
        if hash_value not in target_assets.assets:
            target_assets.assets[hash_value] = source_assets.assets.get(
                hash_value)
            if hash_value not in target_assets.counts:
                target_assets.counts[hash_value] = source_assets.counts.get(
                    hash_value)
            else:
                target_assets.counts[hash_value] += source_assets.counts.get(
                    hash_value)
    target_tracker.conn.execute("BEGIN TRANSACTION")
    for event in results:
        if f"{event[timestamp_ix]}-{event[track_ix]}" in ids:
            continue
        count += 1
        values = list(event)
        with contextlib.suppress(duckdb.ConstraintException):
            target_tracker.conn.execute(
                f"INSERT INTO {SCHEMA_PARAMS.TABLE} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                values)
    target_tracker.conn.execute("COMMIT TRANSACTION")
    total = target_tracker.count_all()
    print(
        f"Copied {count} events and {new_column_count} new columns. New total is {total} events")
    return count
