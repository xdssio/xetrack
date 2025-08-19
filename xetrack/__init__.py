import contextlib
from loguru import logger
from xetrack.config import SCHEMA_PARAMS, TRACKER_CONSTANTS
from xetrack.tracker import Tracker
from xetrack.reader import Reader

__version__ = "0.0.0"

try:
    from importlib.metadata import version
    __version__ = version("xetrack")
except ImportError:
    pass

__all__ = ['Reader', 'Tracker', 'copy']


def copy(source: str, target: str, assets: bool = True):
    """
    Copies the data from one tracker to another
    :param source: The source database file
    :param target: The target database file
    :param assets: Whether to copy assets or not
    """

    import duckdb
    # if handle_duplicate not in ('IGNORE', 'REPLACE'):
    #     raise ValueError(f"Invalid handle_duplicate: {handle_duplicate} - Must be either IGNORE or REPLACE")
    source_tracker: Tracker = Tracker(db=source, engine='duckdb')
    target_tracker: Tracker = Tracker(db=target, engine='duckdb')
    ids = target_tracker.conn.execute(  # type: ignore
        f"SELECT timestamp, track_id track_id FROM {target_tracker.engine.table_name}"
    ).fetchall()  # type: ignore
    ids = {f"{id[0]}-{id[1]}" for id in ids}
    results = source_tracker.conn.execute(
        f"SELECT * FROM {source_tracker.engine.table_name}" # type: ignore
    ).fetchall()
    if len(results) == 0:
        print("No data to copy")
        return
        
    # Preserve column data types by getting schema information from source  
    # For DuckDB, handle table names with db. prefix properly
    table_for_describe = source_tracker.engine.table_name
    if "." in table_for_describe:
        # For DuckDB: db."default" -> db.default (remove quotes around entire thing)
        parts = table_for_describe.split(".")
        if len(parts) == 2:
            table_for_describe = f'{parts[0]}."{parts[1]}"'
    
    source_schema = source_tracker.conn.execute(
        f"DESCRIBE {table_for_describe}"
    ).fetchall()
    source_column_types = {col[0]: col[1] for col in source_schema}
    
    # Add new columns to target with appropriate types
    new_column_count = 0
    for column, hash_value in source_tracker.dtypes.items():
        if column not in target_tracker._columns:
            new_column_count += 1
            # Use the original data type from source if available
            column_type = source_column_types.get(column)
            # Add column with proper data type
            if column_type:
                # Use ADD COLUMN with type instead of the generic add_column
                try:
                    target_tracker.conn.execute(
                        f"ALTER TABLE {target_tracker.engine.table_name} ADD COLUMN {column} {column_type}"
                    )
                except:
                    # Fallback to generic add_column if the direct approach fails
                    target_tracker.add_column(column, hash_value)
            else:
                target_tracker.add_column(column, hash_value)
            
    keys = [column[0] for column in source_tracker.engine._duckdb_types]
    size = len(keys)
    timestamp_ix, track_ix = keys.index(TRACKER_CONSTANTS.TIMESTAMP), keys.index(
        SCHEMA_PARAMS.TRACK_ID
    )
    count = 0
    if assets:
        logger.info("Copying assets")
        assets_count = 0
        source_assets = source_tracker.assets
        target_assets = target_tracker.assets
        for key, hash_value in source_assets.keys.items():
            target_assets.keys[key] = hash_value
            if hash_value not in target_assets.assets:
                target_assets.assets[hash_value] = source_assets.assets.get(hash_value)
                if hash_value not in target_assets.counts:
                    target_assets.counts[hash_value] = source_assets.counts.get(
                        hash_value
                    )
                else:
                    target_assets.counts[hash_value] += source_assets.counts.get(
                        hash_value
                    )
                assets_count += 1
        logger.info(f"Copied {assets_count} assets")
    logger.info("Copying events")
    target_tracker.conn.execute("BEGIN TRANSACTION")
    for event in results:
        if f"{event[timestamp_ix]}-{event[track_ix]}" in ids:
            continue
        count += 1
        values = list(event)
        with contextlib.suppress(duckdb.ConstraintException):
            target_tracker.conn.execute(
                f"INSERT INTO {target_tracker.engine.table_name} ({', '.join(keys)}) VALUES ({', '.join(['?' for _ in range(size)])})",
                values,
            )
    target_tracker.conn.execute("COMMIT TRANSACTION")
    total = target_tracker.count_all()
    logger.info(
        f"Copied {count} events and {new_column_count} new columns. New total is {total} events"
    )
    return count