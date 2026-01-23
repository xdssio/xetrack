import contextlib
from typing import List, Dict, Set, Tuple, Optional
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


def _get_existing_event_ids(tracker: Tracker) -> Set[str]:
    """
    Get set of existing event IDs in the target database.

    :param tracker: Target tracker instance
    :return: Set of composite IDs (timestamp-track_id)
    """
    results = tracker.conn.execute(  # type: ignore
        f"SELECT timestamp, track_id FROM {tracker.engine.table_name}"
    ).fetchall()  # type: ignore
    return {f"{row[0]}-{row[1]}" for row in results}


def _copy_assets(source_db: str, target_db: str, table: str) -> int:
    """
    Copy all assets from source to target database.
    Assets are stored globally in the database, not per-table.

    :param source_db: Source database path
    :param target_db: Target database path
    :param table: Table name to use for accessing assets
    :return: Number of assets copied
    """
    source_tracker = Tracker(db=source_db, engine='duckdb', table=table)
    target_tracker = Tracker(db=target_db, engine='duckdb', table=table)

    source_assets = source_tracker.assets
    target_assets = target_tracker.assets

    assets_copied = 0

    # Copy all asset data (hash -> pickled object)
    for hash_value in source_assets.assets.keys():
        if hash_value not in target_assets.assets:
            target_assets.assets[hash_value] = source_assets.assets[hash_value]

            # Copy or update reference counts
            source_count = source_assets.counts.get(hash_value, 0)
            if hash_value not in target_assets.counts:
                target_assets.counts[hash_value] = source_count
            else:
                target_assets.counts[hash_value] += source_count

            assets_copied += 1

    # Copy key mappings (key name -> hash)
    # Note: Keys with same name from different tables will overwrite each other
    for key, hash_value in source_assets.keys.items():
        target_assets.keys[key] = hash_value

    return assets_copied


def _add_missing_columns(
    source_tracker: Tracker,
    target_tracker: Tracker,
    source_column_types: Dict[str, str]
) -> int:
    """
    Add columns that exist in source but not in target.

    :param source_tracker: Source tracker instance
    :param target_tracker: Target tracker instance
    :param source_column_types: Map of column names to their types
    :return: Number of columns added
    """
    columns_added = 0
    target_columns = set(target_tracker.engine.columns)

    for column, dtype_hash in source_tracker.dtypes.items():
        if column not in target_columns:
            column_type = source_column_types.get(column)

            if column_type:
                try:
                    target_tracker.conn.execute(
                        f"ALTER TABLE {target_tracker.engine.table_name} "
                        f"ADD COLUMN {column} {column_type}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add column {column} with type {column_type}: {e}")
                    target_tracker.add_column(column, dtype_hash)
            else:
                target_tracker.add_column(column, dtype_hash)

            columns_added += 1

    return columns_added


def _get_table_schema(tracker: Tracker) -> Dict[str, str]:
    """
    Get schema information for a table.

    :param tracker: Tracker instance
    :return: Dictionary mapping column names to their types
    """
    table_name = tracker.engine.table_name

    # Handle DuckDB table naming (db.table_name)
    if "." in table_name:
        parts = table_name.split(".")
        if len(parts) == 2:
            table_name = f'{parts[0]}."{parts[1]}"'

    schema_rows = tracker.conn.execute(f"DESCRIBE {table_name}").fetchall()
    return {col[0]: col[1] for col in schema_rows}


def _copy_table_events(
    source_tracker: Tracker,
    target_tracker: Tracker,
    existing_event_ids: Set[str]
) -> int:
    """
    Copy events from source table to target table.

    :param source_tracker: Source tracker instance
    :param target_tracker: Target tracker instance
    :param existing_event_ids: Set of event IDs already in target
    :return: Number of events copied
    """
    import duckdb

    # Fetch all events from source
    source_events = source_tracker.conn.execute(
        f"SELECT * FROM {source_tracker.engine.table_name}"
    ).fetchall()

    if not source_events:
        return 0

    # Get column information
    column_info = source_tracker.engine.execute_sql(
        f"PRAGMA table_info({source_tracker.engine.table_name})"
    )
    columns = [row['name'] for row in column_info.to_dict('records')]

    # Find indices for timestamp and track_id
    timestamp_idx = columns.index(TRACKER_CONSTANTS.TIMESTAMP)
    track_id_idx = columns.index(SCHEMA_PARAMS.TRACK_ID)

    # Insert events
    events_copied = 0
    target_tracker.conn.execute("BEGIN TRANSACTION")

    for event in source_events:
        event_id = f"{event[timestamp_idx]}-{event[track_id_idx]}"

        if event_id in existing_event_ids:
            continue

        values = list(event)
        placeholders = ', '.join(['?' for _ in range(len(columns))])

        with contextlib.suppress(duckdb.ConstraintException):
            target_tracker.conn.execute(
                f"INSERT INTO {target_tracker.engine.table_name} "
                f"({', '.join(columns)}) VALUES ({placeholders})",
                values
            )
            events_copied += 1

    target_tracker.conn.execute("COMMIT TRANSACTION")

    return events_copied


def _copy_single_table(source_db: str, target_db: str, table: str) -> Tuple[int, int]:
    """
    Copy a single table from source to target database.

    :param source_db: Source database path
    :param target_db: Target database path
    :param table: Table name to copy
    :return: Tuple of (events_copied, columns_added)
    """
    logger.info(f"Copying table: {table}")

    source_tracker = Tracker(db=source_db, engine='duckdb', table=table)
    target_tracker = Tracker(db=target_db, engine='duckdb', table=table)

    # Get existing events in target to avoid duplicates
    existing_event_ids = _get_existing_event_ids(target_tracker)

    # Get source schema for column type preservation
    source_schema = _get_table_schema(source_tracker)

    # Add missing columns to target
    columns_added = _add_missing_columns(source_tracker, target_tracker, source_schema)

    # Copy events
    logger.info(f"Copying events from table {table}")
    events_copied = _copy_table_events(source_tracker, target_tracker, existing_event_ids)

    total_events = target_tracker.count_all()
    logger.info(
        f"Copied {events_copied} events and {columns_added} new columns from table {table}. "
        f"Table now has {total_events} events"
    )

    return events_copied, columns_added


def copy(
    source: str,
    target: str,
    assets: bool = True,
    tables: Optional[List[str]] = None
) -> int:
    """
    Copy data from one tracker database to another.

    This function copies events from specified tables and optionally copies assets.
    Assets are stored globally in the database, shared across all tables.

    :param source: Path to source database file
    :param target: Path to target database file
    :param assets: Whether to copy assets (default: True)
    :param tables: List of table names to copy. If None or empty, defaults to ['default']
    :return: Total number of events copied across all tables

    Example:
        >>> copy('source.db', 'target.db', tables=['experiments', 'validation'])
        >>> copy('source.db', 'target.db')  # Copies 'default' table
    """
    # Validate and set default tables
    if not tables:
        tables = ['default']

    total_events_copied = 0
    total_columns_added = 0

    # Copy assets once (shared across all tables)
    if assets:
        logger.info("Copying assets")
        assets_copied = _copy_assets(source, target, tables[0])
        logger.info(f"Copied {assets_copied} assets")

    # Copy each table
    for table in tables:
        events_copied, columns_added = _copy_single_table(source, target, table)
        total_events_copied += events_copied
        total_columns_added += columns_added

    logger.info(
        f"Total: Copied {total_events_copied} events and {total_columns_added} new columns "
        f"across {len(tables)} table(s)"
    )

    return total_events_copied