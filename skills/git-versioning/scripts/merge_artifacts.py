#!/usr/bin/env python3
"""
merge_artifacts.py - DuckDB-based merge/rebase engine for SQLite databases and parquet files.

Usage:
    # Merge SQLite databases
    python merge_artifacts.py --strategy merge --base-db main.db --exp-db exp.db --output-db merged.db

    # Rebase SQLite databases
    python merge_artifacts.py --strategy rebase --base-db main.db --exp-db exp.db --output-db main.db --key-columns version

    # Merge parquet files
    python merge_artifacts.py --strategy merge --base-data main.parquet --exp-data exp.parquet --output-data merged.parquet

    # Combined: merge DB + rebase data
    python merge_artifacts.py --strategy merge --base-db main.db --exp-db exp.db --output-db main.db \
        --data-strategy rebase --base-data main.parquet --exp-data exp.parquet --output-data main.parquet --key-columns id
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    duckdb = None

# Safe SQL identifier pattern: letters, digits, underscores; must start with letter/underscore.
_SAFE_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _validate_identifier(name: str, label: str = 'identifier') -> str:
    """Validate that a string is a safe SQL identifier.

    Args:
        name: The identifier string to validate.
        label: Human-readable label for error messages (e.g. 'table', 'column').

    Returns:
        The validated identifier string (unchanged).

    Raises:
        ValueError: If the identifier contains unsafe characters.
    """
    if not name or not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid SQL {label}: {name!r}. "
            f"Must match /^[A-Za-z_][A-Za-z0-9_]*$/"
        )
    return name


def _validate_identifiers(names: list[str], label: str = 'column') -> list[str]:
    """Validate a list of SQL identifiers.

    Args:
        names: List of identifier strings to validate.
        label: Human-readable label for error messages.

    Returns:
        The validated list (unchanged).

    Raises:
        ValueError: If any identifier is invalid or the list is empty.
    """
    if not names:
        raise ValueError(f"At least one {label} name is required")
    for name in names:
        _validate_identifier(name, label)
    return names


def _require_duckdb() -> None:
    """Exit with error if duckdb is not installed."""
    if duckdb is None:
        print("ERROR: duckdb required. Install with: pip install duckdb")
        sys.exit(1)


def merge_sqlite(base_db: str, exp_dbs: list[str], output_db: str,
                 table: str = 'predictions', id_column: str = 'track_id') -> None:
    """Merge: append all experiment rows into base. Non-destructive.

    Args:
        base_db: Path to the base SQLite database.
        exp_dbs: List of experiment database paths to merge in.
        output_db: Output database path (can be same as base_db for in-place).
        table: Table name to merge. Must be a valid SQL identifier.
        id_column: Column used for deduplication. Must be a valid SQL identifier.
    """
    _require_duckdb()
    _validate_identifier(table, 'table')
    _validate_identifier(id_column, 'column')

    if base_db != output_db:
        shutil.copy2(base_db, output_db)

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{output_db}' AS target_db (TYPE SQLITE)")

    for exp_db in exp_dbs:
        con.execute(f"ATTACH '{exp_db}' AS exp_db (TYPE SQLITE)")

        # Get columns that exist in both tables
        target_cols = set(
            col[0] for col in con.execute(f"SELECT * FROM target_db.{table} LIMIT 0").description
        )
        exp_cols = set(
            col[0] for col in con.execute(f"SELECT * FROM exp_db.{table} LIMIT 0").description
        )
        common_cols = target_cols & exp_cols

        if not common_cols:
            print(f"WARNING: No common columns between {output_db} and {exp_db}, skipping")
            con.execute("DETACH exp_db")
            continue

        cols = ', '.join(sorted(common_cols))

        # Insert rows not already present
        count_before = con.execute(f"SELECT COUNT(*) FROM target_db.{table}").fetchone()[0]
        con.execute(f"""
            INSERT INTO target_db.{table} ({cols})
            SELECT {cols} FROM exp_db.{table}
            WHERE {id_column} NOT IN (SELECT {id_column} FROM target_db.{table})
        """)
        count_after = con.execute(f"SELECT COUNT(*) FROM target_db.{table}").fetchone()[0]
        added = count_after - count_before

        print(f"MERGE DB: {exp_db} -> {output_db} | +{added} rows (skipped {con.execute(f'SELECT COUNT(*) FROM exp_db.{table}').fetchone()[0] - added} duplicates)")
        con.execute("DETACH exp_db")

    con.close()


def rebase_sqlite(base_db: str, exp_dbs: list[str], output_db: str,
                  table: str = 'predictions', key_columns: list[str] | None = None) -> None:
    """Rebase: replace matching rows from experiment into base.

    Args:
        base_db: Path to the base SQLite database.
        exp_dbs: List of experiment database paths.
        output_db: Output database path (can be same as base_db for in-place).
        table: Table name to rebase. Must be a valid SQL identifier.
        key_columns: Columns forming the composite key for matching. Each must be a valid SQL identifier.
    """
    _require_duckdb()
    _validate_identifier(table, 'table')
    if key_columns is None:
        key_columns = ['version']
    _validate_identifiers(key_columns, 'key column')

    if base_db != output_db:
        shutil.copy2(base_db, output_db)

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{output_db}' AS target_db (TYPE SQLITE)")

    for exp_db in exp_dbs:
        con.execute(f"ATTACH '{exp_db}' AS exp_db (TYPE SQLITE)")

        # Delete matching rows using composite key (all key columns must match simultaneously)
        key_condition = " AND ".join(
            f"t.{k} = e.{k}" for k in key_columns
        )
        deleted = con.execute(f"""
            SELECT COUNT(*) FROM target_db.{table} t
            WHERE EXISTS (
                SELECT 1 FROM exp_db.{table} e
                WHERE {key_condition}
            )
        """).fetchone()[0]
        con.execute(f"""
            DELETE FROM target_db.{table}
            WHERE rowid IN (
                SELECT t.rowid FROM target_db.{table} t
                WHERE EXISTS (
                    SELECT 1 FROM exp_db.{table} e
                    WHERE {key_condition}
                )
            )
        """)

        # Insert all experiment rows
        inserted = con.execute(f"SELECT COUNT(*) FROM exp_db.{table}").fetchone()[0]
        con.execute(f"INSERT INTO target_db.{table} SELECT * FROM exp_db.{table}")

        print(f"REBASE DB: {exp_db} -> {output_db} | deleted {deleted}, inserted {inserted} rows (key: {key_columns})")
        con.execute("DETACH exp_db")

    con.close()


def merge_parquet(base_path: str, exp_paths: list[str], output_path: str) -> None:
    """Merge: append experiment rows into base. New columns from exp are added (filled with NULL in base rows).

    Merge direction: base keeps all its rows, exp rows are appended.
    Schema mismatches are handled by UNION ALL BY NAME (missing columns become NULL).

    Args:
        base_path: Path to the base parquet file.
        exp_paths: List of experiment parquet file paths.
        output_path: Output parquet file path.
    """
    _require_duckdb()
    con = duckdb.connect()

    base_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{base_path}')").fetchone()[0]

    # Build UNION ALL BY NAME: base first (keeps all), then experiments
    sources = [f"SELECT * FROM read_parquet('{base_path}')"]
    for exp_path in exp_paths:
        sources.append(f"SELECT * FROM read_parquet('{exp_path}')")

    query = " UNION ALL BY NAME ".join(sources)

    # Write to temp file first if output == base (avoid read+write same file)
    if str(Path(output_path).resolve()) == str(Path(base_path).resolve()):
        tmp_path = output_path + '.tmp'
        con.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET)")
        shutil.move(tmp_path, output_path)
    else:
        con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")

    # Report
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    print(f"MERGE DATA: {len(exp_paths)} files -> {output_path} | {base_rows} base + {total_rows - base_rows} new = {total_rows} total rows")

    con.close()


def rebase_parquet(base_path: str, exp_paths: list[str], output_path: str,
                   key_columns: list[str]) -> None:
    """Rebase: replace matching rows, experiment takes priority.

    Rebase direction: experiment rows win. For each composite key, if the key
    exists in any experiment file, the experiment row replaces the base row.
    New columns from experiments are added (base rows get NULL for those columns).

    Args:
        base_path: Path to the base parquet file.
        exp_paths: List of experiment parquet file paths.
        output_path: Output parquet file path.
        key_columns: Columns forming the composite key. Each must be a valid SQL identifier
                     and must exist in the parquet schemas.

    Raises:
        ValueError: If key_columns is empty, contains invalid identifiers, or references
                    columns not present in the parquet files.
    """
    _require_duckdb()
    _validate_identifiers(key_columns, 'key column')

    con = duckdb.connect()

    # Verify key columns exist in base parquet schema
    base_cols = set(
        r[0] for r in con.execute(f"SELECT * FROM read_parquet('{base_path}') LIMIT 0").description
    )
    missing_in_base = [k for k in key_columns if k not in base_cols]
    if missing_in_base:
        con.close()
        raise ValueError(
            f"Key column(s) {missing_in_base} not found in base parquet '{base_path}'. "
            f"Available columns: {sorted(base_cols)}"
        )

    # Verify key columns exist in at least the first experiment file
    if exp_paths:
        exp_cols = set(
            r[0] for r in con.execute(f"SELECT * FROM read_parquet('{exp_paths[0]}') LIMIT 0").description
        )
        missing_in_exp = [k for k in key_columns if k not in exp_cols]
        if missing_in_exp:
            con.close()
            raise ValueError(
                f"Key column(s) {missing_in_exp} not found in experiment parquet '{exp_paths[0]}'. "
                f"Available columns: {sorted(exp_cols)}"
            )

    # Build experiment union (all experiment rows, later ones override earlier)
    exp_union = " UNION ALL BY NAME ".join(
        f"SELECT * FROM read_parquet('{p}')" for p in exp_paths
    )

    # Composite key match: all key columns must match simultaneously
    keys_condition = " AND ".join(
        f"b.{k} = e.{k}" for k in key_columns
    )

    # Experiment rows first, then non-matching base rows
    query = f"""
        SELECT * FROM ({exp_union})
        UNION ALL BY NAME
        SELECT b.* FROM read_parquet('{base_path}') b
        WHERE NOT EXISTS (
            SELECT 1 FROM ({exp_union}) e
            WHERE {keys_condition}
        )
    """

    # Write to temp file first if output overlaps with any input (avoid read+write same file)
    all_inputs = [str(Path(p).resolve()) for p in [base_path] + exp_paths]
    if str(Path(output_path).resolve()) in all_inputs:
        tmp_path = output_path + '.tmp'
        con.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET)")
        shutil.move(tmp_path, output_path)
    else:
        con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")

    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    print(f"REBASE DATA: {len(exp_paths)} files -> {output_path} | {total} total rows (key: {key_columns})")

    con.close()


def main() -> None:
    """CLI entry point for the merge/rebase engine."""
    parser = argparse.ArgumentParser(description='DuckDB merge/rebase engine for ML artifacts')

    # Strategy
    parser.add_argument('--strategy', choices=['merge', 'rebase'], required=True,
                        help='Merge (append) or rebase (replace)')
    parser.add_argument('--data-strategy', choices=['merge', 'rebase'],
                        help='Override strategy for data files (defaults to --strategy)')

    # SQLite database arguments
    parser.add_argument('--base-db', help='Base SQLite database path')
    parser.add_argument('--exp-db', nargs='+', help='Experiment SQLite database path(s)')
    parser.add_argument('--output-db', help='Output SQLite database path')
    parser.add_argument('--table', default='predictions', help='SQLite table name (default: predictions)')
    parser.add_argument('--id-column', default='track_id', help='Unique ID column for dedup (default: track_id)')

    # Parquet data arguments
    parser.add_argument('--base-data', help='Base parquet data path')
    parser.add_argument('--exp-data', nargs='+', help='Experiment parquet data path(s)')
    parser.add_argument('--output-data', help='Output parquet data path')

    # Key columns (for rebase)
    parser.add_argument('--key-columns', nargs='+', default=['version'],
                        help='Key columns for rebase matching (default: version)')

    args = parser.parse_args()

    data_strategy = args.data_strategy or args.strategy

    # Process SQLite databases
    if args.base_db and args.exp_db:
        output_db = args.output_db or args.base_db
        if args.strategy == 'merge':
            merge_sqlite(args.base_db, args.exp_db, output_db, args.table, args.id_column)
        else:
            rebase_sqlite(args.base_db, args.exp_db, output_db, args.table, args.key_columns)

    # Process parquet files
    if args.base_data and args.exp_data:
        output_data = args.output_data or args.base_data
        if data_strategy == 'merge':
            merge_parquet(args.base_data, args.exp_data, output_data)
        else:
            if not args.key_columns:
                print("ERROR: --key-columns required for data rebase")
                sys.exit(1)
            rebase_parquet(args.base_data, args.exp_data, output_data, args.key_columns)

    print("Done.")


if __name__ == '__main__':
    main()
