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
import shutil
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    duckdb = None


def _require_duckdb() -> None:
    """Exit with error if duckdb is not installed."""
    if duckdb is None:
        print("ERROR: duckdb required. Install with: pip install duckdb")
        sys.exit(1)


def merge_sqlite(base_db: str, exp_dbs: list[str], output_db: str,
                 table: str = 'predictions', id_column: str = 'track_id') -> None:
    """Merge: append all experiment rows into base. Non-destructive."""
    _require_duckdb()
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
    """Rebase: replace matching rows from experiment into base."""
    _require_duckdb()
    if key_columns is None:
        key_columns = ['version']

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
    """
    _require_duckdb()
    con = duckdb.connect()

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
