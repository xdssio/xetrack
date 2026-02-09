#!/usr/bin/env python3
"""
Validate benchmark for common issues:
- Data leaks (same input evaluated multiple times with different track_ids)
- Duplicate executions (same inputs executed multiple times)
- Missing parameters
- Failed executions

Usage:
    python validate_benchmark.py <db_path> <table_name> [--engine=duckdb]
"""

import sys
import argparse
from pathlib import Path

def validate_benchmark(db_path: str, table: str = "predictions", engine: str = "duckdb"):
    """Run validation checks on benchmark database."""
    try:
        from xetrack import Reader
    except ImportError:
        print("❌ Error: xetrack not installed. Run: pip install xetrack[duckdb]")
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"❌ Error: Database not found: {db_path}")
        sys.exit(1)

    print(f"🔍 Validating benchmark: {db_path} (table: {table})")
    print("=" * 60)

    # Read data
    try:
        reader = Reader(db=db_path, engine=engine, table=table)
        df = reader.to_df()
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        sys.exit(1)

    if df.empty:
        print("⚠️  Warning: Table is empty")
        return

    total_rows = len(df)
    print(f"📊 Total rows: {total_rows}")
    print()

    # Check 1: Data leaks (same input_id with different track_ids)
    print("1️⃣  Checking for data leaks...")
    if 'input_id' in df.columns:
        duplicates = df.groupby('input_id')['track_id'].nunique()
        leaks = duplicates[duplicates > 1]

        if len(leaks) > 0:
            print(f"   ❌ Found {len(leaks)} input_ids evaluated multiple times:")
            for input_id, count in leaks.head(5).items():
                print(f"      - input_id={input_id}: {count} different track_ids")
            if len(leaks) > 5:
                print(f"      ... and {len(leaks) - 5} more")
            print(f"   ⚠️  This suggests re-running without cache or failed cache lookup")
        else:
            print(f"   ✅ No data leaks (all input_ids have unique track_ids)")
    else:
        print("   ⚠️  Skipped: No 'input_id' column found")
    print()

    # Check 2: Duplicate track_id + input_id combinations
    print("2️⃣  Checking for duplicate executions...")
    if 'input_id' in df.columns and 'track_id' in df.columns:
        duplicates = df.groupby(['track_id', 'input_id']).size()
        dups = duplicates[duplicates > 1]

        if len(dups) > 0:
            print(f"   ❌ Found {len(dups)} duplicate (track_id, input_id) pairs:")
            for (track_id, input_id), count in dups.head(5).items():
                print(f"      - track_id={track_id}, input_id={input_id}: {count} times")
            print(f"   ⚠️  This suggests the same input was logged multiple times in one run")
        else:
            print(f"   ✅ No duplicate executions")
    else:
        print("   ⚠️  Skipped: Missing 'input_id' or 'track_id' column")
    print()

    # Check 3: Missing parameters
    print("3️⃣  Checking for missing parameters...")
    param_cols = [col for col in df.columns if col.startswith('params_')]
    if param_cols:
        missing = {}
        for col in param_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                missing[col] = null_count

        if missing:
            print(f"   ⚠️  Found columns with NULL values:")
            for col, count in missing.items():
                pct = 100.0 * count / total_rows
                print(f"      - {col}: {count} nulls ({pct:.1f}%)")
        else:
            print(f"   ✅ No missing parameters in {len(param_cols)} param columns")
    else:
        print("   ⚠️  No parameter columns found (looking for 'params_*')")
    print()

    # Check 4: Failed executions
    print("4️⃣  Checking for failed executions...")
    if 'error' in df.columns:
        errors = df[df['error'].notna() & (df['error'] != '')]
        error_count = len(errors)

        if error_count > 0:
            error_pct = 100.0 * error_count / total_rows
            print(f"   ⚠️  Found {error_count} failed executions ({error_pct:.1f}%)")

            # Show error types
            error_types = errors['error'].value_counts().head(5)
            print(f"   Top error types:")
            for error, count in error_types.items():
                error_short = error[:60] + "..." if len(error) > 60 else error
                print(f"      - {error_short}: {count} times")
        else:
            print(f"   ✅ No failed executions")
    else:
        print("   ⚠️  No 'error' column found")
    print()

    # Check 5: Cache effectiveness
    print("5️⃣  Checking cache effectiveness...")
    if 'cache' in df.columns:
        cache_hits = (df['cache'] != '').sum()
        cache_misses = (df['cache'] == '').sum()
        hit_rate = 100.0 * cache_hits / total_rows if total_rows > 0 else 0

        print(f"   Cache hits: {cache_hits} ({hit_rate:.1f}%)")
        print(f"   Cache misses: {cache_misses} ({100-hit_rate:.1f}%)")

        if hit_rate > 50:
            print(f"   ✅ Good cache hit rate")
        elif hit_rate > 0:
            print(f"   ⚠️  Low cache hit rate - consider checking cache configuration")
        else:
            print(f"   ℹ️  No cache hits (expected for first run)")
    else:
        print("   ℹ️  No 'cache' column found (caching not enabled)")
    print()

    # Summary
    print("=" * 60)
    print("✅ Validation complete!")
    print()
    print("💡 Tips:")
    print("   - Use caching to prevent duplicate executions")
    print("   - Ensure all parameters are in frozen dataclass")
    print("   - Check error column for failed executions")
    print("   - Use input_id to track individual data points")


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark database")
    parser.add_argument("db_path", help="Path to database file")
    parser.add_argument("table", nargs="?", default="predictions", help="Table name (default: predictions)")
    parser.add_argument("--engine", default="duckdb", choices=["sqlite", "duckdb"],
                       help="Database engine (default: duckdb)")

    args = parser.parse_args()
    validate_benchmark(args.db_path, args.table, args.engine)


if __name__ == "__main__":
    main()
