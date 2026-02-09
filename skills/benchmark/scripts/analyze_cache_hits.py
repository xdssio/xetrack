#!/usr/bin/env python3
"""
Analyze cache effectiveness in benchmark.

Usage:
    python analyze_cache_hits.py <db_path> <table_name> [--engine=duckdb]
"""

import sys
import argparse
from pathlib import Path

def analyze_cache(db_path: str, table: str = "predictions", engine: str = "duckdb"):
    """Analyze cache hit/miss patterns."""
    try:
        from xetrack import Reader
    except ImportError:
        print("❌ Error: xetrack not installed. Run: pip install xetrack[duckdb]")
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"❌ Error: Database not found: {db_path}")
        sys.exit(1)

    print(f"📊 Cache Analysis: {db_path} (table: {table})")
    print("=" * 60)

    # Read data
    reader = Reader(db=db_path, engine=engine, table=table)
    df = reader.to_df()

    if df.empty:
        print("⚠️  Table is empty")
        return

    if 'cache' not in df.columns:
        print("⚠️  No 'cache' column found - caching not enabled")
        print()
        print("To enable caching:")
        print("  tracker = Tracker(db='...', cache='cache_dir')")
        return

    total = len(df)
    cache_hits = (df['cache'] != '').sum()
    cache_misses = (df['cache'] == '').sum()
    hit_rate = 100.0 * cache_hits / total if total > 0 else 0

    print(f"Total executions: {total}")
    print(f"Cache hits: {cache_hits} ({hit_rate:.1f}%)")
    print(f"Cache misses: {cache_misses} ({100-hit_rate:.1f}%)")
    print()

    # Breakdown by parameter combinations
    param_cols = [col for col in df.columns if col.startswith('params_')]
    if param_cols:
        print(f"Cache effectiveness by parameter combination:")
        print("-" * 60)

        # Group by all param columns
        grouped = df.groupby(param_cols, dropna=False)

        for params, group in grouped:
            group_total = len(group)
            group_hits = (group['cache'] != '').sum()
            group_hit_rate = 100.0 * group_hits / group_total if group_total > 0 else 0

            # Format param values
            if len(param_cols) == 1:
                params = [params]
            param_str = ", ".join([f"{col.replace('params_', '')}={val}"
                                   for col, val in zip(param_cols, params)])

            print(f"  {param_str}")
            print(f"    Total: {group_total}, Hits: {group_hits}, Rate: {group_hit_rate:.1f}%")

    print()
    print("=" * 60)

    # Recommendations
    if hit_rate == 0:
        print("💡 No cache hits detected. This is expected for first run.")
        print("   Re-run the same benchmark to see cache in action!")
    elif hit_rate < 30:
        print("⚠️  Low cache hit rate. Possible issues:")
        print("   - Running with different parameters each time")
        print("   - Cache directory was cleared")
        print("   - Function arguments are not hashable (lists/dicts)")
    elif hit_rate < 70:
        print("✅ Moderate cache hit rate. Good if testing new configurations.")
    else:
        print("✅ Excellent cache hit rate! Caching is working well.")


def main():
    parser = argparse.ArgumentParser(description="Analyze cache effectiveness")
    parser.add_argument("db_path", help="Path to database file")
    parser.add_argument("table", nargs="?", default="predictions", help="Table name (default: predictions)")
    parser.add_argument("--engine", default="duckdb", choices=["sqlite", "duckdb"],
                       help="Database engine (default: duckdb)")

    args = parser.parse_args()
    analyze_cache(args.db_path, args.table, args.engine)


if __name__ == "__main__":
    main()
