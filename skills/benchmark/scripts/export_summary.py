#!/usr/bin/env python3
"""
Export benchmark results as markdown summary.

Usage:
    python export_summary.py <db_path> <table_name> [--engine=duckdb] > RESULTS.md
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

def export_summary(db_path: str, table: str = "events", engine: str = "duckdb"):
    """Generate markdown summary of benchmark results."""
    try:
        from xetrack import Reader
        import pandas as pd
    except ImportError:
        print("❌ Error: xetrack not installed. Run: pip install xetrack[duckdb]")
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"❌ Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Read data
    reader = Reader(db=db_path, engine=engine, table=table)
    df = reader.to_df()

    if df.empty:
        print("⚠️  Table is empty", file=sys.stderr)
        return

    # Generate markdown
    print(f"# Benchmark Results")
    print()
    print(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"**Database:** `{db_path}`")
    print(f"**Table:** `{table}`")
    print(f"**Total Executions:** {len(df)}")
    print()

    # Experiment metadata
    if 'track_id' in df.columns:
        unique_runs = df['track_id'].nunique()
        print(f"**Unique Runs:** {unique_runs}")

    if 'timestamp' in df.columns:
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        print(f"**Time Range:** {start_time} to {end_time}")

    print()

    # Parameter combinations tested
    param_cols = [col for col in df.columns if col.startswith('params_')]
    if param_cols:
        print("## Parameter Combinations")
        print()
        for col in param_cols:
            unique_vals = df[col].dropna().unique()
            print(f"- **{col.replace('params_', '')}**: {', '.join(map(str, unique_vals))}")
        print()

    # Performance metrics
    print("## Performance Metrics")
    print()

    if 'latency' in df.columns:
        print("### Latency Statistics")
        print()
        latency_stats = df['latency'].describe()
        print(f"- **Mean:** {latency_stats['mean']:.4f}s")
        print(f"- **Median:** {latency_stats['50%']:.4f}s")
        print(f"- **Min:** {latency_stats['min']:.4f}s")
        print(f"- **Max:** {latency_stats['max']:.4f}s")
        print(f"- **Std Dev:** {latency_stats['std']:.4f}s")
        print()

    # Accuracy by parameter
    if 'prediction' in df.columns and 'ground_truth' in df.columns:
        print("### Accuracy by Configuration")
        print()

        if param_cols:
            accuracy_df = df.groupby(param_cols).apply(
                lambda g: pd.Series({
                    'accuracy': (g['prediction'] == g['ground_truth']).mean(),
                    'count': len(g)
                })
            ).reset_index()

            print(accuracy_df.to_markdown(index=False))
        else:
            overall_accuracy = (df['prediction'] == df['ground_truth']).mean()
            print(f"**Overall Accuracy:** {overall_accuracy:.4f}")
        print()

    # Error analysis
    if 'error' in df.columns:
        errors = df[df['error'].notna() & (df['error'] != '')]
        if len(errors) > 0:
            print("## Error Analysis")
            print()
            print(f"**Total Errors:** {len(errors)} ({100*len(errors)/len(df):.1f}%)")
            print()

            error_counts = errors['error'].value_counts().head(5)
            print("**Top Errors:**")
            for error, count in error_counts.items():
                error_short = error[:80] + "..." if len(error) > 80 else error
                print(f"- `{error_short}`: {count} times")
            print()

    # Cache statistics
    if 'cache' in df.columns:
        cache_hits = (df['cache'] != '').sum()
        total = len(df)
        hit_rate = 100.0 * cache_hits / total if total > 0 else 0

        print("## Cache Statistics")
        print()
        print(f"- **Cache Hits:** {cache_hits} ({hit_rate:.1f}%)")
        print(f"- **Cache Misses:** {total - cache_hits} ({100-hit_rate:.1f}%)")
        print()

    # Footer
    print("---")
    print()
    print("*Generated with xetrack benchmark skill*")


def main():
    parser = argparse.ArgumentParser(description="Export benchmark summary as markdown")
    parser.add_argument("db_path", help="Path to database file")
    parser.add_argument("table", nargs="?", default="events", help="Table name (default: events)")
    parser.add_argument("--engine", default="duckdb", choices=["sqlite", "duckdb"],
                       help="Database engine (default: duckdb)")

    args = parser.parse_args()
    export_summary(args.db_path, args.table, args.engine)


if __name__ == "__main__":
    main()
