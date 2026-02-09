#!/usr/bin/env python3
"""
version_tag.py - Create annotated git tags with experiment metric descriptions.

Usage:
    # Auto-generate tag description from xetrack database
    python version_tag.py --version e0.1.0 --db results/experiments.db

    # Manual description
    python version_tag.py --version e0.1.0 --description "baseline PII detector"

    # List all experiment tags
    python version_tag.py --list

    # Show details of a specific tag
    python version_tag.py --show e0.1.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

try:
    import duckdb
except ImportError:
    duckdb = None


def get_git_hash() -> str:
    """Get current git commit hash."""
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).decode().strip()


def get_dvc_hash() -> str:
    """Get hash of dvc.lock if it exists."""
    try:
        return subprocess.check_output(
            ['git', 'hash-object', 'dvc.lock']
        ).decode().strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-dvc"


def get_metrics_from_db(db_path: str, version: str, table: str = 'predictions') -> dict[str, str]:
    """Extract key metrics from xetrack database for the given version."""
    if duckdb is None:
        print("WARNING: duckdb not installed, cannot auto-extract metrics")
        return {}

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")

    # Get columns
    con.execute(f"SELECT * FROM db.{table} LIMIT 0")
    cols = [r[0] for r in con.description]

    # Get the latest row for this version
    if 'version' in cols:
        rows = con.execute(f"""
            SELECT * FROM db.{table} 
            WHERE version = '{version}' 
            ORDER BY timestamp DESC LIMIT 1
        """).fetchall()
    else:
        rows = con.execute(f"""
            SELECT * FROM db.{table} 
            ORDER BY timestamp DESC LIMIT 1
        """).fetchall()

    if not rows:
        return {}

    row = dict(zip(cols, rows[0]))
    con.close()

    # Extract meaningful metrics (skip internal columns)
    skip_keys = {'timestamp', 'track_id', '_id', 'version', 'git_commit', 'dvc_hash'}
    metrics = {}
    for k, v in row.items():
        if k in skip_keys or v is None:
            continue
        if isinstance(v, float):
            metrics[k] = f"{v:.4f}"
        elif isinstance(v, (int, str, bool)):
            metrics[k] = str(v)

    return metrics


def build_tag_message(version: str, metrics: dict, description: str = "",
                      data_hash: str = "") -> str:
    """Build a formatted tag message from metrics."""
    parts = []

    # Add key metrics
    priority_keys = ['model', 'f1_score', 'accuracy', 'precision', 'recall',
                     'loss', 'learning_rate', 'batch_size', 'epoch']

    for key in priority_keys:
        if key in metrics:
            short_key = key.replace('_score', '').replace('_rate', '')
            parts.append(f"{short_key}={metrics[key]}")

    # Add remaining metrics (up to 3 more)
    remaining = [k for k in metrics if k not in priority_keys]
    for key in remaining[:3]:
        parts.append(f"{key}={metrics[key]}")

    # Add data hash
    if data_hash:
        parts.append(f"data={data_hash[:6]}")

    # Add description
    if description:
        parts.append(description)

    return " | ".join(parts) if parts else description or f"Experiment {version}"


def create_tag(version: str, message: str) -> None:
    """Create an annotated git tag."""
    try:
        subprocess.run(
            ['git', 'tag', '-a', version, '-m', message],
            check=True, capture_output=True
        )
        print(f"Created tag: {version}")
        print(f"Message: {message}")
    except subprocess.CalledProcessError as e:
        if b'already exists' in e.stderr:
            print(f"ERROR: Tag {version} already exists. Use a different version.")
            print(f"Existing tags: ", end="")
            list_tags()
        else:
            print(f"ERROR: {e.stderr.decode()}")
        sys.exit(1)


def list_tags(prefix: str = 'e') -> None:
    """List all experiment tags with descriptions."""
    try:
        output = subprocess.check_output(
            ['git', 'tag', '-l', f'{prefix}*', '-n9']
        ).decode().strip()
        if output:
            print(output)
        else:
            print(f"No tags found with prefix '{prefix}'")
    except subprocess.CalledProcessError:
        print("ERROR: Not a git repository")


def show_tag(tag: str) -> None:
    """Show full details of a tag."""
    try:
        output = subprocess.check_output(
            ['git', 'show', tag, '--no-patch']
        ).decode().strip()
        print(output)
    except subprocess.CalledProcessError:
        print(f"ERROR: Tag {tag} not found")


def suggest_next_version(prefix: str = 'e') -> str:
    """Suggest the next version number based on existing tags."""
    try:
        output = subprocess.check_output(
            ['git', 'tag', '-l', f'{prefix}*']
        ).decode().strip()
    except subprocess.CalledProcessError:
        return f"{prefix}0.1.0"

    if not output:
        return f"{prefix}0.1.0"

    versions = []
    for tag in output.split('\n'):
        tag = tag.strip()
        if tag.startswith(prefix):
            try:
                parts = tag[len(prefix):].split('.')
                versions.append(tuple(int(p) for p in parts))
            except ValueError:
                continue

    if not versions:
        return f"{prefix}0.1.0"

    latest = max(versions)
    # Suggest minor bump
    return f"{prefix}{latest[0]}.{latest[1] + 1}.0"


def main() -> None:
    """CLI entry point for version tagging."""
    parser = argparse.ArgumentParser(description='Create annotated git tags for ML experiments')

    parser.add_argument('--version', '-v', help='Version string (e.g., e0.1.0)')
    parser.add_argument('--db', help='xetrack SQLite database path')
    parser.add_argument('--table', default='predictions', help='Database table name')
    parser.add_argument('--description', '-d', default='', help='Human-readable description')
    parser.add_argument('--list', '-l', action='store_true', help='List all experiment tags')
    parser.add_argument('--show', '-s', help='Show details of a specific tag')
    parser.add_argument('--suggest', action='store_true', help='Suggest next version number')
    parser.add_argument('--prefix', default='e', help='Tag prefix (default: e)')
    parser.add_argument('--dry-run', action='store_true', help='Show tag message without creating')

    args = parser.parse_args()

    if args.list:
        list_tags(args.prefix)
        # Also list worktree experiment tags
        print("\nWorktree experiment tags:")
        list_tags('exp-')
        return

    if args.show:
        show_tag(args.show)
        return

    if args.suggest:
        suggested = suggest_next_version(args.prefix)
        print(f"Suggested next version: {suggested}")
        return

    if not args.version:
        suggested = suggest_next_version(args.prefix)
        print(f"ERROR: --version required. Suggested: {suggested}")
        sys.exit(1)

    # Build tag message
    metrics = {}
    if args.db:
        metrics = get_metrics_from_db(args.db, args.version, args.table)

    data_hash = get_dvc_hash()
    message = build_tag_message(args.version, metrics, args.description, data_hash)

    if args.dry_run:
        print(f"Would create tag: {args.version}")
        print(f"Message: {message}")
        return

    create_tag(args.version, message)


if __name__ == '__main__':
    main()
