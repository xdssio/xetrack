#!/usr/bin/env python3
"""
model_manager.py - Upload/download models using SQLite as index and DVC/git tags as storage.

Usage:
    # Promote best model from a sweep to production
    python model_manager.py promote --db results/experiments.db --run-id abc123

    # Add model to candidates
    python model_manager.py add-candidate --model-path path/to/model.bin --run-id abc123

    # Download model from a specific version/tag
    python model_manager.py download --version e0.1.0 --output model.bin

    # List all models with metrics
    python model_manager.py list --db results/experiments.db

    # Prune candidates to top-K
    python model_manager.py prune --db results/experiments.db --keep 5 --metric f1_score

    # Show current production model info
    python model_manager.py info --db results/experiments.db
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

try:
    import duckdb
except ImportError:
    duckdb = None


def _require_duckdb() -> None:
    """Exit with error if duckdb is not installed."""
    if duckdb is None:
        print("ERROR: duckdb required. Install with: pip install duckdb")
        sys.exit(1)


PRODUCTION_DIR = "models/production"
CANDIDATES_DIR = "models/candidates"
PRODUCTION_MODEL = f"{PRODUCTION_DIR}/model.bin"


def ensure_dirs() -> None:
    """Create production and candidates directories if they don't exist."""
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)


def promote_model(db_path: str, run_id: str | None = None, version: str | None = None,
                  metric: str = 'f1_score', table: str = 'predictions') -> None:
    """Promote a model to production. By run_id, version, or best metric."""
    _require_duckdb()
    ensure_dirs()
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")

    if run_id:
        row = con.execute(f"""
            SELECT * FROM db.{table} WHERE track_id = '{run_id}' LIMIT 1
        """).fetchone()
    elif version:
        row = con.execute(f"""
            SELECT * FROM db.{table} WHERE version = '{version}'
            ORDER BY {metric} DESC LIMIT 1
        """).fetchone()
    else:
        row = con.execute(f"""
            SELECT * FROM db.{table} ORDER BY {metric} DESC LIMIT 1
        """).fetchone()

    if not row:
        print("ERROR: No matching model found")
        sys.exit(1)

    cols = [r[0] for r in con.description]
    model_info = dict(zip(cols, row))
    track_id = model_info.get('track_id', 'unknown')

    # Check if model exists in candidates
    candidate_path = f"{CANDIDATES_DIR}/{track_id}_model.bin"
    if os.path.exists(candidate_path):
        shutil.copy2(candidate_path, PRODUCTION_MODEL)
        print(f"PROMOTED: {candidate_path} -> {PRODUCTION_MODEL}")
    else:
        print(f"WARNING: Candidate model not found at {candidate_path}")
        print(f"Model info: version={model_info.get('version')}, track_id={track_id}")
        print("You may need to copy the model manually or download from tag.")
        return

    # Print model info
    print(f"Version: {model_info.get('version', 'N/A')}")
    print(f"Track ID: {track_id}")
    for key in ['model', metric, 'accuracy', 'precision', 'recall']:
        if key in model_info and model_info[key] is not None:
            val = model_info[key]
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    con.close()


def add_candidate(model_path: str, run_id: str) -> None:
    """Add a model to the candidates directory."""
    ensure_dirs()
    dest = f"{CANDIDATES_DIR}/{run_id}_model.bin"
    shutil.copy2(model_path, dest)
    print(f"ADDED CANDIDATE: {model_path} -> {dest}")


def download_model(version: str, output_path: str) -> None:
    """Download a model from a specific git tag via DVC."""
    print(f"Downloading model from tag: {version}")

    # Stash current work (only if dirty)
    stash_result = subprocess.run(['git', 'stash'], capture_output=True, text=True)
    did_stash = 'No local changes' not in stash_result.stdout

    try:
        # Checkout tag
        subprocess.run(['git', 'checkout', version], check=True, capture_output=True)

        # Pull model from DVC
        result = subprocess.run(
            ['dvc', 'pull', f'{PRODUCTION_DIR}/model.bin.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            # Try pulling from candidates
            result = subprocess.run(
                ['dvc', 'pull', f'{CANDIDATES_DIR}/.dvc'],
                capture_output=True
            )

        # Copy model to output
        if os.path.exists(PRODUCTION_MODEL):
            shutil.copy2(PRODUCTION_MODEL, output_path)
            print(f"DOWNLOADED: {version} -> {output_path}")
        else:
            print(f"ERROR: Model not found at {PRODUCTION_MODEL} for tag {version}")

    finally:
        # Restore previous branch
        subprocess.run(['git', 'checkout', '-'], capture_output=True)
        if did_stash:
            subprocess.run(['git', 'stash', 'pop'], capture_output=True)
        subprocess.run(['dvc', 'checkout'], capture_output=True)


def list_models(db_path: str, table: str = 'predictions', metric: str = 'f1_score') -> None:
    """List all models with their metrics."""
    _require_duckdb()
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")

    con.execute(f"SELECT * FROM db.{table} LIMIT 0")
    cols = [r[0] for r in con.description]

    # Select relevant columns
    select_cols = ['version', 'track_id', 'model', 'timestamp']
    metric_cols = [c for c in cols if any(m in c.lower() for m in
                   ['f1', 'accuracy', 'precision', 'recall', 'loss', 'score'])]
    select_cols.extend(metric_cols)
    select_cols = [c for c in select_cols if c in cols]

    if not select_cols:
        select_cols = cols[:8]

    query = f"SELECT {', '.join(select_cols)} FROM db.{table} ORDER BY "
    if metric in cols:
        query += f"{metric} DESC"
    else:
        query += "timestamp DESC"

    results = con.execute(query).fetchall()

    # Print as table
    if results:
        header = select_cols
        print(" | ".join(f"{h:>15}" for h in header))
        print("-" * (17 * len(header)))
        for row in results:
            formatted = []
            for v in row:
                if isinstance(v, float):
                    formatted.append(f"{v:>15.4f}")
                elif v is None:
                    formatted.append(f"{'N/A':>15}")
                else:
                    formatted.append(f"{str(v):>15}")
            print(" | ".join(formatted))
    else:
        print("No models found in database.")

    # Show file system state
    print(f"\nProduction model: {'EXISTS' if os.path.exists(PRODUCTION_MODEL) else 'NOT FOUND'}")
    if os.path.exists(CANDIDATES_DIR):
        candidates = [f for f in os.listdir(CANDIDATES_DIR) if f.endswith('.bin')]
        print(f"Candidate models: {len(candidates)}")
        for c in sorted(candidates):
            size_mb = os.path.getsize(f"{CANDIDATES_DIR}/{c}") / 1024 / 1024
            print(f"  {c} ({size_mb:.1f} MB)")

    con.close()


def prune_candidates(db_path: str, keep: int = 5, metric: str = 'f1_score',
                     table: str = 'predictions') -> None:
    """Prune candidates to top-K based on metric."""
    _require_duckdb()
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")

    # Get ranked models
    results = con.execute(f"""
        SELECT track_id, version, {metric}
        FROM db.{table}
        WHERE track_id IS NOT NULL
        ORDER BY {metric} DESC
    """).fetchall()

    keep_ids = {r[0] for r in results[:keep]}
    remove_ids = [(r[0], r[1], r[2]) for r in results[keep:]]

    removed = 0
    for track_id, version, score in remove_ids:
        candidate_path = f"{CANDIDATES_DIR}/{track_id}_model.bin"
        if os.path.exists(candidate_path):
            os.remove(candidate_path)
            print(f"PRUNED: {track_id} (v={version}, {metric}={score:.4f})")
            removed += 1

    print(f"\nKept top {keep}, pruned {removed} candidates")
    print("Note: Pruned models still accessible via their git tags + dvc pull")
    print(f"Run: cd {CANDIDATES_DIR} && dvc add . to update tracking")

    con.close()


def show_info(db_path: str, table: str = 'predictions') -> None:
    """Show info about current production model."""
    if not os.path.exists(PRODUCTION_MODEL):
        print("No production model found.")
        return

    size_mb = os.path.getsize(PRODUCTION_MODEL) / 1024 / 1024
    print(f"Production model: {PRODUCTION_MODEL} ({size_mb:.1f} MB)")

    # Get latest tag
    try:
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0']
        ).decode().strip()
        tag_msg = subprocess.check_output(
            ['git', 'tag', '-l', tag, '-n9']
        ).decode().strip()
        print(f"Latest tag: {tag_msg}")
    except subprocess.CalledProcessError:
        print("No tags found")

    # Get DB stats
    if db_path and os.path.exists(db_path):
        _require_duckdb()
        con = duckdb.connect()
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")

        count = con.execute(f"SELECT COUNT(*) FROM db.{table}").fetchone()[0]
        versions = con.execute(f"""
            SELECT COUNT(DISTINCT version) FROM db.{table}
            WHERE version IS NOT NULL
        """).fetchone()[0]
        print(f"Database: {count} rows, {versions} versions")
        con.close()


def main() -> None:
    """CLI entry point for model management operations."""
    parser = argparse.ArgumentParser(description='Model manager for ML experiments')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Promote
    p = subparsers.add_parser('promote', help='Promote model to production')
    p.add_argument('--db', required=True, help='xetrack database path')
    p.add_argument('--run-id', help='Track ID of the model to promote')
    p.add_argument('--version', help='Version to promote (picks best by metric)')
    p.add_argument('--metric', default='f1_score', help='Metric to sort by')
    p.add_argument('--table', default='predictions')

    # Add candidate
    p = subparsers.add_parser('add-candidate', help='Add model to candidates')
    p.add_argument('--model-path', required=True, help='Path to model file')
    p.add_argument('--run-id', required=True, help='Track ID for the model')

    # Download
    p = subparsers.add_parser('download', help='Download model from git tag')
    p.add_argument('--version', required=True, help='Version/tag to download from')
    p.add_argument('--output', default='downloaded_model.bin', help='Output path')

    # List
    p = subparsers.add_parser('list', help='List all models')
    p.add_argument('--db', required=True, help='xetrack database path')
    p.add_argument('--metric', default='f1_score')
    p.add_argument('--table', default='predictions')

    # Prune
    p = subparsers.add_parser('prune', help='Prune candidates to top-K')
    p.add_argument('--db', required=True, help='xetrack database path')
    p.add_argument('--keep', type=int, default=5, help='Number of candidates to keep')
    p.add_argument('--metric', default='f1_score')
    p.add_argument('--table', default='predictions')

    # Info
    p = subparsers.add_parser('info', help='Show production model info')
    p.add_argument('--db', default='results/experiments.db')
    p.add_argument('--table', default='predictions')

    args = parser.parse_args()

    if args.command == 'promote':
        promote_model(args.db, args.run_id, args.version, args.metric, args.table)
    elif args.command == 'add-candidate':
        add_candidate(args.model_path, args.run_id)
    elif args.command == 'download':
        download_model(args.version, args.output)
    elif args.command == 'list':
        list_models(args.db, args.table, args.metric)
    elif args.command == 'prune':
        prune_candidates(args.db, args.keep, args.metric, args.table)
    elif args.command == 'info':
        show_info(args.db, args.table)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
