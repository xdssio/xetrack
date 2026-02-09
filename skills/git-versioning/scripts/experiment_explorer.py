#!/usr/bin/env python3
"""
experiment_explorer.py - Read-only exploration and retrieval of past ML experiments.

Browse experiments, retrieve specific artifacts (models, data, databases),
and create disposable worktrees for full experiment exploration.

Usage:
    # List all experiments with key metrics
    python experiment_explorer.py list --db results/experiments.db

    # Show full details of a specific experiment
    python experiment_explorer.py show --version e0.3.0 --db results/experiments.db

    # Retrieve a specific artifact from a past experiment
    python experiment_explorer.py get --version e0.3.0 --artifact models/production/model.bin --output /tmp/model.bin

    # Create a disposable worktree to explore a past experiment
    python experiment_explorer.py checkout --version e0.3.0

    # List active exploration worktrees
    python experiment_explorer.py worktrees

    # Remove all exploration worktrees
    python experiment_explorer.py cleanup

    # Compare two experiment versions side by side
    python experiment_explorer.py diff --v1 e0.2.0 --v2 e0.3.0 --db results/experiments.db
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import duckdb
except ImportError:
    duckdb = None

EXPLORE_PREFIX = "explore-"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_repo_root() -> str:
    """Get the git repository root directory."""
    return subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel']
    ).decode().strip()


def _run(cmd: list[str], check: bool = True, capture: bool = True,
         cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, cwd=cwd)


def _require_duckdb() -> None:
    """Exit with error if duckdb is not installed."""
    if duckdb is None:
        print("ERROR: duckdb required for database queries. Install with: pip install duckdb")
        sys.exit(1)


def _connect_db(db_path: str, table: str = 'predictions') -> tuple[object, list[str]]:
    """Connect to a SQLite database via DuckDB and return (connection, columns)."""
    _require_duckdb()
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS db (TYPE SQLITE)")
    con.execute(f"SELECT * FROM db.{table} LIMIT 0")
    cols = [r[0] for r in con.description]
    return con, cols


def _format_table(headers: list[str], rows: list[tuple], max_col_width: int = 30) -> str:
    """Format rows as an aligned text table."""
    if not rows:
        return "(no results)"

    str_rows = []
    for row in rows:
        str_row = []
        for v in row:
            if isinstance(v, float):
                str_row.append(f"{v:.4f}")
            elif v is None:
                str_row.append("-")
            else:
                s = str(v)
                if len(s) > max_col_width:
                    s = s[:max_col_width - 2] + ".."
                str_row.append(s)
        str_rows.append(str_row)

    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers)]
    lines.append("-+-".join("-" * w for w in widths))
    for row in str_rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List experiments with key metrics."""
    con, cols = _connect_db(args.db, args.table)

    # Pick interesting columns
    priority = ['version', 'track_id', 'model', 'timestamp']
    metric_cols = [c for c in cols if any(m in c.lower() for m in
                   ['f1', 'accuracy', 'precision', 'recall', 'loss', 'score', 'latency'])]
    select = [c for c in priority if c in cols] + metric_cols

    if not select:
        select = cols[:8]

    order = 'timestamp DESC'
    if args.sort and args.sort in cols:
        order = f"{args.sort} DESC"

    query = f"SELECT {', '.join(select)} FROM db.{args.table} ORDER BY {order}"
    if args.limit:
        query += f" LIMIT {args.limit}"

    rows = con.execute(query).fetchall()
    print(_format_table(select, rows))
    print(f"\n({len(rows)} experiments)")

    # Show git tags too
    print("\nGit experiment tags:")
    result = _run(['git', 'tag', '-l', 'e*', '-n1'], check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("  (no e* tags found)")

    con.close()


def cmd_show(args: argparse.Namespace) -> None:
    """Show full details of a specific experiment version."""
    con, cols = _connect_db(args.db, args.table)

    if 'version' not in cols:
        print("ERROR: No 'version' column in database. Cannot filter by version.")
        con.close()
        sys.exit(1)

    rows = con.execute(f"""
        SELECT * FROM db.{args.table}
        WHERE version = '{args.version}'
        ORDER BY timestamp DESC
    """).fetchall()

    if not rows:
        print(f"No experiments found for version '{args.version}'")
        con.close()
        return

    print(f"=== Experiment {args.version} ({len(rows)} runs) ===\n")

    for i, row in enumerate(rows):
        data = dict(zip(cols, row))
        if len(rows) > 1:
            print(f"--- Run {i + 1} ---")
        for k, v in data.items():
            if v is not None:
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        print()

    # Show git tag info if exists
    result = _run(['git', 'tag', '-l', args.version, '-n9'], check=False)
    if result.stdout.strip():
        print(f"Git tag:\n  {result.stdout.strip()}")

    # Show what DVC files were tracked at that tag
    result = _run(['git', 'show', f'{args.version}:dvc.lock'], check=False)
    if result.returncode == 0 and result.stdout:
        print(f"\nDVC lock file exists at this tag (data is versioned)")
    else:
        result = _run(['git', 'ls-tree', '--name-only', '-r', args.version], check=False)
        if result.returncode == 0:
            dvc_files = [f for f in result.stdout.splitlines() if f.endswith('.dvc')]
            if dvc_files:
                print(f"\nDVC-tracked artifacts at this version:")
                for f in dvc_files:
                    print(f"  {f}")

    con.close()


def cmd_get(args: argparse.Namespace) -> None:
    """Retrieve a specific artifact from a past experiment.

    Two strategies available:
    - worktree: Create temp worktree, dvc checkout, copy artifact (safe, works for everything)
    - dvc-get: Use 'dvc get' for direct download (fast, no checkout, but needs remote access)
    """
    version = args.version
    artifact = args.artifact
    output = args.output or os.path.join(tempfile.gettempdir(), os.path.basename(artifact))
    strategy = args.strategy

    if strategy == "auto":
        # Auto-detect: try dvc get first (faster), fall back to worktree
        strategy = _detect_best_strategy(version, artifact)

    print(f"Retrieving '{artifact}' from {version} -> {output}")
    print(f"Strategy: {strategy}\n")

    if strategy == "dvc-get":
        _get_via_dvc(version, artifact, output)
    else:
        _get_via_worktree(version, artifact, output)


def _detect_best_strategy(version: str, artifact: str) -> str:
    """Detect whether dvc-get or worktree is better for this retrieval."""
    # Check if artifact is DVC-tracked (has .dvc file)
    result = _run(['git', 'show', f'{version}:{artifact}.dvc'], check=False)
    if result.returncode == 0:
        # DVC-tracked file - check if remote is configured
        remote_result = _run(['dvc', 'remote', 'list'], check=False)
        if remote_result.returncode == 0 and remote_result.stdout.strip():
            return "dvc-get"

    # Check if it's a plain git-tracked file
    result = _run(['git', 'show', f'{version}:{artifact}'], check=False)
    if result.returncode == 0:
        return "git-show"

    # Fall back to worktree (safest)
    return "worktree"


def _get_via_dvc(version: str, artifact: str, output: str) -> None:
    """Retrieve artifact using dvc get (fast, no checkout needed).

    Pros: Fast, no working directory changes, works with remote storage.
    Cons: Requires DVC remote to be configured and accessible.
    """
    repo_root = _get_repo_root()
    # Use '.' for current repo — dvc get fetches from the DVC remote
    result = _run(
        ['dvc', 'get', '.', artifact, '--rev', version, '-o', output],
        check=False, cwd=repo_root
    )
    if result.returncode == 0:
        size = os.path.getsize(output) if os.path.exists(output) else 0
        print(f"Retrieved via dvc get: {output} ({size / 1024:.1f} KB)")
    else:
        print(f"dvc get failed: {result.stderr}")
        print("Falling back to worktree strategy...")
        _get_via_worktree(version, artifact, output)


def _get_via_worktree(version: str, artifact: str, output: str) -> None:
    """Retrieve artifact by creating a temporary worktree.

    Pros: Works for any file (git-tracked, DVC-tracked, generated).
          Safe - doesn't touch your working directory.
    Cons: Slower (full checkout + dvc checkout), uses disk space temporarily.
    """
    repo_root = _get_repo_root()
    parent = Path(repo_root).parent
    worktree_name = f"{EXPLORE_PREFIX}get-{version}"
    worktree_path = parent / worktree_name

    try:
        # Create detached worktree at the tag
        print(f"Creating temporary worktree at {worktree_path}...")
        _run(['git', 'worktree', 'add', '--detach', str(worktree_path), version])

        # DVC checkout in the worktree
        artifact_path = worktree_path / artifact
        if not artifact_path.exists():
            print("Running dvc checkout in worktree...")
            _run(['dvc', 'checkout'], check=False, cwd=str(worktree_path))

        # Check for the artifact
        if artifact_path.exists():
            os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
            import shutil
            shutil.copy2(str(artifact_path), output)
            size = os.path.getsize(output)
            print(f"Retrieved via worktree: {output} ({size / 1024:.1f} KB)")
        else:
            # Try dvc pull for this specific file
            dvc_file = str(artifact_path) + '.dvc'
            if Path(dvc_file).exists():
                print(f"Pulling from DVC remote...")
                _run(['dvc', 'pull', dvc_file], check=False, cwd=str(worktree_path))
                if artifact_path.exists():
                    import shutil
                    shutil.copy2(str(artifact_path), output)
                    size = os.path.getsize(output)
                    print(f"Retrieved via dvc pull: {output} ({size / 1024:.1f} KB)")
                else:
                    print(f"ERROR: Artifact not found even after dvc pull: {artifact}")
            else:
                print(f"ERROR: Artifact not found at {artifact_path}")
                # Show what IS available
                _show_available_files(worktree_path)
    finally:
        # Always clean up the temp worktree
        print("Cleaning up temporary worktree...")
        _run(['git', 'worktree', 'remove', '--force', str(worktree_path)], check=False)


def _show_available_files(worktree_path: Path) -> None:
    """Show available files in the worktree to help the user find what they want."""
    print("\nAvailable files at this version:")
    for root, dirs, files in os.walk(worktree_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), worktree_path)
            print(f"  {rel}")


def cmd_checkout(args: argparse.Namespace) -> None:
    """Create a disposable worktree to fully explore a past experiment.

    This creates ../explore-<version>/ where you can browse all code, data,
    and models exactly as they were at that experiment version.
    """
    version = args.version
    repo_root = _get_repo_root()
    parent = Path(repo_root).parent
    worktree_name = f"{EXPLORE_PREFIX}{version}"
    worktree_path = parent / worktree_name

    if worktree_path.exists():
        print(f"Exploration worktree already exists: {worktree_path}")
        print(f"Use it: cd {worktree_path}")
        print(f"Remove it: python experiment_explorer.py cleanup --version {version}")
        return

    print(f"Creating exploration worktree at {worktree_path}...")

    # Create detached worktree (no new branch needed for exploration)
    _run(['git', 'worktree', 'add', '--detach', str(worktree_path), version])

    # DVC checkout to restore data/models
    print("Running dvc checkout to restore artifacts...")
    result = _run(['dvc', 'checkout'], check=False, cwd=str(worktree_path))
    if result.returncode != 0:
        print("Note: dvc checkout had issues. You may need to run 'dvc pull' for remote artifacts.")

    print(f"\nExploration worktree ready at: {worktree_path}")
    print(f"")
    print(f"  cd {worktree_path}")
    print(f"")
    print(f"Everything is at the exact state of experiment {version}.")
    print(f"This is read-only exploration - changes here won't affect your main branch.")
    print(f"")
    print(f"When done: python experiment_explorer.py cleanup --version {version}")
    print(f"  Or: git worktree remove {worktree_path}")


def cmd_worktrees(args: argparse.Namespace) -> None:
    """List all active exploration worktrees."""
    result = _run(['git', 'worktree', 'list', '--porcelain'], check=False)
    if result.returncode != 0:
        print("ERROR: Could not list worktrees")
        return

    explore_trees = []
    current = {}
    for line in result.stdout.splitlines():
        if line.startswith('worktree '):
            current = {'path': line.split(' ', 1)[1]}
        elif line.startswith('HEAD '):
            current['head'] = line.split(' ', 1)[1][:8]
        elif line.startswith('branch '):
            current['branch'] = line.split(' ', 1)[1]
        elif line == 'detached':
            current['branch'] = '(detached)'
        elif line == '' and current:
            if EXPLORE_PREFIX in current.get('path', ''):
                explore_trees.append(current)
            current = {}

    # Check last entry
    if current and EXPLORE_PREFIX in current.get('path', ''):
        explore_trees.append(current)

    if not explore_trees:
        print("No active exploration worktrees.")
        return

    print(f"Active exploration worktrees ({len(explore_trees)}):\n")
    for tree in explore_trees:
        path = tree['path']
        head = tree.get('head', '?')
        # Extract version from path
        name = Path(path).name
        version = name.replace(EXPLORE_PREFIX, '')
        print(f"  {version:15}  {head}  {path}")

    print(f"\nCleanup all: python experiment_explorer.py cleanup")


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove exploration worktrees."""
    if args.version:
        # Remove a specific exploration worktree
        repo_root = _get_repo_root()
        parent = Path(repo_root).parent
        worktree_path = parent / f"{EXPLORE_PREFIX}{args.version}"

        if not worktree_path.exists():
            print(f"No exploration worktree found for {args.version}")
            return

        _run(['git', 'worktree', 'remove', '--force', str(worktree_path)], check=False)
        print(f"Removed: {worktree_path}")
        return

    # Remove all exploration worktrees
    result = _run(['git', 'worktree', 'list', '--porcelain'], check=False)
    if result.returncode != 0:
        return

    removed = 0
    current_path = None
    for line in result.stdout.splitlines():
        if line.startswith('worktree '):
            current_path = line.split(' ', 1)[1]
        elif line == '' and current_path and EXPLORE_PREFIX in current_path:
            _run(['git', 'worktree', 'remove', '--force', current_path], check=False)
            print(f"Removed: {current_path}")
            removed += 1
            current_path = None

    # Check last entry
    if current_path and EXPLORE_PREFIX in current_path:
        _run(['git', 'worktree', 'remove', '--force', current_path], check=False)
        print(f"Removed: {current_path}")
        removed += 1

    if removed == 0:
        print("No exploration worktrees to clean up.")
    else:
        print(f"\nRemoved {removed} exploration worktree(s).")


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two experiment versions side by side."""
    con, cols = _connect_db(args.db, args.table)

    if 'version' not in cols:
        print("ERROR: No 'version' column in database.")
        con.close()
        sys.exit(1)

    v1_rows = con.execute(f"""
        SELECT * FROM db.{args.table} WHERE version = '{args.v1}'
        ORDER BY timestamp DESC LIMIT 1
    """).fetchall()

    v2_rows = con.execute(f"""
        SELECT * FROM db.{args.table} WHERE version = '{args.v2}'
        ORDER BY timestamp DESC LIMIT 1
    """).fetchall()

    if not v1_rows:
        print(f"No data for version {args.v1}")
    if not v2_rows:
        print(f"No data for version {args.v2}")
    if not v1_rows or not v2_rows:
        con.close()
        return

    v1_data = dict(zip(cols, v1_rows[0]))
    v2_data = dict(zip(cols, v2_rows[0]))

    # Skip internal columns for cleaner output
    skip = {'timestamp', 'track_id', '_id'}

    print(f"{'Column':<25} {'[' + args.v1 + ']':<25} {'[' + args.v2 + ']':<25} Delta")
    print("-" * 90)

    for col in cols:
        if col in skip:
            continue
        v1_val = v1_data.get(col)
        v2_val = v2_data.get(col)

        if v1_val == v2_val and v1_val is not None:
            continue  # Skip identical values

        v1_str = f"{v1_val:.4f}" if isinstance(v1_val, float) else str(v1_val or "-")
        v2_str = f"{v2_val:.4f}" if isinstance(v2_val, float) else str(v2_val or "-")

        delta = ""
        if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
            diff = v2_val - v1_val
            pct = (diff / v1_val * 100) if v1_val != 0 else 0
            sign = "+" if diff > 0 else ""
            delta = f"{sign}{diff:.4f} ({sign}{pct:.1f}%)"

        print(f"{col:<25} {v1_str:<25} {v2_str:<25} {delta}")

    # Show git diff between versions if tags exist
    print(f"\nCode changes between {args.v1} and {args.v2}:")
    result = _run(['git', 'diff', '--stat', args.v1, args.v2], check=False)
    if result.returncode == 0 and result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("  (could not compute git diff - tags may not exist)")

    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for experiment exploration."""
    parser = argparse.ArgumentParser(
        description='Read-only exploration and retrieval of past ML experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Retrieval Strategies:
  worktree   Create temp worktree, dvc checkout, copy artifact.
             + Safe, works for any file type (git, DVC, generated)
             + Doesn't touch your working directory
             - Slower (full checkout), uses temporary disk space

  dvc-get    Use 'dvc get' for direct download from DVC remote.
             + Fast, no checkout needed, minimal disk usage
             - Requires DVC remote configured and accessible
             - Only works for DVC-tracked files

  auto       Try dvc-get first, fall back to worktree (default)

Examples:
  # Browse experiments
  %(prog)s list --db results/experiments.db --sort f1_score
  %(prog)s show --version e0.3.0 --db results/experiments.db

  # Get a specific model
  %(prog)s get --version e0.3.0 --artifact models/production/model.bin

  # Explore full experiment state
  %(prog)s checkout --version e0.3.0
  cd ../explore-e0.3.0
  # ... browse code, data, models ...
  %(prog)s cleanup --version e0.3.0

  # Compare experiments
  %(prog)s diff --v1 e0.2.0 --v2 e0.3.0 --db results/experiments.db
"""
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # list
    p = subparsers.add_parser('list', help='List experiments with key metrics')
    p.add_argument('--db', required=True, help='xetrack SQLite database path')
    p.add_argument('--table', default='predictions', help='Table name (default: predictions)')
    p.add_argument('--sort', help='Column to sort by (descending)')
    p.add_argument('--limit', type=int, default=20, help='Max results (default: 20)')

    # show
    p = subparsers.add_parser('show', help='Show full details of an experiment')
    p.add_argument('--version', '-v', required=True, help='Experiment version (e.g., e0.3.0)')
    p.add_argument('--db', required=True, help='xetrack SQLite database path')
    p.add_argument('--table', default='predictions')

    # get
    p = subparsers.add_parser('get', help='Retrieve a specific artifact from a past experiment')
    p.add_argument('--version', '-v', required=True, help='Experiment version or git tag')
    p.add_argument('--artifact', '-a', required=True, help='Path to artifact (e.g., models/production/model.bin)')
    p.add_argument('--output', '-o', help='Output path (default: /tmp/<filename>)')
    p.add_argument('--strategy', choices=['auto', 'worktree', 'dvc-get'], default='auto',
                   help='Retrieval strategy (default: auto)')

    # checkout
    p = subparsers.add_parser('checkout', help='Create a disposable worktree for exploration')
    p.add_argument('--version', '-v', required=True, help='Experiment version or git tag')

    # worktrees
    subparsers.add_parser('worktrees', help='List active exploration worktrees')

    # cleanup
    p = subparsers.add_parser('cleanup', help='Remove exploration worktrees')
    p.add_argument('--version', '-v', help='Specific version to remove (default: all)')

    # diff
    p = subparsers.add_parser('diff', help='Compare two experiment versions')
    p.add_argument('--v1', required=True, help='First version')
    p.add_argument('--v2', required=True, help='Second version')
    p.add_argument('--db', required=True, help='xetrack SQLite database path')
    p.add_argument('--table', default='predictions')

    args = parser.parse_args()

    commands = {
        'list': cmd_list,
        'show': cmd_show,
        'get': cmd_get,
        'checkout': cmd_checkout,
        'worktrees': cmd_worktrees,
        'cleanup': cmd_cleanup,
        'diff': cmd_diff,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
