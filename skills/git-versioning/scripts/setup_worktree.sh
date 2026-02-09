#!/bin/bash
# setup_worktree.sh - Create a git worktree with shared DVC cache for ML experiments.
#
# Usage:
#   ./setup_worktree.sh <experiment-name> [shared-cache-path]
#
# Examples:
#   ./setup_worktree.sh transformer
#   ./setup_worktree.sh cnn-v2 /data/shared-dvc-cache
#   ./setup_worktree.sh augmented-data ~/.dvc/shared-cache
#
# Creates:
#   ../exp-<name>/  with branch exp/<name> and shared DVC cache configured

set -euo pipefail

NAME="${1:?Usage: $0 <experiment-name> [shared-cache-path]}"
SHARED_CACHE="${2:-}"

# Validate we're in a git repo (must run before git rev-parse --show-toplevel)
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: Not a git repository"
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
PARENT="$(dirname "$REPO_ROOT")"
WORKTREE_PATH="$PARENT/exp-$NAME"
BRANCH_NAME="exp/$NAME"

# Check if worktree already exists
if [ -d "$WORKTREE_PATH" ]; then
    echo "ERROR: Worktree already exists at $WORKTREE_PATH"
    echo "Use it:    cd $WORKTREE_PATH"
    echo "Remove it: git worktree remove $WORKTREE_PATH"
    exit 1
fi

# Check if branch already exists
if git rev-parse --verify "$BRANCH_NAME" > /dev/null 2>&1; then
    echo "ERROR: Branch $BRANCH_NAME already exists"
    echo "Delete it: git branch -D $BRANCH_NAME"
    exit 1
fi

# Detect shared cache path
if [ -z "$SHARED_CACHE" ]; then
    # Try to detect existing shared cache from current DVC config
    EXISTING_CACHE="$(dvc config cache.dir 2>/dev/null || true)"
    if [ -n "$EXISTING_CACHE" ] && [ "$EXISTING_CACHE" != ".dvc/cache" ]; then
        SHARED_CACHE="$EXISTING_CACHE"
        echo "Detected existing shared cache: $SHARED_CACHE"
    else
        # Default to ~/.dvc/shared-cache
        SHARED_CACHE="$HOME/.dvc/shared-cache"
        echo "No shared cache detected. Using default: $SHARED_CACHE"
    fi
fi

# Create shared cache directory if it doesn't exist
mkdir -p "$SHARED_CACHE"

echo ""
echo "Creating experiment worktree..."
echo "  Name:     $NAME"
echo "  Branch:   $BRANCH_NAME"
echo "  Path:     $WORKTREE_PATH"
echo "  DVC cache: $SHARED_CACHE"
echo ""

# Create worktree with new branch
git worktree add "$WORKTREE_PATH" -b "$BRANCH_NAME"

# Configure DVC shared cache in the new worktree
cd "$WORKTREE_PATH"
dvc cache dir "$SHARED_CACHE"
dvc config cache.shared group 2>/dev/null || true
dvc config cache.type symlink,hardlink 2>/dev/null || true

# Pull DVC files to populate from shared cache
dvc checkout 2>/dev/null || echo "Note: dvc checkout had issues (may need 'dvc pull' for remote artifacts)"

echo ""
echo "Worktree ready!"
echo ""
echo "  cd $WORKTREE_PATH"
echo ""
echo "When done:"
echo "  1. dvc add <artifacts>"
echo "  2. git add -A && git commit -m 'exp: $NAME results'"
echo "  3. git tag -a exp-$NAME-v1 -m 'description'"
echo "  4. dvc push   # CRITICAL: before cleanup!"
echo "  5. cd $REPO_ROOT"
echo "  6. python scripts/merge_artifacts.py --strategy merge --base-db results/experiments.db --exp-db $WORKTREE_PATH/results/experiments.db"
echo "  7. git worktree remove $WORKTREE_PATH"
echo "  8. git branch -D $BRANCH_NAME  # tag preserves everything"
