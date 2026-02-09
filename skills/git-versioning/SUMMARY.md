# Git + DVC Versioning Skill - Summary

ML experiment versioning using Git, DVC, xetrack, and DuckDB.

## What This Skill Does

Guides users through reproducible ML experiment versioning with three workflow modes:

| Scenario | Workflow | DB Engine | Branching |
|---|---|---|---|
| Single experiment | Sequential | SQLite | Main branch |
| Param sweep, same code/data | Parallel | DuckDB | Main branch |
| Different code or data per exp | Worktree | SQLite | Branch per exp |

## Key Capabilities

- **Workflow selection**: Decision tree to pick the right experiment mode
- **Git worktree isolation**: Parallel experiments with different code/data via `git worktree`
- **Merge/rebase semantics**: Per-artifact decisions (results DB, data, models) using DuckDB as merge engine
- **Version tagging**: `e{major}.{minor}.{patch}` convention with descriptive tag messages
- **Model management**: Candidates pattern (production vs candidates), promotion, pruning
- **Experiment exploration**: Browse, compare, and retrieve artifacts from past experiments
- **DVC integration**: Shared cache, remote storage, pipeline stages for merge/rebase

## Scripts

| Script | Purpose |
|---|---|
| `setup_worktree.sh` | Create worktree with shared DVC cache |
| `experiment_explorer.py` | Browse experiments, retrieve artifacts (read-only) |
| `merge_artifacts.py` | DuckDB merge/rebase for SQLite + parquet |
| `version_tag.py` | Create annotated tags with metrics |
| `model_manager.py` | Promote/prune models, manage candidates |

## When to Use This vs Benchmark Skill

- **Benchmark skill**: Designing and running individual benchmarks, tracking predictions/metrics with xetrack, caching, schema validation
- **This skill**: Versioning experiments over time, parallel experimentation, merging results across branches, retrieving historical artifacts

## References

- `references/workflows.md` - Step-by-step for all 3 workflows
- `references/merge-rebase.md` - DuckDB merge/rebase semantics
- `references/dvc-setup.md` - DVC cache, remotes, S3 configuration
- `references/use-cases.md` - 14 concrete real-world scenarios
- `references/data_versioning.md` - Core versioning concepts (hash chain, metadata, repo structure)
- `references/quickstart.md` - End-to-end quickstart bridging benchmark + versioning skills
