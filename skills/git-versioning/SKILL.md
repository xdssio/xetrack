---
name: git-dvc-versioning
description: >
  Manage ML experiment versioning using Git + DVC + xetrack + DuckDB.
  Use this skill when the user wants to: version experiments with git tags and DVC,
  run sequential or parallel ML experiments with reproducibility,
  merge or rebase experiment results (SQLite databases, parquet data, models),
  set up DVC remotes and cache (including S3), manage model candidates and promotion,
  use git worktrees for parallel code/data experiments, track experiment metadata
  in SQLite with xetrack, retrieve models or data from past experiments,
  explore or compare historical experiments, or find what data was used.
  Triggers on: "version experiment", "tag experiment",
  "DVC setup", "merge results", "rebase data", "git worktree experiment",
  "parallel experiments", "model promotion", "candidates", "experiment tracking",
  "reproducibility", "dvc pipeline", "merge database", "data versioning",
  "retrieve model", "get model", "find experiment", "explore experiment",
  "compare experiments", "what data was used", "inspect results",
  "restore experiment", "recover model", "browse experiments",
  "download model", "historical model".
---

# Git + DVC Versioning Skill

Guide users through reproducible ML experiment versioning using Git, DVC, xetrack (SQLite/DuckDB tracker), and DuckDB as the merge engine.

## Before You Start

Read the reference files based on what the user needs:

```
Read references/workflows.md        # Sequential, parallel, worktree experiment flows
Read references/merge-rebase.md     # When and how to merge vs rebase artifacts
Read references/dvc-setup.md        # DVC cache, remotes, S3, worktree config
Read references/use-cases.md        # Concrete examples for every scenario
Read references/data_versioning.md  # Full versioning specification
Read references/quickstart.md       # End-to-end quickstart bridging benchmark + versioning
```

## Core Responsibility

You are the decision-making layer. Your job is to:

1. **Help the user choose the right workflow** (sequential, parallel same-branch, parallel worktree)
2. **Help decide merge vs rebase** for each artifact type
3. **Suggest meaningful tag descriptions** based on experiment context
4. **Execute the workflow** using the scripts and templates provided
5. **Keep the repo clean** with the candidates pattern for models

## Workflow Selection

Ask the user these questions to determine the right workflow:

```
Q1: Is this a single experiment or multiple parallel experiments?
    -> Single: Sequential workflow
    -> Multiple: Continue to Q2

Q2: Are code and data identical across experiments (only params differ)?
    -> Yes: Parallel on same branch (DuckDB engine, thread-safe)
    -> No: Continue to Q3

Q3: Do experiments need different code or different data?
    -> Yes: Worktree workflow (git worktree + branch per experiment)
```

### Quick Reference

| Scenario                          | Workflow     | DB Engine | Branching     |
|-----------------------------------|-------------|-----------|---------------|
| Single experiment                 | Sequential  | SQLite    | Main branch   |
| Param sweep, same code/data      | Parallel    | DuckDB    | Main branch   |
| Different code or data per exp    | Worktree    | SQLite    | Branch per exp|

## Version Naming Convention

```
e{major}.{minor}.{patch}

Major: fundamental change (new architecture, new dataset)
Minor: meaningful variation (hyperparams, features)
Patch: reruns, fixes, minor tweaks
```

Examples: `e0.1.0`, `e1.0.0`, `e0.1.3`

## Tag Description Format

Always suggest tag descriptions. Format:

```
git tag -a e0.1.0 -m "model=bert-base | lr=1e-4 | F1=0.89 | data=3a2f1b | description"
```

Pattern: `key=value | key=value | ... | human-readable description`

Include at minimum:
- Model name/type
- Key hyperparameter(s)
- Primary metric + value
- Short data hash (first 6 chars of DVC hash)
- One-line human description

Suggest the description proactively after each experiment completes. Pull values from the xetrack database and git/DVC state.

## Merge vs Rebase Decision

After experiments complete, help the user decide per artifact type:

### Results Database (SQLite)
- **Merge** (default): append all experiment rows. Use after parameter sweeps, when you want full history.
- **Rebase**: when evaluation criteria changed on main and old results are no longer comparable.

### Data Files (Parquet)
- **Merge**: adding new samples, new annotation batches, additive features
- **Rebase**: schema changed, preprocessing overhaul, feature transformation that should apply to all data

### Models
- **Merge**: store experiment model as candidate alongside production model. For ensembles, A/B tests, multi-candidate evaluation.
- **Rebase**: clear winner. Promote experiment model to production.

Use `scripts/merge_artifacts.py` to execute. The merge strategy is parameterized in `params.yaml` and tracked by DVC.

## Candidates Pattern for Models

Keep the main branch clean:

- `models/production/model.bin` - current best, DVC tracked individually
- `models/candidates/` - all non-production models, tracked with a single `candidates.dvc`
- After parallel sweep: promote best to production, keep top-K as candidates, prune rest
- Historical models always accessible via their experiment tags + `dvc checkout`

Use `scripts/model_manager.py` to upload/download models by version using the SQLite DB as index.

> **Note:** xetrack can store Python objects as assets in SQLite automatically — any non-primitive value (e.g., a fitted model) passed to `tracker.log()` or `tracker.track()` is serialized and stored via the assets system (requires `pip install xetrack[assets]`). However, this is **not recommended** for model management unless you're working with small models and few of them. For anything beyond that, use DVC for model storage and xetrack only for metadata/metrics.

## Sequential Workflow Steps

```
1. Define version: e0.X.Y
2. Make code/data/param changes
3. Run experiment -> xetrack logs to SQLite
4. Store git_commit + dvc_hash in metadata
5. dvc add (data, models, db)
6. git add + commit
7. git tag -a eX.Y.Z -m "<suggested description>"
8. dvc push + git push --tags
```

## Parallel Workflow Steps (Same Branch)

```
1. Ensure code + data committed and clean
2. Define param grid in params.yaml
3. Run experiments in parallel with DuckDB engine
4. Each gets unique run_id via xetrack
5. Evaluate results, pick winner
6. Promote best model to production
7. Save candidates with candidates.dvc
8. Commit, tag with best result metrics
```

## Worktree Workflow Steps

Use `scripts/setup_worktree.sh` to automate steps 1-3 (prevents the #1 pitfall: forgetting DVC cache setup):

```bash
# Automated setup (recommended)
./scripts/setup_worktree.sh transformer
# Creates ../exp-transformer with branch exp/transformer and shared DVC cache

# Or manual setup
git worktree add ../exp-NAME -b exp/NAME
cd ../exp-NAME
dvc cache dir /shared/dvc-cache  # CRITICAL: must configure shared cache
```

```
1. ./scripts/setup_worktree.sh NAME  (or manual: git worktree add + dvc cache dir)
2. cd ../exp-NAME
3. Make changes, run experiment
4. dvc add, git commit, git tag
5. dvc push
6. cd ../main-project
7. Merge results DB from worktree to main (DuckDB)
8. Decide: merge or rebase data and models
9. Commit + tag on main
10. git worktree remove ../exp-NAME
11. git branch -D exp/NAME  (tag preserves everything)
```

## DVC Pipeline Integration

Merge and rebase operations should be DVC pipeline stages. Use `assets/dvc_yaml_template.yaml` as a starting point. The merge strategy is a parameter in `params.yaml`, making it tracked, reproducible, and switchable.

## Safety Checklist

Before removing any worktree or branch:
- [ ] `dvc push` completed successfully
- [ ] Tag created with descriptive message
- [ ] Results merged to main database
- [ ] Best model promoted (if applicable)

Before any rebase operation:
- [ ] User confirmed this is destructive
- [ ] Previous state accessible via tags
- [ ] Backup tag exists if uncertain

## Common Pitfalls

1. **Forgetting `dvc push` before worktree removal** - artifacts lost forever
2. **Not sharing DVC cache across worktrees** - duplicates all data
3. **Using SQLite for parallel writes** - use DuckDB engine instead
4. **Not using `persist: true` in DVC pipeline for databases** - DVC deletes output before re-running
5. **Rebasing without confirmation** - destructive operation, always confirm with user
6. **DuckDB + multiprocessing file locks** - use separate DB files per process, merge after

## Exploring Past Experiments

Use `scripts/experiment_explorer.py` for read-only browsing and retrieval. Use `scripts/model_manager.py` for write operations (promote, prune).

### Browsing

```bash
# List experiments ranked by metric
python scripts/experiment_explorer.py list --db results/experiments.db --sort f1_score

# Full details of one experiment (params, metrics, DVC artifacts)
python scripts/experiment_explorer.py show --version e0.3.0 --db results/experiments.db

# Compare two experiments side by side (metric deltas + code diff)
python scripts/experiment_explorer.py diff --v1 e0.2.0 --v2 e0.3.0 --db results/experiments.db
```

### Retrieving Artifacts

Two strategies for getting a specific file from a past experiment:

| Strategy | How it works | Pros | Cons |
|----------|-------------|------|------|
| **worktree** | Create temp worktree at tag, dvc checkout, copy file | Safe, works for any file (git, DVC, generated). Doesn't touch working directory | Slower, uses temporary disk space |
| **dvc-get** | Direct download from DVC remote via `dvc get --rev` | Fast, minimal disk usage, no checkout needed | Requires DVC remote configured and accessible. DVC-tracked files only |
| **auto** (default) | Try dvc-get first, fall back to worktree | Best of both | May try DVC first unnecessarily |

**Recommend based on use case:**
- "Just need one model file" → `dvc-get` (fast)
- "Need a generated file or non-DVC file" → `worktree`
- "Not sure" → `auto`

```bash
# Get a model from a past experiment
python scripts/experiment_explorer.py get \
    --version e0.3.0 \
    --artifact models/production/model.bin \
    --output /tmp/model_v3.bin

# Force worktree strategy (safest)
python scripts/experiment_explorer.py get \
    --version e0.3.0 \
    --artifact models/production/model.bin \
    --strategy worktree

# Get training data from a past experiment
python scripts/experiment_explorer.py get \
    --version e0.2.0 \
    --artifact data/train.parquet \
    --output /tmp/train_v2.parquet
```

### Full Experiment Exploration

When you need to browse the complete state of a past experiment (code, data, models, configs):

```bash
# Create a disposable exploration worktree
python scripts/experiment_explorer.py checkout --version e0.3.0
cd ../explore-e0.3.0

# Everything is exactly as it was at e0.3.0
# Browse code, load models, inspect data, run evaluation...

# When done
python scripts/experiment_explorer.py cleanup --version e0.3.0
```

Exploration worktrees are named `../explore-*` (distinct from experiment worktrees `../exp-*`). They are detached HEAD checkouts - safe, read-only, no branch to clean up.

```bash
# List active exploration worktrees
python scripts/experiment_explorer.py worktrees

# Remove all exploration worktrees
python scripts/experiment_explorer.py cleanup
```

## Troubleshooting

- **`merge_artifacts.py` fails mid-merge**: Output DB may be partially written. Re-run from the original base DB (the `--base-db` file is never modified in-place unless it equals `--output-db`). Keep backups of base DB before merges.
- **Worktree in bad state (detached HEAD, uncommitted changes)**: `cd` back to main, run `git worktree remove --force ../exp-NAME`. If that fails, delete the directory manually then `git worktree prune`.
- **`dvc push` partially succeeds**: Re-run `dvc push` - it's idempotent. Check `dvc status --cloud` to see what's still missing on remote.
- **"database is locked" during parallel writes**: You're using SQLite for concurrent writes. Switch to `engine='duckdb'`, or use separate DB files per process and merge after.
- **DVC checkout fails in worktree**: Shared cache likely not configured. Run `dvc cache dir` to check, then `dvc cache dir /shared/path` to fix. May need `dvc pull` if cache is empty.
- **Tag already exists**: You're re-using a version string. Run `git tag -l 'e*'` to see existing tags, then pick the next version or use `python scripts/version_tag.py --suggest`.
- **Schema mismatch during merge**: `UNION ALL BY NAME` handles this by filling missing columns with NULLs. Check the output with `duckdb -c "SELECT * FROM read_parquet('output.parquet') LIMIT 5"`.
- **Model not found after `promote`**: The candidate file must exist at `models/candidates/<track_id>_model.bin`. Check with `ls models/candidates/`. If model was trained in a worktree, copy it before worktree removal.
- **Exploration worktree won't delete**: Run `git worktree list` to check status. Use `git worktree remove --force` or `python scripts/experiment_explorer.py cleanup`.
- **Lost artifacts after worktree removal**: If `dvc push` wasn't run, artifacts are gone from the worktree. Check if the shared DVC cache still has them: `dvc cache dir` then look in that directory. If using a remote, `dvc push` from another branch that has the `.dvc` files.

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/setup_worktree.sh` | Create worktree with shared DVC cache | Before starting any worktree experiment (prevents #1 pitfall) |
| `scripts/experiment_explorer.py` | Browse experiments, retrieve artifacts, create exploration worktrees | When you need to look at past experiments (read-only) |
| `scripts/merge_artifacts.py` | DuckDB merge/rebase for SQLite + parquet | After any experiment that needs results combined |
| `scripts/version_tag.py` | Create annotated tags with metric descriptions | After every experiment commit |
| `scripts/model_manager.py` | Promote/prune models, manage candidates | When changing production model or managing candidates (write operations) |
