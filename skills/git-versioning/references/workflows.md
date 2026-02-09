# Experiment Workflows

Detailed step-by-step workflows for each experiment mode.

## Prerequisites (All Workflows)

```bash
pip install xetrack xetrack[duckdb] xetrack[assets] duckdb
pip install dvc dvc-s3  # or dvc-gs, dvc-azure as needed
```

Ensure DVC is initialized in the repo:
```bash
dvc init
dvc remote add -d storage s3://your-bucket/dvc-store
```

---

## Workflow 1: Sequential Experiments

Single experiments running one at a time. Code, data, or params may change between runs.

### Step-by-Step

```bash
# 1. Define the version
VERSION="e0.1.0"

# 2. Make your changes (code, data, params)
# ... edit files ...

# 3. Run the experiment
python train.py --version $VERSION
```

```python
# Inside train.py (or equivalent)
from xetrack import Tracker
import subprocess

def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

tracker = Tracker(
    'results/experiments.db',
    params={
        'version': 'e0.1.0',
        'git_commit': get_git_hash(),
        'model': 'bert-base',
        'learning_rate': 1e-4,
    }
)

# Train and log results
result = tracker.track(train_and_evaluate, args=[config])

# Or log manually
tracker.log({
    'f1_score': 0.89,
    'precision': 0.91,
    'recall': 0.87,
    'epoch': 10,
    'train_loss': 0.05
})
```

```bash
# 4. Save artifacts with DVC
dvc add data/train.parquet data/test.parquet
dvc add results/experiments.db
dvc add models/production/model.bin

# 5. Git commit everything
git add -A
git commit -m "experiment: $VERSION - bert-base PII detector"

# 6. Tag with description (agent should suggest this)
git tag -a $VERSION -m "model=bert-base | lr=1e-4 | F1=0.89 | prec=0.91 | PII detector baseline"

# 7. Push
dvc push
git push origin main --tags
```

### Retrieving a Past Sequential Experiment

```bash
git checkout e0.1.0
dvc checkout
# Everything restored: code, data, models, database
python -c "
from xetrack import Reader
df = Reader('results/experiments.db').to_df()
print(df[df.version == 'e0.1.0'])
"
```

---

## Workflow 2: Parallel Experiments (Parameter-Only)

Same code and data, different parameters. Uses DuckDB engine for thread-safe writes.

### Step-by-Step

```python
# parallel_sweep.py
from xetrack import Tracker
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    learning_rate: float
    batch_size: int
    dropout: float
    model_name: str = "bert-base"

def run_single_experiment(config: ExperimentConfig):
    """Run one experiment with given params."""
    tracker = Tracker(
        'results/experiments.db',
        engine='duckdb',  # Thread-safe!
        params={
            'version': 'e0.2.0',
            'model': config.model_name,
        }
    )
    
    # track() auto-unpacks frozen dataclass fields
    result = tracker.track(train_and_evaluate, args=[config])
    return result

# Define parameter grid
configs = [
    ExperimentConfig(learning_rate=1e-3, batch_size=16, dropout=0.1),
    ExperimentConfig(learning_rate=1e-4, batch_size=32, dropout=0.1),
    ExperimentConfig(learning_rate=1e-4, batch_size=32, dropout=0.2),
    ExperimentConfig(learning_rate=1e-5, batch_size=64, dropout=0.3),
]

# Run in parallel
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(run_single_experiment, c) for c in configs]
    results = [f.result() for f in futures]

# Analyze
from xetrack import Reader
df = Reader('results/experiments.db').to_df()
best = df.sort_values('f1_score', ascending=False).iloc[0]
print(f"Best: F1={best.f1_score:.4f} with lr={best.learning_rate}")
```

```bash
# After sweep completes
# Promote best model
python scripts/model_manager.py promote --db results/experiments.db --version e0.2.0 --run-id <best_run_id>

# Save all candidates
cd models/candidates && dvc add . && cd ../..

# Commit and tag
dvc add results/experiments.db models/production/model.bin
git add -A
git commit -m "sweep: e0.2.0 - 4 configs, best F1=0.93"
git tag -a e0.2.0 -m "model=bert-base | lr=1e-4 | bs=32 | drop=0.1 | F1=0.93 | param sweep winner"
dvc push
git push --tags
```

### Important Notes for Parallel

- Use `engine='duckdb'` in xetrack for thread safety
- Each thread gets its own `track_id` automatically
- All results land in the same database, same table
- For multi-process (not multi-thread), use separate DB files and merge after:

```python
# Multi-process alternative
def run_in_process(config, db_path):
    tracker = Tracker(db_path, engine='duckdb', params={...})
    tracker.track(train_and_evaluate, args=[config])

# Each process writes to its own DB
for i, config in enumerate(configs):
    process_db = f'results/experiments_p{i}.db'
    Process(target=run_in_process, args=(config, process_db)).start()

# After all complete, merge with DuckDB
python scripts/merge_artifacts.py \
    --strategy merge \
    --base-db results/experiments.db \
    --exp-db results/experiments_p0.db results/experiments_p1.db ...
```

---

## Workflow 3: Worktree Experiments (Code/Data Changes)

Each experiment needs different code or data. Full isolation via git worktree.

### Step-by-Step

```bash
# 1-2. Create worktree with shared DVC cache (recommended: use the setup script)
./scripts/setup_worktree.sh transformer-pii
# This creates ../exp-transformer-pii with branch exp/transformer-pii
# and auto-configures shared DVC cache

# Or manually:
# git worktree add ../exp-transformer-pii -b exp/transformer-pii
# cd ../exp-transformer-pii
# dvc cache dir /path/to/shared/dvc-cache  # CRITICAL

cd ../exp-transformer-pii

# 3. Make code/data changes
# ... modify architecture, add new data, etc ...

# 4. Run experiment (use SQLite, single writer per worktree)
python train.py --version exp-transformer

# 5. Save everything
dvc add data/ results/experiments.db models/production/model.bin
git add -A
git commit -m "exp: transformer PII detector with attention pooling"

# 6. Tag the experiment
git tag -a exp-transformer-v1 -m "model=transformer | layers=6 | F1=0.94 | 2x slower | attention pooling"

# 7. Push artifacts BEFORE anything else
dvc push

# --- Repeat for another worktree experiment ---
git worktree add ../exp-cnn -b exp/cnn-pii
cd ../exp-cnn
dvc cache dir /path/to/shared/dvc-cache
# ... experiment ...
git tag -a exp-cnn-v1 -m "model=cnn | F1=0.88 | 10x faster | lightweight"
dvc push
```

### Merging Back to Main

```bash
cd /path/to/main-project

# Merge results from both experiments into main DB
python scripts/merge_artifacts.py \
    --strategy merge \
    --base-db results/experiments.db \
    --exp-db ../exp-transformer/results/experiments.db \
           ../exp-cnn/results/experiments.db \
    --output-db results/experiments.db

# Evaluate: transformer won
# Rebase model (promote to production)
cp ../exp-transformer/models/production/model.bin models/production/model.bin

# Merge data if experiment added new samples
python scripts/merge_artifacts.py \
    --strategy merge \
    --base-data data/train.parquet \
    --exp-data ../exp-transformer/data/train.parquet \
    --output-data data/train.parquet

# Or rebase data if schema changed
python scripts/merge_artifacts.py \
    --strategy rebase \
    --base-data data/train.parquet \
    --exp-data ../exp-transformer/data/train.parquet \
    --output-data data/train.parquet \
    --key-columns id

# Save candidates (CNN model is a candidate)
cp ../exp-cnn/models/production/model.bin models/candidates/cnn-pii.bin
cd models/candidates && dvc add . && cd ../..

# Commit and tag
dvc add results/experiments.db models/production/model.bin data/train.parquet
git add -A
git commit -m "promote transformer PII detector, merge experiment results"
git tag -a e0.3.0 -m "model=transformer | F1=0.94 | promoted | merged 2 experiments"
dvc push
git push --tags

# Cleanup worktrees (tags preserve everything)
git worktree remove ../exp-transformer
git worktree remove ../exp-cnn
git branch -D exp/transformer-pii
git branch -D exp/cnn-pii
```

### Worktree Lifecycle Summary

```
Create   -> ./scripts/setup_worktree.sh NAME  (or: git worktree add + dvc cache dir)
Configure-> (handled by setup script)
Work     -> code, train, evaluate
Save     -> dvc add, git commit, git tag -a
Push     -> dvc push (MUST do before cleanup)
Merge    -> back on main, merge results/data/models
Cleanup  -> git worktree remove, git branch -D
```

---

## Listing Experiments

```bash
# All experiment tags with descriptions
git tag -l 'e*' -n9

# All worktree experiment tags
git tag -l 'exp-*' -n9

# Full experiment history from SQLite
python -c "
from xetrack import Reader
import duckdb
df = Reader('results/experiments.db').to_df()
print(df[['version', 'model', 'f1_score', 'timestamp']].sort_values('timestamp'))
"

# DuckDB CLI for ad-hoc queries
duckdb -c "
INSTALL sqlite; LOAD sqlite;
ATTACH 'results/experiments.db' AS db (TYPE SQLITE);
SELECT version, model, f1_score, timestamp 
FROM db.predictions
ORDER BY f1_score DESC 
LIMIT 10;
"
```
