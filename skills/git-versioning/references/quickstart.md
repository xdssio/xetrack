# Quickstart: Benchmark + Versioning

End-to-end guide bridging the **benchmark** skill (run experiments) and the **git-versioning** skill (version them).

## Setup (Once)

```bash
# 1. Install xetrack with DuckDB support
pip install xetrack[duckdb]

# 2. Initialize DVC in your repo
dvc init
dvc remote add -d myremote s3://my-bucket/dvc-store  # or local path

# 3. Create directory structure
mkdir -p data models/production models/candidates results scripts
```

## First Experiment (Sequential)

Use the **benchmark skill** to design and run:

```python
from dataclasses import dataclass
from xetrack import Tracker

@dataclass(frozen=True, slots=True)
class Params:
    model: str = 'baseline'
    learning_rate: float = 0.001

params = Params()
tracker = Tracker(
    db='results/experiments.db',
    params={'experiment_version': 'e0.1.0', 'model': params.model}
)

# Run benchmark (your code here)
for item in dataset:
    tracker.track(predict, args=[item, params])
```

Then use the **git-versioning skill** to version it:

```bash
# Version artifacts
dvc add results/experiments.db models/production/model.bin

# Commit and tag
git add -A
git commit -m "experiment: e0.1.0 - baseline"
git tag -a e0.1.0 -m "model=baseline | lr=0.001 | acc=0.82 | initial baseline"

# Push
dvc push && git push --tags
```

## Second Experiment (Sequential)

Change params, re-run, version with next tag:

```bash
# After running with new params...
dvc add results/experiments.db models/production/model.bin
git add -A && git commit -m "experiment: e0.1.1 - lr tuning"
git tag -a e0.1.1 -m "model=baseline | lr=0.0001 | acc=0.85 | lower lr"
dvc push && git push --tags
```

## Parallel Experiments (Same Code)

When sweeping parameters with identical code/data, use DuckDB engine:

```python
from concurrent.futures import ThreadPoolExecutor
from xetrack import Tracker

def run_experiment(lr):
    tracker = Tracker(
        db='results/experiments.db',
        engine='duckdb',  # Thread-safe parallel writes
        params={'experiment_version': 'e0.2.0', 'learning_rate': lr}
    )
    tracker.track(train_and_evaluate, args=[lr])

with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(run_experiment, [0.001, 0.0001, 0.00001, 0.000001])
```

Then version the sweep results as one tag.

## Parallel Experiments (Different Code/Data)

When experiments need code or data changes, use git worktrees:

```bash
# Create isolated experiment (auto-configures shared DVC cache)
./scripts/setup_worktree.sh transformer

cd ../exp-transformer
# ... modify code, run experiment, commit, tag, dvc push ...

# Return to main and merge results
cd ../main-project
python scripts/merge_artifacts.py \
    --strategy merge \
    --base-db results/experiments.db \
    --exp-db ../exp-transformer/results/experiments.db

# Promote best model, commit, tag, clean up worktree
```

## Compare Experiments

```bash
# List all experiments ranked by metric
python scripts/experiment_explorer.py list --db results/experiments.db --sort accuracy

# Compare two experiments
python scripts/experiment_explorer.py diff --v1 e0.1.0 --v2 e0.2.0 --db results/experiments.db

# Retrieve a model from a past experiment
python scripts/experiment_explorer.py get --version e0.1.0 --artifact models/production/model.bin --output /tmp/old_model.bin
```

## Skill Responsibilities

| Task | Which Skill |
|---|---|
| Design benchmark structure (tables, params, caching) | Benchmark |
| Schema validation before experiments | Benchmark |
| Run predictions and log metrics | Benchmark |
| Version with git tags and DVC | Git-versioning |
| Parallel experiments with worktrees | Git-versioning |
| Merge/rebase results across experiments | Git-versioning |
| Retrieve historical artifacts | Git-versioning |
| Model promotion and candidates | Git-versioning |
