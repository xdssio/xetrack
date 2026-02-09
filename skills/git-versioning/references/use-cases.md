# Use Cases and Examples

Concrete scenarios to help the agent understand when and how to apply each workflow.

---

## Use Case 1: First Experiment Baseline

**Scenario:** User just started a project and wants to track their first experiment.

**Workflow:** Sequential

**Steps:**
```bash
# Initialize versioning
dvc init
dvc remote add -d storage s3://my-bucket/project-x
dvc add data/train.parquet data/test.parquet
git add -A && git commit -m "init: project structure with DVC"

# Run baseline experiment
python train.py --model logistic_regression

# Save results
dvc add results/experiments.db models/production/model.bin
git add -A && git commit -m "experiment: e0.1.0 - logistic regression baseline"
git tag -a e0.1.0 -m "model=logreg | F1=0.72 | acc=0.85 | baseline"
dvc push && git push --tags
```

**Agent should:** Set up the full DVC infrastructure, suggest the tag description, and explain the version convention.

---

## Use Case 2: Hyperparameter Sweep

**Scenario:** User has a working baseline and wants to find optimal learning rate and batch size.

**Workflow:** Parallel (same branch, DuckDB engine)

**Key decision:** Code and data are unchanged, only params differ. No worktrees needed.

```python
from xetrack import Tracker
from concurrent.futures import ThreadPoolExecutor

param_grid = [
    {'lr': 1e-3, 'bs': 16}, {'lr': 1e-3, 'bs': 32},
    {'lr': 1e-4, 'bs': 16}, {'lr': 1e-4, 'bs': 32},
    {'lr': 1e-5, 'bs': 16}, {'lr': 1e-5, 'bs': 32},
]

def sweep(params):
    tracker = Tracker('results/experiments.db', engine='duckdb',
                      params={'version': 'e0.2.0', **params})
    return tracker.track(train_and_evaluate, args=[params])

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(sweep, param_grid))
```

**After sweep:**
- Results DB: already merged (all wrote to same DB via DuckDB)
- Model: REBASE (promote best to production)
- Tag: `e0.2.0 -m "model=bert | lr=1e-4 | bs=32 | F1=0.89 | sweep winner from 6 configs"`

**Agent should:** Confirm code/data haven't changed, suggest DuckDB engine, help analyze results, suggest tag description with sweep metadata.

---

## Use Case 3: New Architecture Experiment

**Scenario:** User wants to try a completely different model architecture while keeping the current one intact.

**Workflow:** Worktree

**Key decision:** Code changes required (new model class, different training loop). Worktree gives isolation.

```bash
# Create isolated environment
git worktree add ../exp-transformer -b exp/transformer
cd ../exp-transformer
dvc cache dir ~/.dvc/shared-cache

# Implement new architecture
# ... write transformer model code ...
python train.py --model transformer --version exp-transformer

# Save and tag
dvc add results/experiments.db models/production/model.bin
git add -A && git commit -m "exp: transformer architecture"
git tag -a exp-transformer-v1 -m "model=transformer | heads=8 | layers=6 | F1=0.94 | new arch"
dvc push
```

**Back on main:**
- Results DB: MERGE (want both baseline and transformer results for comparison)
- Data: NO-OP (same data)
- Model: REBASE if transformer wins, MERGE to candidates if not yet decided

**Agent should:** Suggest worktree workflow, remind about shared DVC cache, help decide merge vs rebase after results.

---

## Use Case 4: Data Augmentation Experiment

**Scenario:** User wants to test if adding synthetic data improves performance.

**Workflow:** Worktree (data changes)

```bash
git worktree add ../exp-augmented -b exp/augmented-data
cd ../exp-augmented
dvc cache dir ~/.dvc/shared-cache

# Generate and add synthetic data
python generate_synthetic.py --output data/synthetic.parquet
# Combine with original
python -c "
import duckdb
duckdb.sql('''
    COPY (
        SELECT * FROM read_parquet('data/train.parquet')
        UNION ALL BY NAME
        SELECT * FROM read_parquet('data/synthetic.parquet')
    ) TO 'data/train.parquet' (FORMAT PARQUET)
''')
"

# Train on augmented data
python train.py --version exp-augmented

dvc add data/train.parquet results/experiments.db models/production/model.bin
git add -A && git commit -m "exp: augmented training data with 10k synthetic samples"
git tag -a exp-augmented-v1 -m "model=bert | F1=0.91 | data=augmented+10k | synthetic PII samples"
dvc push
```

**Back on main:**
- Results DB: MERGE
- Data: MERGE (add synthetic samples to main training set if experiment improved things)
- Model: REBASE if improved

**Agent should:** Recognize data change requires worktree, help with parquet merge using DuckDB, suggest merge for data since it's additive.

---

## Use Case 5: Evaluation Pipeline Update

**Scenario:** User changed the test set or evaluation metrics. Old results are no longer comparable.

**Workflow:** Sequential (rerun on main) + Rebase

```python
# New evaluation pipeline
tracker = Tracker('results/experiments.db', 
                  params={'version': 'e0.3.0', 'eval_version': 'v2'})

# Re-evaluate current production model with new metrics
result = tracker.track(evaluate_model, args=[model, new_test_set])
```

**After re-evaluation:**
- Results DB: REBASE (old results measured with different eval, not comparable)
- Data: REBASE if test set changed
- Model: NO-OP (same model, just re-evaluated)

```bash
# Rebase the results
python scripts/merge_artifacts.py \
    --strategy rebase \
    --base-db results/experiments.db \
    --exp-db results/new_eval_results.db \
    --key-columns version
```

Tag: `e0.3.0 -m "eval=v2 | new test set | F1=0.86 | rebased from v1 metrics"`

**Agent should:** Recognize this is a rebase scenario. Warn that old results will be overwritten. Confirm with user. Suggest backing up with a tag before rebasing.

---

## Use Case 6: Multi-Model Comparison (Ensemble Candidate Selection)

**Scenario:** User has 3 model candidates and wants to keep all of them for ensemble evaluation.

**Workflow:** Multiple worktrees, all merged

```bash
# Create 3 experiments
for exp in "bert-large" "deberta" "roberta"; do
    git worktree add ../exp-$exp -b exp/$exp
    cd ../exp-$exp
    dvc cache dir ~/.dvc/shared-cache
    python train.py --model $exp
    dvc add results/ models/
    git add -A && git commit -m "exp: $exp model"
    git tag -a exp-$exp-v1 -m "model=$exp | trained"
    dvc push
    cd ../main-project
done
```

**Back on main:**
- Results DB: MERGE all three
- Models: MERGE all three to candidates (no clear winner yet)
- Data: NO-OP

```bash
# Merge all results
for exp in "bert-large" "deberta" "roberta"; do
    python scripts/merge_artifacts.py \
        --strategy merge \
        --base-db results/experiments.db \
        --exp-db ../exp-$exp/results/experiments.db
done

# All models become candidates
for exp in "bert-large" "deberta" "roberta"; do
    cp ../exp-$exp/models/production/model.bin models/candidates/$exp.bin
done
cd models/candidates && dvc add . && cd ../..

git add -A && git commit -m "merge 3 model candidates for ensemble evaluation"
git tag -a e0.4.0 -m "candidates=bert-large,deberta,roberta | ensemble eval pending"
```

**Agent should:** Suggest keeping all as candidates. The production model stays unchanged until ensemble evaluation is complete. Track all results for comparison.

---

## Use Case 7: Schema Migration

**Scenario:** User wants to add new features to the training data. The feature engineering changes how existing features are computed.

**Workflow:** Worktree + Rebase data

```bash
git worktree add ../exp-features-v2 -b exp/features-v2
cd ../exp-features-v2
dvc cache dir ~/.dvc/shared-cache

# New feature engineering (changes existing columns + adds new ones)
python feature_engineering_v2.py --input data/raw.parquet --output data/train.parquet
python train.py --version exp-features-v2
```

**Back on main:**
- Results DB: REBASE (old results used old features, not comparable)
- Data: REBASE (new feature engineering replaces old, applies to all rows)
- Model: REBASE (model trained on new features)

```python
# Rebase data: new features replace old
python scripts/merge_artifacts.py \
    --strategy rebase \
    --base-data data/train.parquet \
    --exp-data ../exp-features-v2/data/train.parquet \
    --key-columns sample_id
```

**Agent should:** Recognize schema change requires rebase, not merge. Warn user about destructive operation. Suggest creating a backup tag before rebasing.

---

## Use Case 8: Production Hotfix

**Scenario:** Bug found in production model. Need to quickly retrain and deploy.

**Workflow:** Sequential (fast path)

```bash
# Fix the bug
git checkout main
# ... fix code ...

# Quick retrain
python train.py --version e1.0.1  # Patch version

# Fast commit and deploy
dvc add results/experiments.db models/production/model.bin
git add -A && git commit -m "hotfix: e1.0.1 - fix tokenizer bug in PII detector"
git tag -a e1.0.1 -m "HOTFIX | model=bert | F1=0.91 | fixed tokenizer overflow | prod deploy"
dvc push && git push --tags
```

**Agent should:** Recognize urgency, use patch version, include HOTFIX in tag, skip candidate pattern (direct to production).

---

## Use Case 9: Retrieving a Historical Model

**Scenario:** User needs to get a model from 3 months ago for a customer demo.

**Workflow:** Query SQLite + DVC checkout

```python
# Find the model
import duckdb
con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute("ATTACH 'results/experiments.db' AS db (TYPE SQLITE)")

# Query by version or by metrics
results = con.execute("""
    SELECT version, model, f1_score, timestamp 
    FROM db.predictions 
    WHERE timestamp LIKE '2025-11%'
    ORDER BY f1_score DESC
""").fetchall()
print(results)
# [('e0.5.0', 'deberta', 0.93, '2025-11-15 ...'), ...]
```

```bash
# Checkout that version
git stash  # Save current work
git checkout e0.5.0
dvc pull models/production/model.bin
# Model is now at the exact November state

# When done
git checkout main
git stash pop
dvc checkout
```

**Agent should:** Help query the SQLite DB, find the right version, and walk through the checkout/restore process.

---

## Use Case 10: Cleaning Up Old Candidates

**Scenario:** User has accumulated 20 candidate models and wants to prune to top 5.

**Workflow:** Query + selective deletion

```python
import duckdb
import os

con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute("ATTACH 'results/experiments.db' AS db (TYPE SQLITE)")

# Get all candidates ranked (candidates are model files in models/candidates/)
candidates = con.execute("""
    SELECT track_id, version, f1_score, model
    FROM db.predictions
    ORDER BY f1_score DESC
""").fetchall()

# Keep top 5, remove rest
keep = {c[0] for c in candidates[:5]}
remove = [c for c in candidates[5:]]

for track_id, version, f1, model in remove:
    path = f"models/candidates/{track_id}_model.bin"
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed: {version} ({model}, F1={f1})")

# Re-track candidates directory
os.system("cd models/candidates && dvc add .")
```

```bash
git add -A && git commit -m "prune candidates to top 5"
dvc push
```

**Agent should:** Help identify which candidates to keep based on metrics. All removed models remain accessible via their original experiment tags.

---

## Use Case 11: Retrieve a Model from a Specific Experiment

**Scenario:** User needs to load a model from experiment e0.3.0 for inference or comparison.

**Tool:** `experiment_explorer.py get`

```bash
# Quick retrieval (auto-detects best strategy)
python scripts/experiment_explorer.py get \
    --version e0.3.0 \
    --artifact models/production/model.bin \
    --output /tmp/model_v3.bin

# Load and use
python -c "
import pickle
with open('/tmp/model_v3.bin', 'rb') as f:
    model = pickle.load(f)
print(type(model), model.score(X_test, y_test))
"
```

**Agent should:** Use `experiment_explorer.py get` with auto strategy. If the user doesn't have a DVC remote, suggest `--strategy worktree`.

---

## Use Case 12: Check What Data Was Used for an Experiment

**Scenario:** User wants to see which training data was used for experiment e0.2.0.

**Tool:** `experiment_explorer.py show` + `get`

```bash
# First, check experiment metadata (may include data_version, data_hash)
python scripts/experiment_explorer.py show --version e0.2.0 --db results/experiments.db

# If data version is tracked, retrieve the actual data
python scripts/experiment_explorer.py get \
    --version e0.2.0 \
    --artifact data/train.parquet \
    --output /tmp/train_v2.parquet

# Inspect the data
python -c "
import duckdb
con = duckdb.connect()
print(con.execute('SELECT COUNT(*) as rows, * FROM read_parquet(\"/tmp/train_v2.parquet\") LIMIT 5').fetchdf())
"
```

**Agent should:** First check database for metadata columns (data_version, data_hash). Then retrieve the actual data file for inspection.

---

## Use Case 13: Explore Full State of a Past Experiment

**Scenario:** User wants to deeply explore experiment e0.5.0 - browse code, load models, run evaluation scripts, inspect configs.

**Tool:** `experiment_explorer.py checkout`

```bash
# Create a disposable exploration worktree
python scripts/experiment_explorer.py checkout --version e0.5.0

# Enter the exploration environment
cd ../explore-e0.5.0

# Everything is exactly as it was at e0.5.0
ls models/production/    # Model files
cat params.yaml          # Parameters used
python evaluate.py       # Re-run evaluation

# Compare with current code
diff train.py ../main-project/train.py

# When done exploring
cd ../main-project
python scripts/experiment_explorer.py cleanup --version e0.5.0
```

**Agent should:** Suggest checkout for deep exploration. Remind user this is detached HEAD (read-only). If user wants to build on this experiment, suggest creating a proper worktree branch instead.

---

## Use Case 14: Compare Two Experiments

**Scenario:** User wants to understand the difference between experiment e0.2.0 and e0.3.0.

**Tool:** `experiment_explorer.py diff`

```bash
# Side-by-side metric comparison + code diff summary
python scripts/experiment_explorer.py diff \
    --v1 e0.2.0 --v2 e0.3.0 --db results/experiments.db

# Output shows:
# - Which params changed
# - Metric deltas (absolute + percentage)
# - Git diff --stat between versions
```

**Agent should:** Run the diff command. If metrics improved significantly, suggest promoting the better version. If code changed, help the user understand what drove the improvement.

---

## Decision Quick Reference

| User says... | Workflow | Merge/Rebase |
|---|---|---|
| "Try different learning rates" | Parallel (same branch) | DB: auto-merged, Model: rebase best |
| "Try a new architecture" | Worktree | DB: merge, Model: depends on results |
| "Add more training data" | Worktree | DB: merge, Data: merge, Model: rebase |
| "Changed my evaluation metrics" | Sequential | DB: rebase, Model: no-op |
| "Deploy the best model" | Sequential | Model: rebase (promote) |
| "Keep all models for now" | Any | Model: merge (candidates) |
| "Start fresh with new features" | Worktree | DB: rebase, Data: rebase, Model: rebase |
| "Quick fix and redeploy" | Sequential | All: rebase (hotfix) |
| "Compare 3 approaches" | Worktree x3 | DB: merge all, Models: merge to candidates |
| "Get last month's model" | `explorer get` | Auto-retrieves via DVC or worktree |
| "What data was used for X?" | `explorer show` + `get` | Check metadata, then retrieve data file |
| "Let me look at experiment X" | `explorer checkout` | Disposable worktree for full exploration |
| "Compare experiments X and Y" | `explorer diff` | Side-by-side metrics + code diff |
