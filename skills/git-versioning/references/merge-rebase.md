# Merge and Rebase Semantics

DuckDB is the universal engine for merging and rebasing all artifact types: SQLite databases, parquet data files, and model management.

## Core Principle

- **Merge** = accumulate. Non-destructive. Preserves all history and options.
- **Rebase** = replace. Destructive to old state. Enforces consistency.

General heuristic: **merge when accumulating, rebase when replacing.**

Sequential experiments naturally rebase (each version supersedes the last). Parallel experiments naturally merge first (collect all results) then selectively rebase the winner.

---

## SQLite Results Database

### Merge

Append all experiment rows from source into target. No rows overwritten. Every run_id preserved.

```python
import duckdb

def merge_sqlite(base_db: str, exp_db: str, output_db: str, table: str = 'predictions'):
    """Merge experiment DB rows into base DB. Non-destructive."""
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{base_db}' AS base_db (TYPE SQLITE)")
    con.execute(f"ATTACH '{exp_db}' AS exp_db (TYPE SQLITE)")
    
    # Get all rows from experiment not already in base
    con.execute(f"""
        INSERT INTO base_db.{table}
        SELECT * FROM exp_db.{table}
        WHERE track_id NOT IN (SELECT track_id FROM base_db.{table})
    """)
    
    con.close()
```

**When to use:**
- After parallel parameter sweeps (want all results for comparison)
- After worktree experiments (combine results from all branches)
- When building a comprehensive experiment log
- When you need to compare across experiments

### Rebase

Replace rows in the base DB that match on version/key. Old results are overwritten.

```python
def rebase_sqlite(base_db: str, exp_db: str, output_db: str, 
                  table: str = 'predictions', key_column: str = 'version'):
    """Rebase: replace matching rows from experiment into base."""
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{base_db}' AS base_db (TYPE SQLITE)")
    con.execute(f"ATTACH '{exp_db}' AS exp_db (TYPE SQLITE)")
    
    # Delete rows in base that match experiment's key values
    con.execute(f"""
        DELETE FROM base_db.{table}
        WHERE {key_column} IN (SELECT {key_column} FROM exp_db.{table})
    """)
    
    # Insert all experiment rows
    con.execute(f"""
        INSERT INTO base_db.{table}
        SELECT * FROM exp_db.{table}
    """)
    
    con.close()
```

**When to use:**
- Evaluation criteria changed (old metrics not comparable to new)
- Test set updated (need to re-evaluate all experiments against new data)
- Bug fix in evaluation code (old results were wrong)
- Schema migration (experiment DB has new/corrected columns)

---

## Parquet Data Files

### Merge

Additive union. New rows appended, new columns added with nulls for missing values.

```python
def merge_parquet(base_path: str, exp_path: str, output_path: str):
    """Merge: additive union of parquet files. Expands schema."""
    con = duckdb.connect()
    
    # UNION ALL BY NAME aligns columns by name, fills missing with NULL
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{base_path}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{exp_path}')
        ) TO '{output_path}' (FORMAT PARQUET)
    """)
    
    con.close()
```

Key DuckDB feature: `UNION ALL BY NAME` handles schema mismatches gracefully. If the experiment added new columns, old rows get NULLs for those columns.

**When to use:**
- Adding new training samples or annotation batches
- Adding new features that complement (not replace) existing ones
- Combining data from multiple worktree experiments
- Building a larger dataset from incremental additions

### Rebase

Schema transformation. Experiment's changes applied to entire dataset. Matching rows replaced, rest kept.

```python
def rebase_parquet(base_path: str, exp_path: str, output_path: str, 
                   key_columns: list[str]):
    """Rebase: replace matching rows, keep experiment's schema priority."""
    con = duckdb.connect()
    
    keys_condition = " AND ".join(
        f"b.{k} = e.{k}" for k in key_columns
    )
    
    con.execute(f"""
        COPY (
            -- Experiment rows take priority
            SELECT * FROM read_parquet('{exp_path}')
            UNION ALL BY NAME
            -- Base rows that don't exist in experiment
            SELECT b.* FROM read_parquet('{base_path}') b
            WHERE NOT EXISTS (
                SELECT 1 FROM read_parquet('{exp_path}') e
                WHERE {keys_condition}
            )
        ) TO '{output_path}' (FORMAT PARQUET)
    """)
    
    con.close()
```

**When to use:**
- Preprocessing pipeline changed (new tokenization, normalization)
- Schema overhaul (columns renamed, types changed)
- Feature engineering that replaces old features
- Data cleaning that corrects existing rows

---

## Model Files

Models don't use DuckDB directly. Instead, the pattern is filesystem-based with SQLite as the index.

### Merge (Candidates Pattern)

Keep both models. Production stays, experiment stored as candidate.

```python
import shutil

def merge_model(exp_model_path: str, run_id: str, 
                candidates_dir: str = 'models/candidates'):
    """Add experiment model as a candidate."""
    dest = f"{candidates_dir}/{run_id}_model.bin"
    shutil.copy2(exp_model_path, dest)
    # candidates.dvc will track the whole directory
```

After merge, the structure is:
```
models/
├── production/model.bin          # Current best (unchanged)
└── candidates/
    ├── run-abc123_model.bin      # Experiment A
    ├── run-def456_model.bin      # Experiment B
    └── candidates.dvc            # Tracks all candidates
```

**When to use:**
- Evaluating multiple candidates before choosing
- Building model ensembles
- A/B testing in staging
- Keeping backup models

### Rebase (Promote)

Replace production model with experiment's model.

```python
def rebase_model(exp_model_path: str, 
                 production_dir: str = 'models/production'):
    """Promote experiment model to production."""
    dest = f"{production_dir}/model.bin"
    # Old model accessible via previous git tags
    shutil.copy2(exp_model_path, dest)
```

**When to use:**
- Experiment clearly outperforms current production
- Deploying updated model to production
- After A/B test confirms the new model wins

---

## Decision Matrix

| Artifact    | Merge (when)                              | Rebase (when)                                  |
|-------------|-------------------------------------------|------------------------------------------------|
| Results DB  | After sweeps; want full history            | Eval criteria or test data changed on main     |
| Data        | New samples, additive features             | Schema change, preprocessing overhaul          |
| Models      | Multi-candidate evaluation, ensembles      | Clear winner, promoting to production          |

---

## Combined Operations

In practice, you often merge some artifacts and rebase others in the same operation.

### Typical Post-Experiment Combination

After a worktree experiment where the model improved:

```
Results DB: MERGE    (keep all experiment rows for analysis)
Data:       MERGE    (experiment added new training samples)
Model:      REBASE   (experiment's model is better, promote it)
```

After a worktree experiment with a preprocessing change:

```
Results DB: REBASE   (old results used old preprocessing, not comparable)
Data:       REBASE   (new preprocessing applies to all data)
Model:      REBASE   (model trained on new preprocessing)
```

After a parallel sweep:

```
Results DB: MERGE    (always keep all sweep results)
Data:       NO-OP    (data didn't change)
Model:      REBASE   (promote best sweep result to production)
```

---

## DVC Pipeline Stage

Make merge/rebase a reproducible DVC stage:

```yaml
stages:
  merge_results:
    cmd: python scripts/merge_artifacts.py
      --strategy ${merge.strategy}
      --base-db results/experiments.db
      --exp-db results/experiment_incoming.db
      --base-data data/train.parquet
      --exp-data data/experiment_train.parquet
      --output-db results/experiments.db
      --output-data data/train.parquet
      --key-columns ${merge.key_columns}
    deps:
      - scripts/merge_artifacts.py
      - results/experiment_incoming.db
      - data/experiment_train.parquet
    params:
      - merge.strategy
      - merge.key_columns
    outs:
      - results/experiments.db:
          persist: true
      - data/train.parquet:
          persist: true
```

`persist: true` prevents DVC from deleting the output before re-running. Essential for databases you're appending to.

The strategy is a parameter in `params.yaml`:

```yaml
merge:
  strategy: merge     # change to "rebase" when needed
  key_columns:
    - run_id
    - version
```

Switching between merge and rebase: one-line change in `params.yaml`, fully tracked by git.
