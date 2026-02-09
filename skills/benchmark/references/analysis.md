# Validation & Analysis

Phase 6 (Validate Results), Phase 7 (Analysis with DuckDB), and Schema Validation. Referenced from SKILL.md.

## Phase 6: Validate Results

Before analysis, check for common pitfalls:

### Run Validation Scripts:

```bash
# 1. Check for data leaks (same input_id evaluated multiple times)
python scripts/validate_benchmark.py benchmark.db predictions

# 2. Check for missing parameters
xt sql benchmark.db "SELECT COUNT(*) FROM db.predictions WHERE params_model_name IS NULL"
```

### Common Pitfalls to Check:

**Data Leakage:**
- Same `input_id` appears with different `track_id` (means it was evaluated in multiple runs)
- Solution: Use cache or check existing results before re-running

**Missing Metadata:**
- NULL values in parameter columns
- Solution: Ensure all params are in frozen dataclass

**Failed Executions:**
```sql
SELECT error, COUNT(*) FROM db.predictions WHERE error IS NOT NULL GROUP BY error
```

---

## Phase 7: Analysis with DuckDB

xetrack + DuckDB = powerful analysis:

### Teach DuckDB CLI:

```bash
# Start DuckDB UI (if duckdb >= 1.2.2)
duckdb -ui

# In the terminal or browser UI:
D INSTALL sqlite; LOAD sqlite;
D ATTACH 'benchmark.db' AS db (TYPE sqlite);
D SELECT * FROM db.predictions LIMIT 10;
```

### Common Analysis Queries:

**Aggregate by model:**
```sql
SELECT
    params_model_name,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(latency) as avg_latency,
    COUNT(*) as n_predictions
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name
ORDER BY accuracy DESC;
```

**Find errors by model:**
```sql
SELECT params_model_name, error, COUNT(*) as count
FROM db.predictions
WHERE error IS NOT NULL
GROUP BY params_model_name, error;
```

**Cache hit rate by model:**
```sql
SELECT
    params_model_name,
    COUNT(CASE WHEN cache != '' THEN 1 END) as cache_hits,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(CASE WHEN cache != '' THEN 1 END) / COUNT(*), 2) as hit_rate
FROM db.predictions
GROUP BY params_model_name;
```

### Teach xetrack Python API:

**Reference:** `examples/07_data_analysis.py` for complete Reader guide

```python
from xetrack import Reader
import pandas as pd

# Read all predictions
# API Reference: Reader(db, engine='sqlite|duckdb', table='predictions')
reader = Reader(db='benchmark.db', engine='duckdb', table='predictions')
df = reader.to_df()  # Gets all data by default
# Or filter: df = reader.to_df(track_id='specific-run-id')

# Calculate accuracy by model
accuracy = df.groupby('params_model_name').apply(
    lambda g: (g['prediction'] == g['ground_truth']).mean()
)
print(accuracy)

# Analyze latency distribution
import matplotlib.pyplot as plt
df.boxplot(column='latency', by='params_model_name')
plt.show()
```

**Validation Checkpoint:**
- ✓ `Reader()` parameters match README documentation?
- ✓ Usage pattern matches `examples/07_data_analysis.py`?
- ✓ DataFrame columns match what `Tracker` logged?

---

### For Large Datasets: Use DuckDB Directly or Polars

**⚠️ Performance Warning:** `Reader.to_df()` uses pandas and loads entire dataset into memory. For large benchmarks (>1M rows), use:

**Option 1: Query DuckDB directly (Recommended for aggregations)**

```python
import duckdb

# Connect to database
conn = duckdb.connect()
conn.execute("INSTALL sqlite; LOAD sqlite;")
conn.execute("ATTACH 'benchmark.db' AS db (TYPE sqlite);")

# Query without loading all data
result = conn.execute("""
    SELECT
        params_model_name,
        AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
        COUNT(*) as n_predictions
    FROM db.predictions
    WHERE error IS NULL
    GROUP BY params_model_name
""").fetchdf()  # Returns small aggregated DataFrame

print(result)
conn.close()
```

**Option 2: Use Polars lazy mode (Recommended for complex transformations)**

```python
import polars as pl

# Lazy query - doesn't load data until .collect()
df = pl.scan_parquet('predictions.parquet')  # or pl.read_database() if supported

# Build query lazily
result = (
    df
    .filter(pl.col('error').is_null())
    .group_by('params_model_name')
    .agg([
        ((pl.col('prediction') == pl.col('ground_truth')).mean()).alias('accuracy'),
        pl.count().alias('n_predictions')
    ])
    .collect()  # Execute only when needed
)

print(result)
```

**Export to Parquet for Polars:**

```bash
# Export from DuckDB to Parquet (more efficient than pandas)
xt sql benchmark.db "COPY (SELECT * FROM db.predictions) TO 'predictions.parquet' (FORMAT PARQUET)"
```

**When to use each:**
- **DuckDB direct**: Best for SQL-style aggregations, filtering, window functions
- **Polars lazy**: Best for complex transformations, joining multiple tables
- **Reader.to_df()**: Only for small datasets (< 100K rows) or quick prototyping

**Validation Checkpoint:**
- ✓ For large data, NOT using `Reader.to_df()`?
- ✓ Using DuckDB directly for aggregations?
- ✓ Using Polars lazy mode if complex transformations needed?

### Export Results:

```bash
# Export to CSV for sharing
xt sql benchmark.db "COPY (SELECT * FROM db.predictions) TO 'results.csv' (HEADER, DELIMITER ',')"

# Generate markdown summary
python scripts/export_summary.py benchmark.db predictions > RESULTS.md
```

---

### Recommended: Schema Validation Before Experiments

**Critical check:** Detect parameter renames or schema drift before running experiments.

**Problem:** If you rename a parameter in code, xetrack creates a NEW column instead of reusing the old one:

```python
# Experiment 1
params = {'learning_rate': 0.001}  # Creates column 'learning_rate'

# Experiment 2 - renamed parameter (BUG!)
params = {'lr': 0.001}  # Creates NEW column 'lr'
# Old data in 'learning_rate', new data in 'lr' - split across columns!
```

**Solution:** Validate schema before running experiments:

```python
from xetrack import Reader
from dataclasses import fields
from difflib import get_close_matches

def validate_schema_before_experiment(db_path, table, new_params_dataclass):
    """
    Compare current database schema with new experiment parameters.
    Detect potential issues: renamed params, similar names, missing columns.
    """
    # 1. Get current schema from database (engine-agnostic via xetrack Reader)
    try:
        reader = Reader(db_path, table=table)
        existing_columns = set(reader.to_df().head(0).columns)
    except Exception:
        existing_columns = set()

    # 2. Extract parameter names from new dataclass
    if hasattr(new_params_dataclass, '__dataclass_fields__'):
        # It's a dataclass
        new_param_names = {f'params_{f.name}' for f in fields(new_params_dataclass)}
    else:
        # It's a dict or object
        new_param_names = {f'params_{k}' for k in new_params_dataclass.__dict__.keys()}

    # 3. Detect potential issues
    issues = []

    # Check for renamed parameters (similar names)
    for new_param in new_param_names:
        if new_param not in existing_columns:
            # Find similar column names (potential renames)
            similar = get_close_matches(new_param, existing_columns, n=1, cutoff=0.6)
            if similar:
                issues.append({
                    'type': 'POTENTIAL_RENAME',
                    'new_param': new_param,
                    'old_param': similar[0],
                    'similarity': 'high'
                })

    # Check for missing parameters (were in schema, not in new code)
    param_columns = {col for col in existing_columns if col.startswith('params_')}
    missing_params = param_columns - new_param_names
    if missing_params:
        issues.append({
            'type': 'MISSING_PARAMS',
            'params': missing_params
        })

    # 4. Report issues and get user input
    if issues:
        print("\n⚠️  SCHEMA VALIDATION ISSUES DETECTED:\n")

        for issue in issues:
            if issue['type'] == 'POTENTIAL_RENAME':
                print(f"❌ Potential parameter rename detected:")
                print(f"   Old column: {issue['old_param']}")
                print(f"   New param:  {issue['new_param']}")
                print(f"\n   This will create a NEW column, splitting data across two columns!")
                print(f"\n   Options:")
                print(f"   1. Rename column in database: ALTER TABLE {table} RENAME COLUMN {issue['old_param']} TO {issue['new_param']}")
                print(f"   2. Change code to use old name: {issue['old_param']}")
                print(f"   3. Confirm this is intentional (creates new column)\n")

            elif issue['type'] == 'MISSING_PARAMS':
                print(f"⚠️  Parameters from previous experiments missing in new code:")
                for param in issue['params']:
                    print(f"   - {param}")
                print(f"\n   If these were renamed, use ALTER TABLE to rename columns.")
                print(f"   If intentionally removed, this is OK (old data preserved).\n")

        print("   Actions:")
        print("   [r] Rename column in database (recommended if parameter was renamed)")
        print("   [c] Continue anyway (will create new columns)")
        print("   [a] Abort (fix code first)")

        choice = input("\n   Your choice: ").strip().lower()

        if choice == 'r':
            # Help user rename column
            for issue in issues:
                if issue['type'] == 'POTENTIAL_RENAME':
                    old_col = issue['old_param']
                    new_col = issue['new_param']
                    print(f"\n📝 Renaming {old_col} → {new_col}")

                    # Execute rename (SQLite-specific DDL operation)
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")
                    conn.commit()
                    conn.close()

                    print(f"✅ Column renamed successfully!")

            print("\n✅ Schema updated. Safe to run experiment.")
            return True

        elif choice == 'c':
            print("⚠️  Continuing with schema drift. New columns will be created.")
            return True

        else:  # 'a' or anything else
            print("❌ Experiment aborted. Fix code or schema first.")
            return False

    else:
        print("✅ Schema validation passed - no issues detected")
        return True

# Usage before running experiment:
if not validate_schema_before_experiment('benchmark.db', 'predictions', ModelParams):
    exit(1)  # Don't run experiment if validation failed
```

**Example output:**

```
⚠️  SCHEMA VALIDATION ISSUES DETECTED:

❌ Potential parameter rename detected:
   Old column: params_learning_rate
   New param:  params_lr

   This will create a NEW column, splitting data across two columns!

   Options:
   1. Rename column in database: ALTER TABLE predictions RENAME COLUMN params_learning_rate TO params_lr
   2. Change code to use old name: params_learning_rate
   3. Confirm this is intentional (creates new column)

   Actions:
   [r] Rename column in database (recommended if parameter was renamed)
   [c] Continue anyway (will create new columns)
   [a] Abort (fix code first)

   Your choice: r

📝 Renaming params_learning_rate → params_lr
✅ Column renamed successfully!
✅ Schema updated. Safe to run experiment.
```

**Why this matters:**
- **Prevents data fragmentation** - Keeps related data in one column
- **Maintains clean schema** - Avoids accumulating renamed columns
- **SQLite flexibility** - Easy to rename columns with ALTER TABLE
- **Catches mistakes early** - Before running expensive experiments

**Best practice workflow:**
1. Make code changes
2. Run schema validation (detects renames)
3. Fix schema or code
4. Run experiment with clean schema

For pre-run data validation, DVC commit checks, hash tracking, and experiment history queries, see the **`git-versioning` skill** which provides `scripts/version_tag.py` and `scripts/experiment_explorer.py` for these workflows.
