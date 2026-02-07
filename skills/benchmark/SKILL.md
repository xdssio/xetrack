---
name: benchmark
description: Guide users through rigorous ML/AI benchmarking experiments using xetrack. Use when users want to: (1) Compare ML models, hyperparameters, or architectures, (2) Benchmark LLM prompts, few-shot examples, or generation strategies, (3) Evaluate data processing pipelines or embeddings, (4) Set up reproducible experiments with caching and validation, (5) Debug existing benchmarks for data leaks or inconsistencies, (6) Analyze benchmark results with SQL/DuckDB. Helps design experiments end-to-start following single-execution principles.
---

# Benchmark Skill

Guide users through methodologically rigorous ML/AI benchmarking using xetrack, from experiment design to analysis.

## Core Philosophy

**Design experiments end-to-start.** Build robust single-execution functions first, cache aggressively, save raw responses, and version everything. The goal: implement once, run once, analyze without pain.

> *"If you get the unglamorous parts right—experiment design, execution, logging, and failure handling—analysis becomes almost boring. If you don't, every surprising result triggers a debugging session instead of insight."*

**Critical principle:** Design experiments so that **contradictory outcomes are possible and informative**, not just confirmatory. Avoid confirmation bias—your setup should be robust enough to distinguish genuine measurements from genuine discoveries.

## Prerequisites

**CRITICAL:** Before starting any benchmark, ensure xetrack is properly set up and understood:

### Step 0: Setup & Learn xetrack

1. **Verify git branch:**
   ```bash
   git branch --show-current
   # Must be on a feature branch (e.g., feat/experiment-name), not main!
   # If on main: git checkout -b feat/benchmark-experiment
   ```

2. **Install xetrack and dependencies:**
   ```bash
   pip install xetrack[duckdb,cache,assets]
   pip install polars  # For large dataset analysis (optional)
   ```

3. **Read xetrack documentation:**
   - **MUST READ**: `/path/to/xetrack/README.md` - Full API documentation
   - **MUST READ**: `/path/to/xetrack/examples/README.md` - Examples guide
   - **Run example**: `python examples/01_quickstart.py` to verify installation

4. **Understand core concepts:**
   - `Tracker(db, params=dict, engine='sqlite|duckdb', table='events')` - Main tracking interface
   - `tracker.log(dict)` - Log arbitrary data
   - `tracker.track(func, args, kwargs)` - Track function execution with auto-unpacking
   - `Reader(db, engine='sqlite|duckdb', table='events')` - Read tracked data
   - `xt` CLI - Command-line interface for queries

**⚠️ IMPORTANT:** xetrack is not a common package. DO NOT hallucinate APIs. Always reference README/examples for correct usage patterns.

## Workflow

Follow this sequential workflow with users:

### 1. Understand Goals → Design Experiment
### 2. Build Single-Execution Function → Validate
### 3. Add Caching → Test Reproducibility
### 4. Parallelize (if needed) → Run
### 5. Validate Results → Analyze

---

## Phase 1: Understand Goals & Design Experiment

**Start from the end.** Before writing code, clarify:

### Ask the User:

1. **"What questions do you want to answer?"**
   - Example: "Is Model A better than Model B?" "Which prompt works best?" "Does my preprocessing improve accuracy?"

2. **"What comparisons or segmentations will you perform?"**
   - This determines what parameters to track
   - Examples: model type, hyperparameters, data subsets, prompt variations
   - **Common segmentations**: prompt length, domain, difficulty level, failure mode, cost bracket

   > *"Storage is cheap; regret is expensive. Future you will appreciate past you's paranoia."*

3. **"What metrics matter?"**
   - Accuracy, F1, latency, **cost** (especially for LLMs!), throughput?
   - Save raw outputs (probabilities, full LLM responses) for future re-analysis
   - **For LLMs**: Track token counts, cost per prediction, API latency

4. **"What data are you benchmarking on?"**
   - Size, format, location
   - Will it fit in memory? Need batching?

### Design Decisions:

Based on answers, recommend:

**Database Engine Decision Matrix:**

| Factor | SQLite | DuckDB |
|--------|--------|--------|
| **Multiprocessing** | ✅ Works | ❌ Database locks |
| **Single-process** | ✅ Works | ✅ Works |
| **Analytics queries** | ⚠️ Limited | ✅ Advanced SQL |
| **Large datasets** | ⚠️ Slower | ✅ Faster |
| **Schema flexibility** | ✅ Add columns later | ✅ Add columns later |
| **SQL table names** | `predictions` | `db.predictions` |
| **Default** | ✅ Yes | No |

**Note on table naming:** In Python code, always use simple table names like `table='predictions'`. The `db.` prefix is only needed when querying via DuckDB CLI after attaching the SQLite file: `ATTACH 'benchmark.db' AS db (TYPE sqlite);`

**Decision flowchart:**
```
Will you use multiprocessing?
├─ YES → Use SQLite (engine='sqlite')
└─ NO  → Use DuckDB (engine='duckdb') for better analytics
```

**Installation:**
```bash
pip install xetrack[duckdb]  # For DuckDB support
```

**Important:** Once you choose an engine, use it consistently throughout your benchmark!

**Table Organization (IMPORTANT - Two Tables Pattern):**

Always create **two separate tables**:

1. **Predictions table** (e.g., `predictions`, `training_steps`, `requests`)
   - Stores every single execution/prediction
   - One row per datapoint
   - Enables detailed segmentation analysis
   - Format: `(track_id, timestamp, input_id, model, params..., prediction, ground_truth, latency, error...)`

2. **Metrics table** (e.g., `metrics`, `final_metrics`, `throughput_summary`)
   - Stores aggregated results per experiment configuration
   - One row per parameter combination
   - Enables quick model comparison
   - Format: `(track_id, timestamp, model, params..., accuracy, avg_latency, total_cost, n_samples...)`

**Naming suggestions:**
- Predictions: `predictions`, `inferences`, `training_steps`, `eval_checkpoints`, `requests`
- Metrics: `metrics`, `summary`, `final_metrics`, `experiment_results`

Users can customize table names, but this two-table pattern is recommended.

**Parameter Tracking:**
- Use frozen dataclasses for all experiment parameters (enables caching)
- Track git commit hash if reproducibility is critical
- Track data version (use DVC commit hash: `git log -n 1 --pretty=format:%H -- data.dvc`)
- Track timestamps, model versions, hardware specs (xetrack does this automatically)
- **Experiment naming:** xetrack uses coolname (e.g., `purple-mountain-4392`) - human-readable and unique

### Output from Phase 1:

A clear specification:
```python
# Example output
"""
Goal: Compare 3 embedding models on text classification
Data: 1000 labeled examples
Metrics: Accuracy, F1, inference latency
Params to track: model_name, embedding_dim, batch_size
Database: benchmark.db with DuckDB engine
Tables: predictions (individual), experiments (aggregated)
"""
```

---

## Phase 2: Build Single-Execution Function

**Critical principle: Every datapoint executes exactly once.**

### Single-Execution Pattern:

```python
from dataclasses import dataclass
from xetrack import Tracker

@dataclass(frozen=True, slots=True)  # frozen=True makes it hashable for caching
class BenchmarkParams:
    """All parameters that affect the result."""
    model_name: str
    embedding_dim: int
    temperature: float = 0.0
    batch_size: int = 32
    # Use immutable types only: no lists/dicts, use tuples/frozensets

def run_single_prediction(input_data: dict, params: BenchmarkParams) -> dict:
    """
    Single-execution function: stateless, thread-safe, cacheable.

    Returns everything you might need later, including:
    - prediction (the actual output)
    - raw_response (for re-analysis)
    - latency
    - error (if any)
    - metadata
    """
    import time
    start = time.time()

    try:
        # Your model inference here
        prediction = your_model.predict(input_data, params)
        raw_response = your_model.get_raw_output()
        error = None
    except Exception as e:
        prediction = None
        raw_response = None
        error = str(e)

    latency = time.time() - start

    return {
        'input_id': input_data['id'],
        'prediction': prediction,
        'raw_response': raw_response,  # ALWAYS save raw outputs
        'ground_truth': input_data.get('label'),
        'latency': latency,
        'error': error
    }
```

### Teach xetrack: Track with Frozen Dataclass

**Reference:** `examples/03_dataclass_unpacking.py` for complete example

```python
# Initialize tracker with DuckDB
# API Reference: Tracker(db, engine='sqlite|duckdb', table='events', params=dict, cache=str)
tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',  # Recommended for benchmarks (or 'sqlite' for multiprocessing)
    table='predictions',
    params={'experiment_id': 'exp-001'}  # Groups this run
)

# Track execution - dataclass params are auto-unpacked!
# API Reference: tracker.track(function, args=list, kwargs=dict, params=dict)
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
result = tracker.track(
    run_single_prediction,
    args=[input_data, params]
)
# Tracked columns: input_id, prediction, latency, error, params_model_name, params_embedding_dim...
```

**Xetrack Feature: Automatic Dataclass Unpacking**
- Frozen dataclasses are automatically unpacked into individual columns
- `params.model_name` → `params_model_name` column
- Enables easy filtering: `SELECT * WHERE params_model_name = 'bert-base'`
- See `examples/03_dataclass_unpacking.py` for full documentation

**Validation Checkpoint:**
- ✓ Verify `Tracker()` parameters match README documentation
- ✓ Confirm `tracker.track()` usage matches `examples/02_track_functions.py`
- ✓ Check dataclass unpacking works as in `examples/03_dataclass_unpacking.py`

### Validation Checklist:

Before proceeding, validate:

- [ ] Function is **stateless** (no shared mutable state like global lists)
- [ ] Function is **deterministic** (same inputs → same outputs)
- [ ] Function returns **everything you might need** (including errors)
- [ ] Parameters are in a **frozen dataclass** (for caching)
- [ ] **Raw outputs are saved** (not just processed results)
- [ ] **Failures are captured** (error field, not silent failures)

**Run validation script:**
```bash
python scripts/validate_benchmark.py benchmark.db
```

---

## Phase 3: Add Caching

**Caching is not optimization—it's a correctness tool.** Prevents duplicate executions and wasted compute.

### Enable Caching:

**Reference:** `examples/05_function_caching.py` for complete caching guide

**Installation:** `pip install xetrack[cache]` (requires diskcache)

```python
# API Reference: Tracker(cache='directory_path')
tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    cache='cache_dir',  # Enable disk-based caching
    table='predictions'
)

# First call: computes and caches
result1 = tracker.track(run_single_prediction, args=[input_data, params])

# Second call with same args: instant cache hit!
result2 = tracker.track(run_single_prediction, args=[input_data, params])
```

**Validation Checkpoint:**
- ✓ Installed `xetrack[cache]`?
- ✓ Cache parameter usage matches `examples/05_function_caching.py`?
- ✓ Verify cache directory is created
- ✓ Test cache hit by re-running same function call

**Caching Requirements:**
- All function arguments must be **hashable** (use frozen dataclasses)
- **Treat floats as hostile**—round or quantize before hashing (`round(temperature, 2)`)
- Lists/dicts break caching—use tuples/frozensets instead
- **"Cache you cannot observe is a liability"** - always verify caching works with validation scripts

**LLM-Specific Caching:**
For LLM benchmarks, consider specialized caching:
- **GPTCache**: Semantic similarity-based caching for LLM responses
- **LangChain caching**: Built-in cache backends for chat models
- These can cache even with slight prompt variations (semantic matching)
- Especially useful for expensive API calls (GPT-4, Claude)

**Cache Lineage Tracking:**
- xetrack tracks cache hits via the `cache` column
- Empty string `""` = computed (cache miss)
- track_id value = cache hit (references original execution)

### Teach xetrack CLI: Check Caching

**Reference:** README.md "CLI" section for all `xt` commands

```bash
# View first rows (xt head)
# API Reference: xt head <db> --n=<rows> --engine=<sqlite|duckdb>
xt head benchmark.db --n=10 --engine=duckdb
# Look for 'cache' column - track_id values indicate cache hits

# Execute SQL query (xt sql)
# API Reference: xt sql <db> "<query>" --engine=<sqlite|duckdb>
xt sql benchmark.db "SELECT input_id, params_model_name, cache FROM db.predictions LIMIT 10"

# Other useful commands:
xt tail benchmark.db --n=10  # View last rows
xt stats describe benchmark.db --columns=latency,accuracy  # Statistics
```

**Validation Checkpoint:**
- ✓ `xt` command available? (run `xt --help`)
- ✓ Commands match README CLI documentation?
- ✓ SQL syntax correct for engine (SQLite vs DuckDB)?

### Validation:

```bash
# Check cache effectiveness
python scripts/analyze_cache_hits.py benchmark.db predictions
```

Expected output:
```
Cache Analysis:
- Total executions: 1000
- Cache hits: 0 (0.0%) ← First run
- Cache misses: 1000 (100.0%)
- Unique parameter combinations: 3
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Dataclass Unpacking Only Works with `.track()`

**Problem:**
```python
# This WON'T unpack dataclass params
tracker = Tracker(db='bench.db', params={'run_id': 'exp1'})
# Result: column named 'run_id', not 'params_run_id'
```

**Solution:**
- `Tracker(params={...})` stores keys as-is (no `params_` prefix)
- Dataclass unpacking ONLY happens when dataclass is a **function argument**:
```python
@dataclass(frozen=True)
class Config:
    model: str

def predict(data, config: Config):  # config is argument
    ...

tracker.track(predict, args=[data, Config(model='bert')])
# Result: columns include 'params_model', 'params_config_model'
```

### Pitfall 2: DuckDB + Multiprocessing = Database Locks

**Problem:**
```python
# This FAILS with "database is locked" error
with Pool(processes=4) as pool:
    results = pool.map(worker_func, data)  # Multiple processes write to DuckDB
```

**Solution:**
- **Use SQLite engine** for multiprocessing (handles concurrent writes better):
```python
tracker = Tracker(db='bench.db', engine='sqlite')  # NOT duckdb
```

- **Or use threading** instead of multiprocessing (single process, multiple threads)
- **Or batch results** in memory and write sequentially

### Pitfall 3: System Monitoring + Multiprocessing = AssertionError

**Problem:**
```python
# This FAILS: "daemonic processes are not allowed to have children"
tracker = Tracker(db='bench.db', log_system_params=True)
with Pool(processes=4) as pool:  # Fails!
    ...
```

**Solution:**
```python
# Disable system monitoring in multiprocessing contexts
tracker = Tracker(db='bench.db', log_system_params=False)
```

- System monitoring spawns child processes, incompatible with Pool workers
- Use system monitoring ONLY in single-process or threading contexts

### Pitfall 4: Model Objects Bloat Database

**Problem:**
```python
def train(X, y, params):
    model = RandomForest().fit(X, y)
    return {'model': model}  # Stores full model in database!

tracker.track(train, args=[X, y, params])
# Database grows to 100s of MB
```

**Solution:**
- **Save model hash only** (requires `pip install xetrack[assets]`):
```python
return {'model': model}  # xetrack saves as asset, stores hash in DB
```

- **Or don't return model** from tracked function (save separately):
```python
return {'model_path': 'models/model_001.pkl'}
# Save model outside of tracking
```

### Pitfall 5: Cache Column Missing

**Problem:**
Cache directory is created but `cache` column doesn't appear in database.

**Root causes:**
1. **Most likely:** DuckDB engine may not populate cache column (known limitation)
2. Cache feature requires `pip install xetrack[cache]` (diskcache)
3. xetrack version may not support cache tracking

**Solution:**
```python
# Try SQLite engine instead
tracker = Tracker(db='bench.db', engine='sqlite', cache='cache_dir')  # Not duckdb

# Verify cache column appears
df = Reader('bench.db', engine='sqlite').to_df()
print('cache' in df.columns)  # Should be True

# If still missing, verify diskcache installed:
# pip install xetrack[cache]
```

**Workaround:** Cache still works even if column is missing (results are cached), but you won't have lineage tracking.

### Pitfall 6: Float Parameters Break Caching

**Problem:**
```python
@dataclass(frozen=True)
class Config:
    learning_rate: float  # 0.0001 vs 0.00010000001 are different!

# These are treated as different configs even if functionally identical
tracker.track(train, args=[Config(learning_rate=1e-4)])
tracker.track(train, args=[Config(learning_rate=0.0001)])  # Cache miss!
```

**Solution:**
```python
@dataclass(frozen=True)
class Config:
    learning_rate: float

    def __post_init__(self):
        # Round floats for consistent hashing
        object.__setattr__(self, 'learning_rate', round(self.learning_rate, 6))
```

### Pitfall 7: Metrics Table Doesn't Have `params_*` Columns

**Problem:**
```python
metrics_tracker.log({
    'model': 'bert',
    'accuracy': 0.85
})
# Metrics table has 'model', NOT 'params_model'
```

**This is expected!** When using `.log()`, you manually control column names. To match predictions table format, manually include params:

```python
metrics_tracker.log({
    'model_type': params.model_type,  # Match your param names
    'regularization': params.regularization,
    'accuracy': accuracy
})
```

---

## Phase 4: Parallelize (Optional)

If benchmark is slow, parallelize after validating single-execution:

```python
from multiprocessing import Pool
from functools import partial

# Stateless function is safe to parallelize
def run_with_tracker(item, params):
    tracker = Tracker(db='benchmark.db', engine='duckdb',
                     cache='cache_dir', table='predictions')
    return tracker.track(run_single_prediction, args=[item, params])

# Parallel execution
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
with Pool(processes=4) as pool:
    results = pool.map(partial(run_with_tracker, params=params), data)
```

**Important:** Each process creates its own tracker. DuckDB/SQLite handle concurrent writes safely.

---

## Phase 5: Run Full Benchmark Loop

### Pre-Run Validation Checklist

Before running experiments, validate your setup:

```python
# 1. SCHEMA VALIDATION (recommended)
# Detect parameter renames and schema drift
if not validate_schema_before_experiment('benchmark.db', 'predictions', ModelParams):
    print("❌ Schema validation failed. Fix issues before running.")
    exit(1)

# 2. DATA COMMIT CHECK (optional)
# Ensure data.dvc is committed for reproducibility
# check_data_committed()  # Uncomment if using DVC
```

### Recommended Workflow: Debug Small, Then Scale

**Best practice:** Start with a small subset to debug your pipeline before running the full benchmark.

```python
# 1. DEBUG MODE: Test with small subset first
DEBUG = True  # Set to False for full run
dataset_subset = dataset[:10] if DEBUG else dataset  # Only 10 items for testing

# Run benchmark on subset
for item in dataset_subset:
    result = predictions_tracker.track(run_prediction, args=[item, params])

# 2. Check results look correct
print(f"Processed {len(dataset_subset)} items")
reader = Reader(db='benchmark.db', table='predictions')
print(reader.to_df().tail())

# 3. If everything looks good, delete test runs to keep database clean
if DEBUG:
    # Get track_id from test run
    track_ids = reader.to_df()['track_id'].unique()
    for tid in track_ids:
        print(f"Deleting test track_id: {tid}")
        # Delete test data (CLI method)
        subprocess.run(['xt', 'delete', 'benchmark.db', tid], check=True)
        # Or use Python API (if available)
        # tracker.delete(track_id=tid)

    print("✅ Test runs deleted. Ready to run full benchmark.")
    print("   Set DEBUG = False and run again.")
```

**Why this matters:**
- **Catch bugs early** - Fix issues with 10 items, not 10,000
- **Save time** - Don't wait hours to discover a parameter was wrong
- **Clean database** - Delete test runs so they don't pollute analysis
- **Iterate quickly** - Test → debug → delete → repeat until perfect

**Deleting test runs:**
```bash
# CLI: Delete by track_id
xt delete benchmark.db ancient-falcon-1234

# Delete multiple track_ids
for tid in test-id-1 test-id-2; do
    xt delete benchmark.db $tid
done

# Or delete all data from specific table (nuclear option)
xt sql benchmark.db "DELETE FROM db.predictions WHERE track_id = 'ancient-falcon-1234'"
```

### Full Benchmark

Once validated with subset, run the full benchmark using **both tables**:

```python
from xetrack import Tracker, Reader

# Run predictions for all parameter combinations
params_grid = [
    BenchmarkParams(model_name='bert-base', embedding_dim=768),
    BenchmarkParams(model_name='roberta-base', embedding_dim=768),
    BenchmarkParams(model_name='distilbert', embedding_dim=768),
]

for params in params_grid:
    print(f"Running: {params.model_name}...")

    # Tracker for individual predictions
    predictions_tracker = Tracker(
        db='benchmark.db',
        engine='duckdb',
        cache='cache_dir',
        table='predictions',  # Individual results
        params={'experiment_id': f'exp-{params.model_name}'}
    )

    # Tracker for aggregated metrics
    metrics_tracker = Tracker(
        db='benchmark.db',
        engine='duckdb',
        table='metrics',  # Aggregated results
        params={'experiment_id': f'exp-{params.model_name}'}
    )

    # Run predictions
    results = []
    for item in dataset:
        result = predictions_tracker.track(run_single_prediction, args=[item, params])
        results.append(result)

    # Calculate and log aggregated metrics
    successful = [r for r in results if r.get('error') is None]
    if successful:
        accuracy = sum(1 for r in successful if r['prediction'] == r['ground_truth']) / len(successful)
        avg_latency = sum(r['latency'] for r in successful) / len(successful)

        metrics_tracker.log({
            'model_name': params.model_name,
            'embedding_dim': params.embedding_dim,
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'n_predictions': len(dataset),
            'n_successful': len(successful),
            'n_failed': len(results) - len(successful)
        })

    print(f"  Completed {len(dataset)} predictions - Accuracy: {accuracy:.4f}")
```

**Why two tables?**
- **Predictions table**: Detailed data for segmentation ("which examples did Model A get wrong?")
- **Metrics table**: Quick comparison ("which model is best overall?")

**Rerun Safety:** If script crashes, restart it! Cached results prevent re-execution.

---

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
# API Reference: Reader(db, engine='sqlite|duckdb', table='events')
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

## Common Patterns by Use Case

### Pattern 1: sklearn Model Comparison

See `assets/sklearn_benchmark_template.py` for complete example.

**Key points:**
- Use frozen dataclass for hyperparameters
- Cache fitted models in xetrack assets (requires `pip install xetrack[assets]`)
- Track both training and test metrics

### Pattern 2: LLM Prompt Benchmarking

See `assets/llm_finetuning_template.py` for complete example.

**Key points:**
- Save full LLM response (not just parsed output)
- Track token counts, cost, latency
- Cache responses to avoid re-querying
- Handle failures gracefully (rate limits, timeouts)

### Pattern 3: Load Testing / Throughput

See `assets/throughput_benchmark_template.py` for complete example.

**Key points:**
- Use `log_system_params=True` for CPU/memory tracking
- Measure requests per second, p50/p95/p99 latency
- Simulate concurrent load with multiprocessing

---

## Helper Scripts

All scripts are in `scripts/`:

- **`validate_benchmark.py`** - Check for data leaks, duplicate executions
- **`analyze_cache_hits.py`** - Analyze caching effectiveness
- **`export_summary.py`** - Generate markdown summaries

Usage:
```bash
python scripts/validate_benchmark.py <db_path> <table_name>
```

---

## Data & Database Versioning with DVC

**Strongly recommend using DVC** for benchmarks that will be rerun or shared.

### Why DVC?

- **Data versioning**: Track dataset changes separately from code
- **Database versioning**: Version your benchmark.db results
- **Reproducibility**: Know exactly which data produced which results
- **Storage efficiency**: Git stores pointers, DVC stores actual data remotely

### Quick Setup:

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gdrive, dvc-azure, etc.

# Initialize
dvc init

# Track your data
dvc add data/
dvc add benchmark.db

# This creates data.dvc and benchmark.db.dvc
git add data.dvc benchmark.db.dvc .dvc/.gitignore
git commit -m "feat(benchmark): add dataset and initial results"

# Push data to remote storage (not git!)
dvc remote add -d storage s3://my-bucket/xetrack-benchmarks
dvc push
```

### How Rigorous Should Versioning Be?

**Decision tree:**

#### Minimal (solo exploration, throwaway analysis)
- ✅ Git for code
- ❌ Skip DVC
- Track git hash in params (optional)

#### Standard (team experiments, production benchmarks)
- ✅ Git for code
- ✅ DVC for data + database
- ✅ Track `data.dvc` commit hash in params
- ✅ Commit after each experiment run
- Pattern: `git add benchmark.db.dvc && git commit -m "experiment: purple-mountain results"`

#### Maximum (research, audits, regulatory)
- ✅ Everything from Standard
- ✅ Branch-per-experiment pattern
- ✅ Track full git state (hash, branch, dirty status)
- ✅ Lock data.dvc during experiment (no changes mid-run)
- ✅ Push to DVC remote before merging

**Example tracker setup:**

```python
def get_data_version():
    """Get data.dvc commit hash."""
    return subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()

tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    params={
        'data_version': get_data_version(),  # Critical for reproducibility
        'git_hash': get_git_hash(),
        'experiment_name': 'purple-mountain-4392'
    }
)
```

### Database Management

**Pattern 1: Append-Only (Simplest)**
- Single `benchmark.db` file
- New experiments append rows
- Version with DVC: `dvc add benchmark.db && git commit`
- ✅ Simple, works for most cases
- ⚠️ Database grows over time

**Pattern 2: Table-per-Experiment**
- Use different table names: `predictions_exp001`, `predictions_exp002`
- Same database file
- ✅ Clean separation
- ✅ Easy to compare experiments with SQL JOINs
- ⚠️ More complex table management

```python
# Table-per-experiment
experiment_name = 'exp001'
tracker = Tracker(
    db='benchmark.db',
    table=f'predictions_{experiment_name}',
    params={'experiment': experiment_name}
)
```

**Pattern 3: Database-per-Experiment (Clean)**
- Different files: `exp001.db`, `exp002.db`
- DVC tracks each separately
- ✅ Clean separation, easy to archive
- ✅ Can delete old experiments easily
- ⚠️ Harder to compare across experiments

```python
# Database-per-experiment
tracker = Tracker(
    db=f'{experiment_name}.db',
    table='predictions'
)
```

**Recommendation:**
- Start with **Pattern 1** (append-only)
- Move to **Pattern 2** if database > 100MB or > 10 experiments
- Use **Pattern 3** for completely independent experiments

### Cleaning Up Old Experiments

```bash
# List all tracked databases
ls *.db.dvc

# Remove old experiment (DVC keeps it in cache)
dvc remove old_experiment.db.dvc
git add old_experiment.db.dvc
git commit -m "chore: archive old experiment"

# Fully delete from DVC cache (permanent!)
dvc gc --workspace --force
```

---

## Advanced: Reproducibility

For maximum reproducibility, track git state and data versions:

```python
import subprocess

def get_git_hash():
    """Get current code commit hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def get_data_version():
    """Get data version from DVC (if using DVC)."""
    try:
        return subprocess.check_output(
            ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
        ).decode().strip()
    except:
        return None

tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    params={
        'git_hash': get_git_hash(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current']).decode().strip(),
        'data_version': get_data_version()  # Tracks data changes via DVC
    }
)
```

**Branch-per-Experiment Pattern** (maximum reproducibility):
1. Create experiment branch: `git checkout -b experiment/purple-mountain-4392`
2. Track branch name in params: `'git_branch': 'experiment/purple-mountain-4392'`
3. Commit code + data.dvc + benchmark.db.dvc
4. Push to remote (makes it immutable)
5. Can always `git checkout experiment/purple-mountain-4392` to reproduce exactly

**Trade-off:** High overhead, many branches. Consider if reproducibility justifies complexity.

---

## Git Tag-Based Experiment Versioning

**Recommended pattern for clean experiment history:**

### Workflow:

```python
import subprocess

def get_latest_experiment_tag():
    """Get the latest experiment tag (e*), or return e0.0.0 if none exist."""
    try:
        # Get all tags matching experiment pattern (e*)
        all_tags = subprocess.check_output(
            ['git', 'tag', '-l', 'e*']
        ).decode().strip().split('\n')

        if not all_tags or all_tags == ['']:
            return 'e0.0.0'

        # Sort and get latest
        return sorted(all_tags, key=lambda t: [int(x) for x in t.lstrip('e').split('.')])[-1]
    except:
        return 'e0.0.0'

def increment_tag(tag: str) -> str:
    """Increment patch version: e0.0.5 -> e0.0.6"""
    parts = tag.lstrip('e').split('.')
    parts[-1] = str(int(parts[-1]) + 1)
    return 'e' + '.'.join(parts)

# 1. Get next experiment version
latest_tag = get_latest_experiment_tag()  # e.g., 'e0.0.5'
next_tag = increment_tag(latest_tag)  # 'e0.0.6'

print(f"Running experiment: {next_tag}")

# 2. Run experiment with tag as parameter
tracker_predictions = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='predictions',
    params={'experiment_version': next_tag}  # Tag all rows
)

tracker_metrics = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='metrics',
    params={'experiment_version': next_tag}
)

# 3. Run your benchmark
for item in dataset:
    result = tracker_predictions.track(predict, args=[item, params])

# 4. Log aggregated metrics
tracker_metrics.log({
    'model': 'bert-base',
    'accuracy': 0.85,
    'experiment_version': next_tag  # Explicitly include tag
})

# 5. Commit database with tag
subprocess.run(['git', 'add', 'benchmark.db'], check=True)
subprocess.run(['git', 'commit', '-m', f'experiment: {next_tag} results'], check=True)

# 6. Generate tag description based on experiment
def generate_tag_description(results: dict, params: dict) -> str:
    """Generate informative tag description from experiment results."""
    parts = []

    # Model/config info
    if 'model' in params:
        parts.append(f"model={params['model']}")
    if 'learning_rate' in params:
        parts.append(f"lr={params['learning_rate']}")
    if 'batch_size' in params:
        parts.append(f"batch={params['batch_size']}")

    # Results
    if 'accuracy' in results:
        parts.append(f"acc={results['accuracy']:.4f}")
    if 'loss' in results:
        parts.append(f"loss={results['loss']:.4f}")

    # Data info (if tracked)
    data_version = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%h', '--', 'data.dvc']
    ).decode().strip()
    if data_version:
        parts.append(f"data={data_version[:7]}")

    return ' | '.join(parts)

# Auto-generate description
tag_description = generate_tag_description(
    results={'accuracy': 0.85},
    params={'model': 'bert-base', 'learning_rate': 1e-4}
)
# Example: "model=bert-base | lr=0.0001 | acc=0.8500 | data=a3f2b1c"

# Let user review/override
print(f"\n📝 Suggested tag description:")
print(f"   {tag_description}")
response = input("   Use this description? [Y/n/edit]: ").strip().lower()

if response == 'edit':
    tag_description = input("   Enter custom description: ").strip()
elif response == 'n':
    tag_description = f"Experiment {next_tag}"

# Create annotated tag with description
subprocess.run([
    'git', 'tag', '-a', next_tag,
    '-m', tag_description
], check=True)

print(f"✅ Experiment {next_tag} complete and tagged!")
print(f"   Description: {tag_description}")
```

### Complete Workflow: DVC + Git + Commit Hash Tracking

Here's the full end-to-end workflow for tracking experiments with DVC versioning:

```python
#!/usr/bin/env python3
"""Complete experiment workflow with DVC tracking and commit hash recording."""
import subprocess

# 1. Run your experiment (as shown above)
# ... experiment code ...

# 2. Track database with DVC
subprocess.run(['dvc', 'add', 'benchmark.db'], check=True)
print("✅ Database tracked with DVC")

# 3. Add .dvc file to git (NOT the database itself!)
subprocess.run(['git', 'add', 'benchmark.db.dvc', '.dvc/.gitignore'], check=True)

# 4. Commit the .dvc file
commit_msg = f"experiment: {next_tag} results"
subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

# 5. Get the commit hash of this commit
db_commit_hash = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD']
).decode().strip()

# 6. Get data version (commit hash when data.dvc was last modified)
try:
    data_commit_hash = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()
except:
    data_commit_hash = None

# 7. Record commit hashes in a tracking file or database
# Option A: Add to metrics table
tracker_metrics.log({
    'experiment_version': next_tag,
    'db_commit': db_commit_hash[:7],  # Short hash
    'data_commit': data_commit_hash[:7] if data_commit_hash else None,
    'accuracy': 0.85,
    'model': 'bert-base'
})

# Option B: Save to tracking file
with open('experiment_log.txt', 'a') as f:
    f.write(f"{next_tag}|{db_commit_hash}|{data_commit_hash}|{tag_description}\n")

# 8. Create git tag with auto-generated description
subprocess.run([
    'git', 'tag', '-a', next_tag,
    '-m', tag_description
], check=True)

# 9. Push database to DVC remote storage
subprocess.run(['dvc', 'push'], check=True)

# 10. Push git commits and tags
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
subprocess.run(['git', 'push', 'origin', next_tag], check=True)

print(f"""
✅ Experiment {next_tag} complete!

   Database commit: {db_commit_hash[:7]}
   Data version:    {data_commit_hash[:7] if data_commit_hash else 'N/A'}
   Description:     {tag_description}

   To reproduce later:
     git checkout {next_tag}
     dvc pull
""")
```

### Checking File Version Hashes

Useful commands for tracking version information:

```bash
# Get commit hash of current HEAD
git rev-parse HEAD

# Get short hash (7 characters)
git rev-parse --short HEAD

# Get commit hash when specific file was last changed
git log -n 1 --pretty=format:%H -- benchmark.db.dvc
git log -n 1 --pretty=format:%H -- data.dvc

# Get commit hash at specific tag
git rev-parse e0.0.3

# Check if file has changed since last commit
git diff --quiet benchmark.db.dvc && echo "No changes" || echo "Modified"

# Get DVC file hash (MD5 of actual data)
cat benchmark.db.dvc | grep md5
```

### Why Track Both Hashes?

1. **Code commit hash** (`git rev-parse HEAD`):
   - **Captures entire repository state** - code, .dvc files, everything
   - Guarantees exact reproducibility of the experiment
   - Used for `git checkout <hash>` to restore complete state

2. **Data commit hash** (`git log -n 1 ... -- data.dvc`):
   - **Captures only when data.dvc changed** - tracks data versions independently
   - Multiple experiments can share same data version (different code, same data)
   - Useful for identifying "same data, different model" comparisons

**Critical purpose:** By tracking data.dvc commit separately, you can detect if data changed between experiments without comparing entire repo state.

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
import sqlite3
from dataclasses import fields
from difflib import get_close_matches

def validate_schema_before_experiment(db_path, table, new_params_dataclass):
    """
    Compare current database schema with new experiment parameters.
    Detect potential issues: renamed params, similar names, missing columns.
    """
    # 1. Get current schema from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    conn.close()

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

                    # Execute rename in SQLite
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

### Recommended: Pre-Run Data Check

**Best practice:** Verify data is committed before running experiments to ensure reproducibility:

```python
def check_data_committed():
    """
    Optional safety check: warn if data.dvc has uncommitted changes.
    Helps prevent running experiments with unversioned data.
    """
    try:
        # Check if data.dvc has uncommitted changes
        result = subprocess.run(
            ['git', 'diff', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            print(
                "⚠️  WARNING: data.dvc has uncommitted changes!\n"
                "   Recommended: commit data changes before running experiments:\n"
                "   1. dvc add data/\n"
                "   2. git add data.dvc\n"
                "   3. git commit -m 'data: updated dataset'\n"
                "\n"
                "   This ensures reproducibility - every experiment will have\n"
                "   a committed data version.\n"
            )
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                raise RuntimeError("Experiment cancelled by user")

        # Also check if data.dvc is staged but not committed
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            print(
                "⚠️  WARNING: data.dvc is staged but not committed!\n"
                "   Recommended: git commit -m 'data: updated dataset'\n"
            )

        print("✅ Data is committed - good for reproducibility")
        return True
    except FileNotFoundError:
        print("ℹ️  data.dvc not found (may not be using DVC for data)")
        return True

# Optional: call this before running experiments
# check_data_committed()
```

**Why this helps:**
- Prevents "ghost experiments" with unversioned data
- Improves reproducibility - every result can be traced to exact data version
- Catches mistakes where data changed but wasn't committed

**Note:** This check is optional. For quick prototyping or debugging, you may want to skip it.

**Example use case:**
```python
# In your benchmark params
def get_versions():
    """Get both commit hashes for tracking."""
    code_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode().strip()

    data_commit = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()

    return code_commit[:7], data_commit[:7]

# Before running experiment
check_data_committed()

code_commit, data_commit = get_versions()
params = {
    'model': 'bert-base',
    'learning_rate': 0.0001,
    'code_commit': code_commit,      # Entire repo state
    'data_commit': data_commit       # Just data version
}
```

This lets you later query:
```sql
-- Find all experiments using the same data version
-- (useful to compare "same data, different hyperparameters")
SELECT experiment_version, model, accuracy, code_commit, data_commit
FROM metrics
WHERE data_commit = '3a2f1b'  -- Same data across all these experiments
ORDER BY accuracy DESC;

-- Find experiments where ONLY data changed (code stayed same)
SELECT experiment_version, accuracy, data_commit
FROM metrics
WHERE code_commit = 'a1b2c3d'  -- Same code
ORDER BY data_commit;
```

### Benefits:

1. **Clear separation**: `e*` tags for experiments, `v*` tags for code releases
2. **No conflicts**: Won't interfere with semantic versioning (v1.0.0, v1.1.0, etc.)
3. **Traceable results**: Each tag corresponds to exact code + data + results
4. **Easy comparison**: Query by tag to compare experiments
5. **Reproducible**: Checkout tag to get exact state

### View experiment history with descriptions:

```bash
# List only experiment tags (not version tags)
git tag -l 'e*' -n9

# Example output:
# e0.0.1          model=logistic | lr=0.001 | acc=0.8200 | data=3a2f1b
# e0.0.2          model=bert-base | lr=0.0001 | acc=0.8500 | data=3a2f1b
# e0.0.3          model=bert-base | lr=0.0001 | acc=0.8900 | data=7c4e2a (new dataset)
# e0.0.4          model=roberta | lr=0.00005 | acc=0.9100 | data=7c4e2a

# Meanwhile, your code versions remain separate:
git tag -l 'v*'
# v1.0.0          Initial release
# v1.1.0          Add new feature

# Show specific experiment details
git show e0.0.3
```

### Query by experiment version:

```sql
-- Compare experiments
SELECT
    experiment_version,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy
FROM db.predictions
WHERE experiment_version IN ('e0.0.1', 'e0.0.2')
GROUP BY experiment_version;

-- Get metrics for specific experiment
SELECT * FROM db.metrics WHERE experiment_version = 'e0.0.3';
```

### With DVC:

```bash
# After tagging, push database to DVC
dvc add benchmark.db
git add benchmark.db.dvc
git commit --amend --no-edit  # Add to same commit
dvc push

# Later, reproduce experiment
git checkout e0.0.3
dvc pull  # Gets exact database state
```

### Automation Script:

Create `run_experiment.py`:

```python
#!/usr/bin/env python
"""
Run a new experiment with automatic versioning.

Usage:
    python run_experiment.py --model bert-base --data dataset.csv
"""

import subprocess
import argparse
from your_benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    # Get next version
    latest = get_latest_tag()
    next_tag = increment_tag(latest)

    print(f"🚀 Running experiment {next_tag}")

    # Run benchmark
    results = run_benchmark(
        model=args.model,
        data=args.data,
        experiment_version=next_tag
    )

    # Generate informative tag description
    tag_desc = f"model={args.model} | data={args.data} | acc={results['accuracy']:.4f}"

    # Commit and tag
    subprocess.run(['git', 'add', 'benchmark.db'], check=True)
    subprocess.run(['git', 'commit', '-m', f'experiment: {next_tag} - {args.model}'], check=True)
    subprocess.run(['git', 'tag', '-a', next_tag, '-m', tag_desc], check=True)

    print(f"✅ Experiment complete!")
    print(f"   Tag: {next_tag}")
    print(f"   Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

**Trade-off:** Requires discipline to always use the workflow. Consider automation or git hooks.

---

## References

For deeper guidance, see:

- **`references/methodology.md`** - Core benchmarking principles and philosophy
- **`references/duckdb-analysis.md`** - DuckDB queries and analysis recipes

---

## Quick Start: Complete Minimal Example

**30-line end-to-end benchmark** showing two-table pattern:

```python
from dataclasses import dataclass
from xetrack import Tracker, Reader

# 1. Define params as frozen dataclass
@dataclass(frozen=True, slots=True)
class ModelParams:
    model: str
    threshold: float = 0.5

# 2. Single-execution function
def predict(item, params):
    prediction = item['x'] > params.threshold
    return {
        'input_id': item['id'],
        'prediction': prediction,
        'ground_truth': item['label'],
        'confidence': abs(item['x'] - params.threshold)
    }

# 3. Create trackers (two tables: predictions + metrics)
pred_tracker = Tracker(db='bench.db', engine='sqlite', cache='cache', table='predictions')
metrics_tracker = Tracker(db='bench.db', engine='sqlite', table='metrics')

# 4. Run benchmark
dataset = [{'id': 1, 'x': 0.3, 'label': False}, {'id': 2, 'x': 0.7, 'label': True}]
params = ModelParams(model='baseline', threshold=0.5)

results = [pred_tracker.track(predict, args=[item, params]) for item in dataset]

# 5. Calculate and log metrics
accuracy = sum(r['prediction'] == r['ground_truth'] for r in results) / len(results)
metrics_tracker.log({'model': params.model, 'threshold': params.threshold, 'accuracy': accuracy})

# 6. Analyze
print("Predictions:", Reader(db='bench.db', engine='sqlite', table='predictions').to_df())
print("Metrics:", Reader(db='bench.db', engine='sqlite', table='metrics').to_df())
```

**Why this example uses SQLite:** Safe for any scenario (single-process or multiprocessing).

Done! Now run the validation scripts and start analyzing.

---

## When NOT to Use This Workflow

Not every benchmark needs this level of rigor. **Skip this workflow** when:

**✅ Use simpler approach:**
- **Quick one-off comparison** (< 5 minutes to re-run everything)
- **Early prototyping phase** (speed of iteration > reproducibility)
- **Small-scale experiments** (< 100 datapoints, re-running is cheap)
- **Solo exploration** (no team coordination, throwaway analysis)

**🚫 Use this workflow:**
- **Results will be shared or published**
- **Experiment takes > 10 minutes to run**
- **Testing expensive APIs** (LLMs, cloud services, paid APIs)
- **Team collaboration** (multiple people running experiments)
- **Production benchmarks** (decisions depend on results)
- **Reproducibility matters** (research, A/B tests, audits)

**Key principle:** Match infrastructure complexity to problem complexity. Over-engineering wastes more time than it saves.

**Remember:**
> *"Storage is cheap; regret is expensive. The data you didn't save is the question you'll care about most."*

When in doubt, err on the side of tracking more.

---

## Alternative Tools & Platforms

While this skill focuses on xetrack + DuckDB/SQLite, consider these alternatives for different needs:

**Experiment Tracking Platforms:**
- **[Weights & Biases](https://wandb.ai/)**: Cloud-based, great for teams, rich visualizations
- **[MLflow](https://mlflow.org/)**: Open-source, self-hosted, model registry
- **[Neptune](https://neptune.ai/)**: Metadata store, experiment comparison
- **[Aim](https://github.com/aimhubio/aim)**: Lightweight, self-hosted, focuses on metrics

**Caching Libraries:**
- **[joblib](https://joblib.readthedocs.io/)**: Function result caching for complex objects
- **[diskcache](https://grantjenks.com/docs/diskcache/)**: Disk-based cache (what xetrack uses)
- **[GPTCache](https://github.com/zilliztech/GPTCache)**: LLM-specific semantic caching

**When to use alternatives:**
- Large teams → W&B or MLflow (better collaboration features)
- Complex pipelines → MLflow (pipeline tracking, model registry)
- LLM-heavy → GPTCache (semantic similarity caching)
- Need hosted solution → W&B or Neptune (no infrastructure management)

**When xetrack + DuckDB/SQLite wins:**
- Solo or small team
- Want full control and ownership of data
- Need SQL flexibility for custom analysis
- Prefer local-first, no cloud dependencies
- Git-based versioning workflow
