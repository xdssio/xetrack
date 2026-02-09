# Build, Cache & Common Pitfalls

Phase 2 (Build), Phase 3 (Caching), and Common Pitfalls. Referenced from SKILL.md.

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
# API Reference: Tracker(db, engine='sqlite|duckdb', table='predictions', params=dict, cache=str)
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
