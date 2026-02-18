---
name: xetrack
description: API reference and usage patterns for xetrack — lightweight experiment tracking with SQLite/DuckDB. Use when writing code that uses xetrack, when unsure about Tracker or Reader API, when using the xt CLI, when working with caching/diskcache, or when tracking ML experiments. Triggers on "xetrack", "Tracker", "Reader", "tracker.track", "tracker.log", "xt cli", "xetrack cache", "experiment tracking", "track function", "track metrics".
---

# xetrack API Reference

xetrack is a lightweight experiment tracking tool. It stores metrics, parameters, and assets in SQLite or DuckDB. This skill is the canonical API reference — use it to write correct xetrack code.

**DO NOT hallucinate APIs.** If a method isn't listed here, check the source before using it.

## Install

```bash
pip install xetrack                      # Core (SQLite)
pip install xetrack[duckdb]              # DuckDB engine
pip install xetrack[cache]               # diskcache for function caching
pip install xetrack[assets]              # sqlitedict for asset storage
pip install xetrack[duckdb,cache,assets] # All extras
```

## Imports

```python
from xetrack import Tracker, Reader, copy
```

---

## Tracker

The main interface for tracking experiments.

### Constructor

```python
Tracker(
    db: str = "track.db",               # Database path, or ":memory:" for in-memory
    params: Dict[str, Any] | None = None,# Default params attached to every event
    reset: bool = False,                 # Drop and recreate events table
    log_system_params: bool = False,     # Track CPU, RAM, disk usage
    log_network_params: bool = False,    # Track bytes sent/received
    raise_on_error: bool = True,         # Raise exceptions from tracked functions
    measurement_interval: float = 1,     # System stats interval (seconds)
    track_id: Optional[str] = None,      # Custom track ID (auto-generated if None)
    logs_path: Optional[str] = None,     # Directory for log files
    logs_file_format: Optional[str] = None, # Log file name format (e.g., "{time:YYYY-MM-DD}.log")
    logs_stdout: bool = False,           # Print logs to stdout
    jsonl: Optional[str] = None,         # Path to JSONL structured log file
    cache: Optional[str] = None,         # Path to diskcache directory (enables function caching)
    compress: bool = False,              # Compress database
    warnings: bool = True,               # Show warnings
    git_root: Optional[str] = None,      # Git root for commit hash tracking
    engine: Literal["duckdb", "sqlite"] = "sqlite",  # Database engine
    table: str = "default",              # Table name for multi-experiment support
)
```

### Properties

```python
tracker.db          # str — database path
tracker.conn        # database connection object
tracker.table_name  # str — resolved table name
tracker.columns     # list — column names in the events table
tracker.dtypes      # dict — column name -> type mapping
tracker.assets      # AssetsManager | None — asset storage (requires sqlitedict)
tracker.track_id    # str — current track ID (e.g., "cool-name-1234")
tracker.params      # dict — default params attached to every event
tracker.latest      # dict — last logged event data
```

### Core Methods

#### `tracker.log(data: dict) -> dict`

Log arbitrary key-value data as an event. Adds `track_id` and `timestamp` automatically. Creates columns dynamically.

```python
tracker.log({"accuracy": 0.95, "loss": 0.05, "epoch": 10})
tracker.log({"model": "bert", "dataset": "squad"})
```

**Returns:** The validated data dict (with track_id and timestamp added).

#### `tracker.track(func, params=None, args=None, kwargs=None, cache_force=False)`

Execute a function, log its execution metadata, and return the result. Automatically tracks `function_name`, `function_time`, `args`, `kwargs`, `error`, and the function's return value.

```python
def train(lr: float, epochs: int) -> dict:
    return {"accuracy": 0.95, "loss": 0.05}

# Basic tracking
result = tracker.track(train, args=[0.01, 10])

# With extra params (logged alongside function data)
result = tracker.track(train, args=[0.01, 10], params={"model": "bert"})

# With kwargs
result = tracker.track(train, kwargs={"lr": 0.01, "epochs": 10})

# Force cache refresh (skip cache read, rewrite entry)
result = tracker.track(train, args=[0.01, 10], cache_force=True)
```

**Return value handling:**
- If the function returns a `dict`, its keys are merged into the logged event as columns.
- If the function returns a primitive (int, float, str, etc.), it's stored in `function_result`.
- If the function returns a non-primitive, it's converted to string and stored in `function_result`.

**Dataclass argument unpacking:**
When arguments are `dataclass` or Pydantic `BaseModel` instances, their fields are automatically extracted as columns with `{param_name}_{field_name}` naming:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    lr: float
    batch_size: int

def train(config: Config) -> dict:
    return {"accuracy": 0.95}

# Logs: config_lr=0.01, config_batch_size=32, accuracy=0.95
tracker.track(train, args=[Config(lr=0.01, batch_size=32)])
```

#### `tracker.wrap(params=None)`

Decorator factory for tracking functions.

```python
@tracker.wrap(params={"experiment": "baseline"})
def predict(x):
    return x * 2

result = predict(5)  # Tracked automatically
```

#### `tracker.log_batch(data: List[dict]) -> dict`

Log multiple events at once (batch insert).

```python
tracker.log_batch([
    {"step": 1, "loss": 0.5},
    {"step": 2, "loss": 0.3},
    {"step": 3, "loss": 0.1},
])
```

### Parameter Management

```python
tracker.set_param("model", "bert")         # Set a param for all future events
tracker.set_params({"model": "bert", "version": "v2"})  # Set multiple
tracker.set_value("accuracy", 0.95)        # Update value for this track_id
tracker.set_where("label", "good", "accuracy", 0.95)  # Conditional update
```

### Query Methods

```python
tracker.to_df(all=False)       # DataFrame for this track_id (all=True for everything)
tracker.head(n=5)              # First n rows for this track_id
tracker.tail(n=5)              # Last n rows for this track_id
tracker.count_all()            # Total record count in table
tracker.count_run()            # Record count for this track_id
len(tracker)                   # Same as count_run()
tracker["accuracy"]            # Column values for this track_id (pd.Series)
tracker[1]                     # First row as dict
```

### Export

```python
tracker.to_csv("output.csv", all=False)
tracker.to_parquet("output.parquet", all=False)
```

### Assets

Requires `pip install xetrack[assets]`. Stores complex Python objects with hash-based deduplication.

```python
# Objects are auto-stored as assets when logged via tracker.log()
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
tracker.log({"model_obj": model})  # Stored as asset, hash reference in DB

# Retrieve
tracker.get("model_obj_hash")
```

### Static/Utility

```python
Tracker.generate_track_id()           # -> "cool-name-1234"
Tracker.IN_MEMORY                     # ":memory:" constant
Tracker.SKIP_INSERT                   # Skip DB insert (logging only)
```

---

## Caching (diskcache)

Requires `pip install xetrack[cache]` or `pip install diskcache`.

### How It Works

When `cache` path is provided to `Tracker`, `tracker.track()` automatically caches function results:

1. **Cache key** = `(module.func_name, args, kwargs, params)` — all must be hashable
2. **Cache hit** = skip function execution, log cached result with lineage reference
3. **Cache miss** = execute function, store result, log with empty cache field

### Cache Lineage in DB

The `cache` column in the DB tracks lineage:
- `""` (empty string) = this row was computed fresh
- `"cool-name-1234"` (a track_id) = this row was served from cache, originally computed by that track_id

### cache_force

Skip cache read and re-execute the function, overwriting the existing cache entry:

```python
# Normal (uses cache if available)
result = tracker.track(func, args=[1, 2])

# Force refresh (re-executes, overwrites cache)
result = tracker.track(func, args=[1, 2], cache_force=True)
```

### What Gets Cached

- Only successful executions (exceptions are NOT cached)
- Only calls with fully hashable arguments (primitives, frozen dataclasses, objects with `__hash__`)
- Unhashable args (lists, dicts, non-frozen dataclasses) silently skip caching

### Frozen Dataclasses for Caching

```python
@dataclass(frozen=True)  # frozen=True makes it hashable -> cacheable
class Config:
    model: str
    lr: float

# Different Config values = different cache entries
tracker.track(train, args=[Config("bert", 0.01)])    # Cached separately
tracker.track(train, args=[Config("gpt", 0.01)])     # Cached separately
tracker.track(train, args=[Config("bert", 0.01)])    # Cache hit!
```

---

## Reader

Read-only interface for accessing tracked data.

### Constructor

```python
Reader(
    db: str,                                          # Database path
    engine: Literal["duckdb", "sqlite"] = "sqlite",   # Database engine
    table: str = "default",                            # Table name
)
```

### Properties

```python
reader.db          # str
reader.conn        # database connection
reader.columns     # list
reader.dtypes      # dict
reader.assets      # AssetsManager | None
```

### Query Methods

```python
reader.to_df(track_id=None, head=None, tail=None)  # -> pd.DataFrame
reader.latest()                                      # -> pd.DataFrame (last track_id's data)
len(reader)                                          # Total record count
```

### Data Modification

```python
reader.delete_run("cool-name-1234")                           # Delete all events for track_id
reader.set_value("key", "value", track_id="cool-name-1234")   # Update value
reader.set_where("key", "value", "where_key", "where_val", track_id=None)
reader.remove_asset("hash", column=None, remove_keys=True)
```

### Class Methods — Database

```python
Reader.read_db(db, engine="sqlite", table="default", track_id=None, head=None, tail=None)
```

### Class Methods — Logs

```python
Reader.read_logs(path, limit=None)  # -> pd.DataFrame from log files
Reader.read_jsonl(path)             # -> pd.DataFrame from JSONL file
```

### Class Methods — Cache

```python
# Read a specific cache entry by key
Reader.read_cache(cache_path, key)  # -> dict {"result": ..., "cache": track_id} | None

# Iterate all cache entries
for key, cached_data in Reader.scan_cache(cache_path):
    print(cached_data["result"], cached_data["cache"])

# Delete all cache entries for a track_id
deleted_count = Reader.delete_cache_by_track_id(cache_path, "cool-name-1234")
```

---

## CLI (`xt`)

### Database Queries

```bash
xt head db.db --n=10                              # First 10 rows
xt tail db.db --n=10                              # Last 10 rows
xt columns db.db                                  # List columns
xt ls db.db                                       # List column names
xt ls db.db track_id --unique                     # Unique track IDs
xt sql db.db "SELECT * FROM \"default\" LIMIT 5"  # Raw SQL (quote 'default')
```

### Data Operations

```bash
xt set db.db key value --track-id=cool-name-1234
xt delete db.db cool-name-1234
xt copy source.db target.db --assets/--no-assets --table=default --table=other
```

### Cache Management

```bash
xt cache ls ./cache_dir                           # List cache entries with lineage
xt cache delete ./cache_dir cool-name-1234        # Delete cache by track_id
```

### Asset Management

```bash
xt assets ls db.db                                # List all assets
xt assets export db.db <hash> output.pkl          # Export asset to file
xt assets delete db.db <hash>                     # Delete asset
```

### Statistics

```bash
xt stats describe db.db --columns=accuracy,loss   # Describe columns
xt stats top db.db accuracy                       # Row with max accuracy
xt stats bottom db.db loss                        # Row with min loss
```

### Plotting (requires `pip install xetrack[bashplotlib]`)

```bash
xt plot hist db.db accuracy --bins=20
xt plot scatter db.db epoch loss
```

### Common Options

All DB commands support:
- `--engine sqlite|duckdb` (default: sqlite)
- `--table <name>` (default: "default")
- `--json` (head/tail only — output as JSON)

---

## copy()

Copy data between databases (DuckDB only).

```python
from xetrack import copy

# Copy default table with assets
copy("source.db", "target.db")

# Copy specific tables
copy("source.db", "target.db", tables=["predictions", "metrics"])

# Skip assets
copy("source.db", "target.db", assets=False)
```

---

## Engine Selection Guide

| Feature | SQLite | DuckDB |
|---|---|---|
| Default | Yes | No (`pip install xetrack[duckdb]`) |
| Concurrent writes | multiprocessing.Pool | ThreadPoolExecutor |
| Table name in SQL | `"default"` (quoted) | `db.default` |
| Best for | Local dev, single process | Analytics, parallel I/O |
| copy() support | No | Yes |

---

## Common Patterns

### Multi-Table Tracking

```python
pred_tracker = Tracker(db="exp.db", table="predictions", cache="cache")
metrics_tracker = Tracker(db="exp.db", table="metrics")

# Track predictions per-item
for item in dataset:
    pred_tracker.track(predict, args=[item, params])

# Log aggregate metrics
metrics_tracker.log({"accuracy": 0.95, "model": "bert"})
```

### JSONL Logging (for data synthesis pipelines)

```python
tracker = Tracker(db=Tracker.SKIP_INSERT, jsonl="output.jsonl", logs_stdout=True)
tracker.log({"prompt": "...", "response": "...", "label": 1})

# Read back
df = Reader.read_jsonl("output.jsonl")
```

### Git Commit Tracking

```python
tracker = Tracker(db="exp.db", git_root=".")
# Automatically adds git_commit column to every event
```

### System Resource Monitoring

```python
tracker = Tracker(db="exp.db", log_system_params=True, log_network_params=True)
result = tracker.track(expensive_func, args=[data])
# Logs: cpu_percent, ram_percent, p_memory_percent, disk_percent, bytes_sent, bytes_recv
```

---

## Gotchas

1. **Table name quoting:** SQLite table `default` is a reserved keyword. In raw SQL use `"default"` (quoted). The engine handles this automatically.
2. **Unhashable args skip caching silently.** Lists, dicts, non-frozen dataclasses won't be cached. Use frozen dataclasses.
3. **Multiple Trackers on same DB:** Each Tracker instance has its own `_columns` set. Creating a second Tracker on the same DB may trigger "duplicate column" errors if both try to log. Use a single Tracker or separate track_ids on the same instance.
4. **Dynamic schema:** Columns are added on first use. If you rename a param, xetrack creates a NEW column — the old one stays with NULLs for new rows.
5. **Assets require sqlitedict:** `tracker.assets` is `None` unless `pip install xetrack[assets]`.
6. **DuckDB table names:** DuckDB uses `db.tablename` syntax. The engine handles this, but raw SQL queries need to use the right format.
