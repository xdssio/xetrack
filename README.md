<p align="center">
   <img src="https://raw.githubusercontent.com/xdssio/xetrack/main/docs/images/logo.jpg" alt="logo" width="400" />
</p>


<p align="center">
    <a href="https://github.com/xdssio/xetrack/actions/workflows/ci.yml">
        <img src="https://github.com/xdssio/xetrack/actions/workflows/ci.yml/badge.svg" alt="CI Status" />
    </a>
    <a href="https://pypi.org/project/xetrack/">
        <img src="https://img.shields.io/pypi/v/xetrack.svg" alt="PyPI version" />
    </a>
    <a href="https://pypi.org/project/xetrack/">
        <img src="https://img.shields.io/pypi/pyversions/xetrack.svg" alt="Python versions" />
    </a>
    <a href="https://github.com/xdssio/xetrack/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" />
    </a>
    <a href="https://github.com/xdssio/xetrack/issues">
        <img src="https://img.shields.io/github/issues/xdssio/xetrack.svg" alt="GitHub issues" />
    </a>
    <a href="https://github.com/xdssio/xetrack/network/members">
        <img src="https://img.shields.io/github/forks/xdssio/xetrack.svg" alt="GitHub forks" />
    </a>
    <a href="https://github.com/xdssio/xetrack/stargazers">
        <img src="https://img.shields.io/github/stars/xdssio/xetrack.svg" alt="GitHub stars" />
    </a>
</p>

# xetrack

Lightweight, local-first experiment tracker and benchmark store built on [SQLite](https://sqlite.org/index) and [duckdb](https://duckdb.org).


### Why xetrack Exists
Most experiment trackers — like Weights & Biases — rely on cloud servers...
xetrack is a lightweight package to track benchmarks, experiments, and monitor structured data.   
It is focused on simplicity and flexibility.
You create a "Tracker", and let it track benchmark results, model training and inference monitoring. later retrieve as pandas or connect to it directly as a database.


## Features

* Simple
* Embedded
* Fast
* Pandas-like
* SQL-like
* Object store with deduplication
* CLI for basic functions
* Multiprocessing reads and writes
* Loguru logs integration
* Experiment tracking
* Model monitoring

## Installation

```bash
pip install xetrack
pip install xetrack[duckdb] # to use duckdb as engine
pip install xetrack[assets] # to be able to use the assets manager to save objects
pip install xetrack[cache] # to enable function result caching
```

## Examples

**Complete examples for every feature** are available in the `examples/` directory:

```bash
# Run all examples
python examples/run_all.py

# Run individual examples
python examples/01_quickstart.py
python examples/02_track_functions.py
# ... etc
```

See [`examples/README.md`](examples/README.md) for full documentation of all 9+ examples.

## Quickstart

```python
from xetrack import Tracker

tracker = Tracker('database_db', 
                  params={'model': 'resnet18'}
                  )
tracker.log({"accuracy":0.9, "loss":0.1, "epoch":1}) # All you really need

tracker.latest
{'accuracy': 0.9, 'loss': 0.1, 'epoch': 1, 'model': 'resnet18', 'timestamp': '18-08-2023 11:02:35.162360',
 'track_id': 'cd8afc54-5992-4828-893d-a4cada28dba5'}


tracker.to_df(all=True)  # retrieve all the runs as dataframe
                    timestamp                              track_id     model  loss  epoch  accuracy
0  26-09-2023 12:17:00.342814  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      1       0.9
```

**Multiple experiment types**: Use different table names to organize different types of experiments in the same database.

```python
model_tracker = Tracker('experiments_db', table='model_experiments')
data_tracker = Tracker('experiments_db', table='data_experiments')
```

**Params** are values which are added to every future row:

```python
$ tracker.set_params({'model': 'resnet18', 'dataset': 'cifar10'})
$ tracker.log({"accuracy":0.9, "loss":0.1, "epoch":2})

{'accuracy': 0.9, 'loss': 0.1, 'epoch': 2, 'model': 'resnet18', 'dataset': 'cifar10', 
 'timestamp': '26-09-2023 12:18:40.151756', 'track_id': '398c985a-dc15-42da-88aa-6ac6cbf55794'}

```

You can also set a value to an entire run with *set_value* ("back in time"):

```python
tracker.set_value('test_accuracy', 0.9) # Only known at the end of the experiment
tracker.to_df()

                    timestamp                              track_id     model  loss  epoch  accuracy  dataset  test_accuracy
0  26-09-2023 12:17:00.342814  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      1       0.9      NaN            0.9
2  26-09-2023 12:18:40.151756  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      2       0.9  cifar10            0.9

```

## Track functions

You can track any function.

* The return value is logged before returned

```python
tracker = Tracker('database_db', 
    log_system_params=True, 
    log_network_params=True, 
    measurement_interval=0.1)
image = tracker.track(read_image, *args, **kwargs)
tracker.latest
{'result': 571084, 'name': 'read_image', 'time': 0.30797290802001953, 'error': '', 'disk_percent': 0.6,
 'p_memory_percent': 0.496507, 'cpu': 0.0, 'memory_percent': 32.874608, 'bytes_sent': 0.0078125,
 'bytes_recv': 0.583984375}
```

Or with a wrapper:

```python

@tracker.wrap(params={'name':'foofoo'})
def foo(a: int, b: str):
    return a + len(b)

result = foo(1, 'hello')
tracker.latest
{'function_name': 'foo', 'args': "[1, 'hello']", 'kwargs': '{}', 'error': '', 'function_time': 4.0531158447265625e-06, 
 'function_result': 6, 'name': 'foofoo', 'timestamp': '26-09-2023 12:21:02.200245', 'track_id': '398c985a-dc15-42da-88aa-6ac6cbf55794'}
```

### Automatic Dataclass and Pydantic BaseModel Unpacking

**NEW**: When tracking functions, xetrack automatically unpacks frozen dataclasses and Pydantic BaseModels into individual tracked fields with dot-notation prefixes.

This is especially useful for ML experiments where you have complex configuration objects:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"

@tracker.wrap()
def train_model(config: TrainingConfig):
    # Your training logic here
    return {"accuracy": 0.95, "loss": 0.05}

config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)
result = train_model(config)

# All config fields are automatically unpacked and tracked!
tracker.latest
{
    'function_name': 'train_model',
    'config_learning_rate': 0.001,      # ← Unpacked from dataclass
    'config_batch_size': 32,            # ← Unpacked from dataclass
    'config_epochs': 10,                # ← Unpacked from dataclass
    'config_optimizer': 'adam',         # ← Unpacked from dataclass
    'accuracy': 0.95,
    'loss': 0.05,
    'timestamp': '...',
    'track_id': '...'
}
```

**Works with multiple dataclasses:**

```python
@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    num_layers: int

@dataclass(frozen=True)
class DataConfig:
    dataset: str
    batch_size: int

def experiment(model_cfg: ModelConfig, data_cfg: DataConfig):
    return {"score": 0.92}

result = tracker.track(
    experiment,
    args=[
        ModelConfig(model_type="transformer", num_layers=12),
        DataConfig(dataset="cifar10", batch_size=64)
    ]
)

# Result includes: model_cfg_model_type, model_cfg_num_layers, 
#                  data_cfg_dataset, data_cfg_batch_size, score
```

**Also works with Pydantic BaseModel:**

```python
from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    experiment_name: str
    seed: int
    use_gpu: bool = True

@tracker.wrap()
def run_experiment(cfg: ExperimentConfig):
    return {"status": "completed"}

config = ExperimentConfig(experiment_name="exp_001", seed=42)
result = run_experiment(config)

# Automatically tracks: cfg.experiment_name, cfg.seed, cfg.use_gpu, status
```

**Benefits:**
- Clean function signatures (one config object instead of many parameters)
- All config values automatically tracked individually for easy filtering/analysis
- Works with both `tracker.track()` and `@tracker.wrap()` decorator
- Supports both frozen and non-frozen dataclasses
- Compatible with Pydantic BaseModel via `model_dump()`

## Track assets (Oriented for ML models)

Requirements: `pip install xetrack[assets]` (installs sqlitedict)

When you attempt to track a non primitive value which is not a list or a dict - xetrack saves it as assets with deduplication and log the object hash:

* Tips: If you plan to log the same object many times over, after the first time you log it, just insert the hash instead for future values to save time on encoding and hashing.

```python
$ tracker = Tracker('database_db', params={'model': 'logistic regression'})
$ lr = Logisticregression().fit(X_train, y_train)
$ tracker.log({'accuracy': float(lr.score(X_test, y_test)), 'lr': lr})
{'accuracy': 0.9777777777777777, 'lr': '53425a65a40a49f4',  # <-- this is the model hash
    'dataset': 'iris', 'model': 'logistic regression', 'timestamp': '2023-12-27 12:21:00.727834', 'track_id': 'wisteria-turkey-4392'}

$ model = tracker.get('53425a65a40a49f4') # retrieve an object
$ model.score(X_test, y_test)
0.9777777777777777
```

You can retrieve the model in CLI if you need only the model in production and mind carring the rest of the file

```bash
# bash
xt assets export database.db 53425a65a40a49f4 model.cloudpickle
```

```python
# python
import cloudpickle
with open("model_cloudpickle", 'rb') as f:
    model = cloudpickle.loads(f.read())
# LogisticRegression()
```

## Function Result Caching

Xetrack provides transparent disk-based caching for expensive function results using [diskcache](https://grantjenks.com/docs/diskcache/). When enabled, results are automatically cached based on function name, arguments, and keyword arguments.

### Installation

```bash
pip install xetrack[cache]
```

### Basic Usage

Simply provide a `cache` parameter with a directory path to enable automatic caching:

```python
from xetrack import Tracker

tracker = Tracker(db='track_db', cache='cache_dir')

def expensive_computation(x: int, y: int) -> int:
    """Simulate expensive computation"""
    return x ** y

# First call - executes function
result1 = tracker.track(expensive_computation, args=[2, 10])  # Computes 2^10 = 1024

# Second call with same args - returns cached result instantly
result2 = tracker.track(expensive_computation, args=[2, 10])  # Cache hit!

# Different args - executes function again
result3 = tracker.track(expensive_computation, args=[3, 10])  # Computes 3^10 = 59049

# Tracker params also affect cache keys
result4 = tracker.track(expensive_computation, args=[2, 10], params={"model": "v2"})  # Computes (different params)
result5 = tracker.track(expensive_computation, args=[2, 10], params={"model": "v2"})  # Cache hit!
```

### Cache Observability & Lineage Tracking

Cache behavior is tracked in the database with the `cache` field for full lineage tracking:

```python
from xetrack import Reader

df = Reader(db='track_db').to_df()
print(df[['function_name', 'function_time', 'cache', 'track_id']])
#   function_name           function_time  cache           track_id
# 0 expensive_computation   2.345          ""              abc123      # Computed (cache miss)
# 1 expensive_computation   0.000          "abc123"        def456      # Cache hit - traces back to abc123
# 2 expensive_computation   2.891          ""              ghi789      # Different args (computed)
```

The `cache` field provides lineage:
- **Empty string ("")**: Result was computed (cache miss or no cache)
- **track_id value**: Result came from cache (cache hit), references the original execution's track_id

### Reading Cache Directly

You can inspect cached values without re-running functions. Cache stores dicts with "result" and "cache" keys:

```python
from xetrack import Reader

# Read specific cached value by key
# Note: _generate_cache_key is a private method for advanced usage
cache_key = tracker._generate_cache_key(expensive_computation, [2, 10], {}, {})
if cache_key is not None:  # Will be None if any arg is unhashable
    cached_data = Reader.read_cache('cache_dir', cache_key)
    print(f"Result: {cached_data['result']}, Original execution: {cached_data['cache']}")
    # Result: 1024, Original execution: abc123

# Scan all cached entries
for key, cached_data in Reader.scan_cache('cache_dir'):
    print(f"{key}: result={cached_data['result']}, from={cached_data['cache']}")
```

### Use Cases

- **ML Model Inference**: Cache predictions for repeated inputs
- **Data Processing**: Cache expensive transformations or aggregations
- **API Calls**: Cache external API responses (with appropriate TTL considerations)
- **Scientific Computing**: Cache results of long-running simulations

### Force Cache Refresh

Use `cache_force=True` to skip the cache lookup and re-execute the function. The new result overwrites the existing cache entry:

```python
# Normal call — uses cache if available
result = tracker.track(expensive_computation, args=[2, 10])

# Force refresh — re-executes the function and overwrites the cache
result = tracker.track(expensive_computation, args=[2, 10], cache_force=True)

# Next normal call will use the refreshed cache entry
result = tracker.track(expensive_computation, args=[2, 10])  # Cache hit (from force-refreshed entry)
```

**When to use `cache_force`:**
- Model or data changed but function signature is the same
- Cached result might be stale or corrupted
- You want to re-run a specific computation without clearing the entire cache

### Delete Cache Entries

Remove all cache entries associated with a specific experiment run:

```python
from xetrack import Reader

# Delete all cache entries produced by a specific track_id
deleted = Reader.delete_cache_by_track_id('cache_dir', 'cool-name-1234')
print(f"Deleted {deleted} cache entries")
```

**CLI:**

```bash
# List all cache entries with their track_id lineage
xt cache ls cache_dir

# Delete cache entries by track_id
xt cache delete cache_dir cool-name-1234
```

### Important Notes

- **Cache keys** are generated from tuples of (function name, args, kwargs, **tracker params**)
- Different tracker params create separate cache entries (e.g., different model versions)
- Exceptions are **not cached** - failed calls will retry on next invocation
- Cache is persistent across Python sessions
- Lineage tracking: the `cache` field links cached results to their original execution via track_id

### Handling Objects in Cache Keys

Xetrack intelligently handles different types of arguments:

- **Primitives** (int, float, str, bool, bytes): Used as-is in cache keys
- **Hashable objects** (custom classes with `__hash__`): Uses `hash()` for consistent keys across runs
- **Unhashable objects** (list, dict, sets): **Caching skipped entirely** for that call (warning issued once per type)

```python
# Hashable custom objects work great
class Config:
    def __init__(self, value):
        self.value = value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return isinstance(other, Config) and self.value == other.value

# Cache hits work across different object instances with same hash
config1 = Config("production")
config2 = Config("production")
tracker.track(process, args=[config1])  # Computed, cached
tracker.track(process, args=[config2])  # Cache hit! (same hash)

# Unhashable objects skip caching entirely
tracker.track(process, args=[[1, 2, 3]])  # Computed, NOT cached (warning issued)
tracker.track(process, args=[[1, 2, 3]])  # Computed again, still NOT cached

# Make objects hashable to enable caching
class HashableList:
    def __init__(self, items):
        self.items = tuple(items)  # Use tuple for hashability
    def __hash__(self):
        return hash(self.items)
    def __eq__(self, other):
        return isinstance(other, HashableList) and self.items == other.items

tracker.track(process, args=[HashableList([1, 2, 3])])  # ✅ Cached!
```

### Using Frozen Dataclasses for Complex Configurations

**Recommended Pattern**: When your function has many parameters or complex configurations, use frozen dataclasses to enable caching. This is especially useful for ML experiments with multiple hyperparameters.

```python
from dataclasses import dataclass

# ✅ RECOMMENDED: frozen=True makes dataclass hashable automatically, slots efficient in memory
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str
    optimizer: str = "adam"

def train_model(config: TrainingConfig) -> dict:
    """Complex training function with many parameters"""
    # ... training logic ...
    return {"accuracy": 0.95, "loss": 0.05}

# Caching works seamlessly with frozen dataclasses
config1 = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10, model_name="bert")
result1 = tracker.track(train_model, args=[config1])  # Computed, cached

config2 = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10, model_name="bert")
result2 = tracker.track(train_model, args=[config2])  # Cache hit! (identical config)

# Different config computes again
config3 = TrainingConfig(learning_rate=0.002, batch_size=32, epochs=10, model_name="bert")
result3 = tracker.track(train_model, args=[config3])  # Computed (different learning_rate)
```

**Benefits:**
- Clean, readable function signatures (one config object instead of many parameters)
- Type safety with automatic validation
- Automatic hashability with `frozen=True` 
- Cache works across different object instances with identical values
- Easier to version and serialize configurations


### Tips and Tricks

* ```Tracker(Tracker.IN_MEMORY, logs_path='logs/') ``` Let you run only in memory - great for debugging or working with logs only

### Pandas-like

```python
print(tracker)
                                    _id                              track_id                 date    b    a  accuracy
0  48154ec7-1fe4-4896-ac66-89db54ddd12a  fd0bfe4f-7257-4ec3-8c6f-91fe8ae67d20  16-08-2023 00:21:46  2.0  1.0       NaN
1  8a43000a-03a4-4822-98f8-4df671c2d410  fd0bfe4f-7257-4ec3-8c6f-91fe8ae67d20  16-08-2023 00:24:21  NaN  NaN       1.0

tracker['accuracy'] # get accuracy column
tracker.to_df() # get pandas dataframe of current run

```

### SQL-like
You can filter the data using SQL-like syntax using [duckdb](https://duckdb.org/docs):
* The sqlite database is attached as **db** and the table is **events**. Assts are in the **assets** table.   
* To use the duckdb as backend, `pip install xetrack[duckdb]` (installs duckdb) and add the parameter engine="duckdb" to Tracker like so:

```python
Tracker(..., engine='duckdb')
```



#### Python
```python
tracker.conn.execute(f"SELECT * FROM db.events WHERE accuracy > 0.8").fetchall()
```

#### Duckdb CLI
* Install: `curl https://install.duckdb.org | sh`   
* If duckdb>=1.2.2, you can use [duckdb local ui](https://duckdb.org/2025/03/12/duckdb-ui.html)    

```bash
$ duckdb -ui
┌──────────────────────────────────────┐
│                result                │
│               varchar                │
├──────────────────────────────────────┤
│ UI started at http://localhost:4213/ │
└──────────────────────────────────────┘

D INSTALL sqlite; LOAD sqlite; ATTACH 'database_db' AS db (TYPE sqlite);
# navigate browser to http://localhost:4213/

# or run directly in terminal
D SELECT * FROM db.events;
┌────────────────────────────┬──────────────────┬──────────┬───────┬──────────┬────────┐
│         timestamp          │     track_id     │  model   │ epoch │ accuracy │  loss  │
│          varchar           │     varchar      │ varchar  │ int64 │  double  │ double │
├────────────────────────────┼──────────────────┼──────────┼───────┼──────────┼────────┤
│ 2023-12-27 11:25:59.244003 │ fierce-pudu-1649 │ resnet18 │     1 │      0.9 │    0.1 │
└────────────────────────────┴──────────────────┴──────────┴───────┴──────────┴────────┘
```

### Logger integration
This is very useful in an environment where you can use normal logs, and don't want to manage a separate logger or file.
On great use-case is **model monitoring**.

`logs_stdout=true` print to stdout every tracked event
`logs_path='logs'` writes logs to a file

```python
$ Tracker(db=Tracker.IN_MEMORY, logs_path='logs',logs_stdout=True).log({"accuracy":0.9})
2023-12-14 21:46:55.290 | TRACKING | xetrack.logging:log:176!📁!{"accuracy": 0.9, "timestamp": "2023-12-14 21:46:55.290098", "track_id": "marvellous-stork-4885"}

$ Reader.read_logs(path='logs')
   accuracy                   timestamp                track_id
0       0.9  2023-12-14 21:47:48.375258  unnatural-polecat-1380
```

### JSONL Logging for Data Synthesis and GenAI Datasets

JSONL (JSON Lines) format is ideal for building machine learning datasets, data synthesis, and GenAI training data. Each tracking event is written as a single-line JSON with structured metadata.

**Use Cases:**
- Building datasets for LLM fine-tuning
- Creating synthetic data for model training
- Structured data collection for data synthesis pipelines
- Easy integration with data processing tools

```python
# Enable JSONL logging
tracker = Tracker(
    db='database_db',
    jsonl='logs/data.jsonl'  # Write structured logs to JSONL
)

# Every log call writes structured JSON
tracker.log({"subject": "taxes", "prompt": "Help me with my taxes"})
tracker.log({"subject": "dance", "prompt": "Help me with my moves"})

# Read JSONL data into pandas DataFrame
df = Reader.read_jsonl('logs/data.jsonl')
print(df)
#                          timestamp     level  subject                    prompt                track_id
# 0  2024-01-15T10:30:00.123456+00:00  TRACKING    taxes  Help me with my taxes  ancient-falcon-1234
# 1  2024-01-15T10:35:00.234567+00:00  TRACKING    dance  Help me with my moves  ancient-falcon-1234

# Or use pandas directly (JSONL is standard format)
import pandas as pd
df = pd.read_json('logs/data.jsonl', lines=True)
```

**JSONL Entry Format:**
Each line contains flattened structured data suitable for ML pipelines:
```json
{"timestamp": "2024-01-15T10:30:00.123456+00:00", "level": "TRACKING", "accuracy": 0.95, "loss": 0.05, "epoch": 1, "model": "test-model", "track_id": "xyz-123"}
```

Note: Timestamp is in ISO 8601 format with timezone for maximum compatibility.

**Reading Data:**
```python
# From JSONL file
df = Reader.read_jsonl('logs/tracking.jsonl')

# From database (class method for convenience)
df = Reader.read_db('database_db', engine='sqlite', table='default')

# From database with filtering
df = Reader.read_db('database_db', track_id='specific-run-id', head=100)
```

## Analysis
To get the data of all runs in the database for analysis:   
Use this for further analysis and plotting.

* This works even while a another process is writing to the database.

```python
from xetrack import Reader
df = Reader('database_db').to_df() 
```

### Model Monitoring

Here is how we can save logs on any server and monitor them with xetrack:    
We want to print logs to a file or *stdout* to be captured normally.   
We save memory by not inserting the data to the database (even though it's fine).
Later we can read the logs and do fancy visualisation, online/offline analysis, build dashboards etc.

```python
tracker = Tracker(db=Tracker.SKIP_INSERT, logs_path='logs', logs_stdout=True)
tracker.logger.monitor("<dict or pandas DataFrame>") # -> write to logs in a structured way, consistent by schema, no database file needed


df = Reader.read_logs(path='logs')
"""
Run drift analysis and outlier detection on your logs: 
"""
```

### ML Tracking

```python
tracker.logger.experiment(<model evaluation and params>) # -> prettily write to logs

df = Reader.read_logs(path='logs')
"""
Run fancy visualisation, online/offline analysis, build dashboards etc.
"""
```

## CLI

For basic and repetative needs.

```bash
$ xt head database.db --n=2
|    | timestamp                  | track_id                 | model    |   accuracy | data   | params           |
|---:|:---------------------------|:-------------------------|:---------|-----------:|:-------|:-----------------|
|  0 | 2023-12-27 11:36:45.859668 | crouching-groundhog-5046 | xgboost  |        0.9 | mnist  | 1b5b2294fc521d12 |
|  1 | 2023-12-27 11:36:45.863888 | crouching-groundhog-5046 | xgboost  |        0.9 | mnist  | 1b5b2294fc521d12 |
...


$ xt tail database.db --n=1
|    | timestamp                  | track_id        | model    |   accuracy | data   | params           |
|---:|:---------------------------|:----------------|:---------|-----------:|:-------|:-----------------|
|  0 | 2023-12-27 11:37:30.627189 | ebony-loon-6720 | lightgbm |        0.9 | mnist  | 1b5b2294fc521d12 |

$ xt set database.db accuracy 0.8 --where-key params --where-value 1b5b2294fc521d12 --track-id ebony-loon-6720

$ xt delete database.db ebony-loon-6720 # delete experiments with a given track_id

# Cache management
$ xt cache ls cache_dir                         # list cache entries with track_id lineage
$ xt cache delete cache_dir cool-name-1234      # delete cache entries for a specific run

# run any other SQL in a oneliner
$ xt sql database.db "SELECT * FROM db.events;"

# retrieve a model (any object) which was saved into a file using cloudpickle
$ xt assets export database.db hash output 

# remove an object from the assets
$ xt assets delete database.db hash 

# If you have two databases, and you want to merge one to the other
# Only works with duckdb at this moment 
$ xt copy source.db target.db --assets/--no-assets --table=<table>


# Stats
$ xt stats describe database.db --columns=x,y,z

$ xt stats top/bottom database.db x # print the entry with the top/bottom result of a value

# bashplotlib (`pip install bashplotlib` is required)
$ xt plot hist database.db x
    ----------------------
    |    x histogram     |
    ----------------------

 225|      o
 200|     ooo
 175|     ooo
 150|     ooo
 125|     ooo
 100|    ooooo
  75|    ooooo
  50|    ooooo
  25|   ooooooo
   1| oooooooooo
     ----------

-----------------------------------
|             Summary             |
-----------------------------------
|        observations: 1000       |
|      min value: -56.605967      |
|         mean : 2.492545         |
|       max value: 75.185944      |
-----------------------------------
$ xt plot scatter database.db x y

```
# SQLite vs Duckdb
1. Dynamic Typing & Column Affinity
    * Quirk: SQLite columns have affinity (preference) rather than strict types.
    * Impact: "42" (str) will happily go into an INTEGER column without complaint.
    * Mitigation: As you’ve done, use explicit Python casting based on expected dtype.

2. Booleans Are Integers
    * Quirk: SQLite doesn’t have a native BOOLEAN type. True becomes 1, False becomes 0.
    * Impact: Any boolean stored/retrieved will behave like an integer.
    * Mitigation: Handle boolean ↔ integer conversion in code if you care about type fidelity.

3. NULLs Can Be Inserted into ANY Column
    * Quirk: Unless a column is explicitly declared NOT NULL, SQLite allows NULL in any field — even primary keys.
    * Impact: Can result in partially complete or duplicate-prone rows if you’re not strict.
    * Mitigation: Add NOT NULL constraints and enforce required fields at the application level.

# Tests for development
```bash
pip install pytest-testmon pytest
pytest -x -q -p no:warnings --testmon  tests
```

---

# Benchmark Skill for Claude Code

xetrack includes a comprehensive **benchmark skill** for Claude Code that guides you through rigorous ML/AI benchmarking experiments.

## What is the Benchmark Skill?

The benchmark skill is an AI agent guide that helps you:
- **Design experiments** following best practices (single-execution, caching, reproducibility)
- **Track predictions & metrics** with the two-table pattern
- **Validate results** for data leaks, duplicate executions, and missing params
- **Analyze with DuckDB** using powerful SQL queries
- **Version experiments** with git tags and DVC
- **Avoid common pitfalls** (multiprocessing issues, cache problems, etc.)

> The 7-phase workflow is **genuinely well-structured**. The "design end-to-start" principle and single-execution principle are real insights that save people from common mistakes [...] The two-table pattern [...] is a concrete, opinionated design that **eliminates decision paralysis** [...] 8+ pitfalls discovered through actual simulations — **this is rare and valuable**. Most skills are written from theory; yours was battle-tested with real databases [...] The engine decision matrix [...] with multiprocessing gotchas is **genuinely useful** — this is a pitfall that costs hours to debug [...] Validation scripts [...] are actionable — they produce real recommendations, not just data [...] Scripts are functional, not just documentation [...] The experiment explorer [...] is a serious tool — auto-detection of retrieval strategy [...] side-by-side diff, disposable worktrees for exploration [...] The model manager with the candidates pattern **solves a real organizational problem** [...] The artifact merger using DuckDB for schema-flexible merges is clever [...] The 14 use cases [...] are concrete and map directly to real workflow decisions [...] The workflow decision matrix is **the killer feature** — exactly the kind of decision that's hard to make and easy to get wrong [...] The merge vs rebase semantics for each artifact type is **genuinely novel**; nobody has codified this for ML experiments before [...] The two skills complement each other perfectly — one runs experiments, the other versions them [...] Safety checklists [...] prevent data loss [...] Deep DuckDB integration for analysis is a differentiator [...] Local-first philosophy means **zero infrastructure to start**.
>
> — Claude, on first review of the benchmark & git-versioning skills

## Installation

### Option 1: Install from Plugin Marketplace (Recommended)

The easiest way to install the benchmark skill is directly from the xetrack repository using Claude Code's plugin marketplace:

```bash
# In Claude Code, add the xetrack marketplace
/plugin marketplace add xdssio/xetrack

# Install the benchmark skill
/plugin install benchmark@xetrack
```

That's it! Claude Code will automatically download and configure the skill.

**Update to latest version:**
```bash
/plugin marketplace update
```

### Option 2: Manual Installation

```bash
# Clone the xetrack repository
git clone https://github.com/xdssio/xetrack.git

# Copy the benchmark skill to Claude's skills directory
cp -r xetrack/skills/benchmark ~/.claude/skills/benchmark

# Verify installation
ls ~/.claude/skills/benchmark/SKILL.md
```

## Usage with Claude Code

Once installed, simply ask Claude to help with benchmarking:

**Example prompts:**

```
"Help me benchmark 3 embedding models on my classification task"

"Set up a benchmark comparing prompt variations for my LLM classifier"

"I want to benchmark different sklearn models with hyperparameter search"

"Debug my benchmark - I'm getting inconsistent results"
```

Claude will automatically use the benchmark skill and guide you through:

0. **Phase 0**: Planning what to track (ideation)
1. **Phase 1**: Understanding your goals and designing the experiment
2. **Phase 2**: Building a robust single-execution function
3. **Phase 3**: Adding caching for efficiency
4. **Phase 4**: Parallelizing (if needed)
5. **Phase 5**: Running the full benchmark loop
6. **Phase 6**: Validating results for common pitfalls
7. **Phase 7**: Analyzing results with DuckDB

## Features

### Two-Table Pattern

The skill teaches the recommended pattern of storing data in two tables:

- **Predictions table**: Every single prediction/execution (detailed)
- **Metrics table**: Aggregated results per experiment (summary)

```python
# Predictions table - granular data
predictions_tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='predictions',
    cache='cache_dir'
)

# Metrics table - aggregated results
metrics_tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='metrics'
)
```

### Git Tag-Based Versioning

Automatic experiment versioning with git tags:

```python
# Skill helps you run experiments with versioned tags
# e0.0.1 → e0.0.2 → e0.0.3

# View experiment history:
git tag -l 'e*' -n9
# e0.0.1  model=logistic | lr=0.001 | acc=0.8200 | data=3a2f1b
# e0.0.2  model=bert-base | lr=0.0001 | acc=0.8500 | data=3a2f1b
# e0.0.3  model=bert-base | lr=0.0001 | acc=0.8900 | data=7c4e2a
```

### DVC Integration

Built-in guidance for data and database versioning with DVC:

```bash
# Skill recommends DVC for reproducibility
dvc add data/
dvc add benchmark.db

git add data.dvc benchmark.db.dvc
git commit -m "experiment: e0.0.3 results"
git tag -a e0.0.3 -m "model=bert-base | acc=0.8900"
```

### Validation Scripts

Helper scripts to catch common issues:

```bash
# Check for data leaks, duplicates, missing params
python skills/benchmark/scripts/validate_benchmark.py benchmark.db predictions

# Analyze cache effectiveness
python skills/benchmark/scripts/analyze_cache_hits.py benchmark.db predictions

# Export markdown summary
python skills/benchmark/scripts/export_summary.py benchmark.db predictions > RESULTS.md
```

### Common Pitfalls Documented

The skill warns you about:
- ⚠️ DuckDB + multiprocessing = database locks (use SQLite instead)
- ⚠️ System monitoring incompatible with multiprocessing
- ⚠️ Dataclass unpacking only works with `.track()`, not `.log()`
- ⚠️ Model objects can bloat database (use assets)
- ⚠️ Float parameters need rounding for consistent caching

## Example Templates

The skill includes complete examples for common scenarios:

```bash
# sklearn model comparison
python skills/benchmark/assets/sklearn_benchmark_template.py

# LLM finetuning simulation
python skills/benchmark/assets/llm_finetuning_template.py

# Load testing / throughput benchmark
python skills/benchmark/assets/throughput_benchmark_template.py
```

## Documentation

Full documentation is in the skill itself:

- **SKILL.md**: Complete workflow and guidance
- **references/methodology.md**: Core benchmarking principles
- **references/duckdb-analysis.md**: SQL query recipes
- **scripts/**: Helper validation and analysis scripts
- **assets/**: Complete example templates

## When to Use the Skill

**Use the benchmark skill when:**
- Comparing multiple models or hyperparameters
- Testing expensive APIs (LLMs, cloud services)
- Results will be shared or published
- Reproducibility is critical
- Running experiments that take > 10 minutes

**Skip for:**
- Quick one-off comparisons (< 5 minutes to rerun)
- Early prototyping (speed > reproducibility)
- Solo throwaway analysis

## Troubleshooting

**"Database is locked" errors with DuckDB:**
- **Cause**: DuckDB doesn't handle concurrent writes from multiple processes
- **Solution**: Switch to SQLite engine if using multiprocessing
- **Details**: See `references/build-and-cache.md` Pitfall 2 for full explanation

**Cache not working:**
- **Check installation**: Ensure `pip install xetrack[cache]` was run
- **Check dataclass**: Must be frozen: `@dataclass(frozen=True, slots=True)`
- **Float parameters**: Need rounding for consistent hashing (see `references/build-and-cache.md` Pitfall 6)
- **Verify cache directory**: Check that cache path is writable

**Import errors:**
- **xetrack not found**: Run `pip install xetrack`
- **DuckDB features**: Run `pip install xetrack[duckdb]`
- **Asset management**: Run `pip install xetrack[assets]`
- **Caching support**: Run `pip install xetrack[cache]`

**"Dataclass not unpacking" issues:**
- **Check method**: Auto-unpacking only works with `.track()`, not `.log()`
- **Verify frozen**: Dataclass must have `frozen=True`
- **See `references/build-and-cache.md`**: Pitfall 1 for detailed explanation

## Git Versioning Skill

The **git-versioning** skill is a companion to the benchmark skill. While the benchmark skill runs experiments, the git-versioning skill handles versioning, merging, and retrieval of experiment artifacts.

### When to Use

Use the git-versioning skill when you need to:
- Version experiments with git tags and DVC
- Merge or rebase experiment results across branches
- Promote models from candidates to production
- Set up parallel experiments with git worktrees
- Retrieve models or data from past experiments
- Compare historical experiments side by side

### Installation

```bash
# Plugin marketplace
/plugin install git-versioning@xetrack

# Manual
cp -r xetrack/skills/git-versioning ~/.claude/skills/git-versioning
```

### Core Concepts

**Workflow selection** — The skill helps you choose the right approach:

| Scenario | Workflow | DB Engine | Branching |
|----------|---------|-----------|-----------|
| Single experiment | Sequential | SQLite | Main branch |
| Param sweep, same code/data | Parallel | DuckDB | Main branch |
| Different code or data per exp | Worktree | SQLite | Branch per exp |

**Merge vs Rebase** — A novel decision framework for ML artifacts:
- **Databases**: Merge (append rows) vs Rebase (replace when schema changed)
- **Data files**: Merge (add samples) vs Rebase (preprocessing overhaul)
- **Models**: Merge (keep as candidate) vs Rebase (promote to production)

**Candidates pattern** — Keep models organized:
- `models/production/model.bin` — current best (DVC tracked)
- `models/candidates/` — runner-ups for A/B tests and ensembles

### Scripts

| Script | Purpose |
|--------|---------|
| `setup_worktree.sh` | Create worktree with shared DVC cache (prevents the #1 pitfall) |
| `experiment_explorer.py` | Browse, compare, and retrieve past experiments |
| `merge_artifacts.py` | DuckDB-powered merge/rebase for databases and parquet files |
| `version_tag.py` | Create annotated tags with metric descriptions |
| `model_manager.py` | Promote/prune models, manage candidates |

### Example Prompts

```
"Help me version my experiment and create a git tag"

"Set up parallel experiments using git worktrees"

"Merge results from my experiment branch back to main"

"Retrieve the model from experiment e0.2.0"

"Compare experiments e0.1.0 and e0.2.0 side by side"
```

### How the Skills Work Together

```
Benchmark Skill                    Git Versioning Skill
─────────────────                  ──────────────────────
Phase 0-3: Design & Build    →    (not needed yet)
Phase 4-5: Run experiments   →    Choose workflow (sequential/parallel/worktree)
Phase 6-7: Validate & Analyze →   Tag experiment, push artifacts
                                   Merge results, promote models
                                   Explore & compare past experiments
```

## Contributing

Found an issue or want to improve the skills? Please open an issue or PR!

The skills were developed by running real simulations and discovering pitfalls, so real-world feedback is valuable.
