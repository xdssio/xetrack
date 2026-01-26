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

## Quickstart

```python
from xetrack import Tracker

tracker = Tracker('database.db', 
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
model_tracker = Tracker('experiments.db', table='model_experiments')
data_tracker = Tracker('experiments.db', table='data_experiments')
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
tracker = Tracker('database.db', 
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

## Track assets (Oriented for ML models)

Requirements: `pip install xetrack[assets]` (installs sqlitedict)

When you attempt to track a non primitive value which is not a list or a dict - xetrack saves it as assets with deduplication and log the object hash:

* Tips: If you plan to log the same object many times over, after the first time you log it, just insert the hash instead for future values to save time on encoding and hashing.

```python
$ tracker = Tracker('database.db', params={'model': 'logistic regression'})
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
with open("model.cloudpickle", 'rb') as f:
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

tracker = Tracker(db='track.db', cache='cache_dir')

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

df = Reader(db='track.db').to_df()
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

D INSTALL sqlite; LOAD sqlite; ATTACH 'database.db' AS db (TYPE sqlite);
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
2023-12-14 21:46:55.290 | TRACKING | xetrack.logging:log:69!📁!{"a": 1, "b": 2, "timestamp": "2023-12-14 21:46:55.290098", "track_id": "marvellous-stork-4885"}

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
    db='database.db',
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
df = Reader.read_db('database.db', engine='sqlite', table='default')

# From database with filtering
df = Reader.read_db('database.db', track_id='specific-run-id', head=100)
```

## Analysis
To get the data of all runs in the database for analysis:   
Use this for further analysis and plotting.

* This works even while a another process is writing to the database.

```python
from xetrack import Reader
df = Reader('database.db').to_df() 
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
tracker.logger.experiemnt(<model evaluation and params>) # -> prettily write to logs

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

$ xet set accuracy 0.8 --where-key params --where-value 1b5b2294fc521d12 --track-id ebony-loon-6720

$ xt delete database.db ebony-loon-6720 # delete experiments with a given track_id

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
$ xt describe database.db --columns=x,y,z

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
