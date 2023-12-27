# Xetrack

xetrack is a lightweight package to track experiments benchmarks, and monitor stractured data using [duckdb](https://duckdb.org) and [sqlite](https://sqlite.org/index.html).   
It is focuesed on simplicity and flexability.

You create a "Tracker", and let it track data. You can retrive it later as pandas or connect to it as a database.   
Each instance of the tracker has a "track_id" which is a unique identifier for a single run.

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


tracker.to_df(all=True)  # Retrive all the runs as dataframe
                    timestamp                              track_id     model  loss  epoch  accuracy
0  26-09-2023 12:17:00.342814  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      1       0.9
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

When you attempt to track a non primitive value which is not a list or a dict - xetrack saves it as assets with deduplication and log the object hash:

* Tips: If you plan to log the same object many times over, after the first time you log it, just insert the hash instead for future values to save time on encoding and hashing.

```python
$ tracker = Tracker('database.db', params={'model': 'logistic regression'})
$ lr = Logisticregression().fit(X_train, y_train)
$ tracker.log({'accuracy': float(lr.score(X_test, y_test)), 'lr': lr})
{'accuracy': 0.9777777777777777, 'lr': '53425a65a40a49f4',  # <-- this is the model hash
    'dataset': 'iris', 'model': 'logistic regression', 'timestamp': '2023-12-27 12:21:00.727834', 'track_id': 'wisteria-turkey-4392'}

$ model = tracker.get('53425a65a40a49f4') # retrive an object
$ model.score(X_test, y_test)
0.9777777777777777
```

You can retrive the model in CLI if you need only the model in production and mind carring the rest of the file
```bash
# bash
xt get database.db 53425a65a40a49f4 model.cloudpickle
```

```python
# python
import cloudpickle
with open("model.cloudpickle", 'rb') as f:
    model = cloudpickle.loads(f.read())
# LogisticRegression()
```


### Tips and tricks

* ```Tracker(Tracker.IN_MEMORY) ``` Let you run only in memory

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

#### Python
```python
tracker.conn.execute(f"SELECT * FROM db.events WHERE accuracy > 0.8").fetchall()
```

#### Duckdb CLI
```bash
duckdb
D ATTACH 'database.db' AS db (TYPE sqlite);
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
$ Tracker(db=Tracker.IN_MEMROY, logs_path='logs',logs_stdout=True).log({"accuracy":0.9})
2023-12-14 21:46:55.290 | TRACKING | xetrack.logging:log:69!📁!{"a": 1, "b": 2, "timestamp": "2023-12-14 21:46:55.290098", "track_id": "marvellous-stork-4885"}

$ Reader.read_logs(path='logs')
   accuracy                   timestamp                track_id
0       0.9  2023-12-14 21:47:48.375258  unnatural-polecat-1380
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

### ML tracking

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

$ xt delete database.db ebony-loon-6720 # delete experiments wiht a given track_id

# run any other SQL in a oneliner
$ xt sql database.db "SELECT * FROM db.events;"

# retrive a model which was saved into a files as files using cloudpickle
$ xt get database.db model hash  output 

# If you have two databases, and you want to merge one to the other
$ xt copy source.db target.db 
```
