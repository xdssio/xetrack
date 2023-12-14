# Xetrack
xetrack is a lightweight package to track experiments and benchmarks data using duckdb.
It looks and feels like pandas and is very easy to use.   

Each instance of the tracker has a "track_id" which is a unique identifier for a single run.

## Features
* Simple
* Embedded 
* Fast
* Pandas-like
* SQL-like
* Multiprocessing reads and writes
* Loguru text integration for file monitoring

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
tracker.log(accuracy=0.9, loss=0.1, epoch=1)
{'accuracy': 0.9, 'loss': 0.1, 'epoch': 1, 'model': 'resnet18', 'timestamp': '18-08-2023 11:02:35.162360',
 'track_id': 'cd8afc54-5992-4828-893d-a4cada28dba5'}

tracker.latest
{'accuracy': 0.9, 'loss': 0.1, 'epoch': 1, 'model': 'resnet18', 'timestamp': '18-08-2023 11:02:35.162360',
 'track_id': 'cd8afc54-5992-4828-893d-a4cada28dba5'}


tracker.to_df(all=True)  # as dataframe
                    timestamp                              track_id     model  loss  epoch  accuracy
0  26-09-2023 12:17:00.342814  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      1       0.9
1  26-09-2023 12:17:29.771021  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      2       0.9


```
Params are values which are added to every future row:

```python
tracker.set_params({'model': 'resnet18', 'dataset': 'cifar10'})
tracker.log(accuracy=0.9, loss=0.1, epoch=2)
{'accuracy': 0.9, 'loss': 0.1, 'epoch': 2, 'model': 'resnet18', 'dataset': 'cifar10', 
 'timestamp': '26-09-2023 12:18:40.151756', 'track_id': '398c985a-dc15-42da-88aa-6ac6cbf55794'}

```

You can also set a value to an entire run with *set_value* ("back in time"):

```python
tracker.set_value('test_accuracy', 0.9)
tracker.to_df()

                    timestamp                              track_id     model  loss  epoch  accuracy  dataset  test_accuracy
0  26-09-2023 12:17:00.342814  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      1       0.9      NaN            0.9
2  26-09-2023 12:18:40.151756  398c985a-dc15-42da-88aa-6ac6cbf55794  resnet18   0.1      2       0.9  cifar10            0.9

```

## Track functions
You can track any function.
* The return value is logged before returned

```python
tracker = Tracker('database.db', log_system_params=True, log_network_params=True, measurement_interval=0.1)
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
 'function_result': 6, 'name': 'foofoo', 'disk_percent': 0, 'p_memory_percent': 0, 'cpu': 0, 'memory_percent': 0, 
 'bytes_sent': 0.0, 'bytes_recv': 0.0, 'model': 'resnet18', 'dataset': 'cifar10', 'timestamp': '26-09-2023 12:21:02.200245', 
 'track_id': '398c985a-dc15-42da-88aa-6ac6cbf55794'}
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
* The sqlite database is attached as **db** and the table is **events**
```python
tracker.conn.execute(f"SELECT * FROM db.events WHERE accuracy > 0.8").fetchall()
```

### Logger integration
This is very useful in an environment where you can use normal logs, and don't want to manage a separate logger or file.   
On great use-case is **model monitoring**.

`pip install xetrack[loguru]`

`logs_stdout=true` print to stdout every tracked event
`logs_path='logs'` writes logs to a file

```python
$ Tracker(db=Tracker.IN_MEMROY, logs_path='logs',logs_stdout=True).log(accuracy=0.9)
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
LAter we can read the logs and do fancy visualisation, online/offline analysis, build dashboards etc.

```python
tracker = Tracker(db=Tracker.SKIP_INSERT, logs_path='logs',logs_stdout=True)
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

## Merge two databases
If you have two databases, and you want to merge them into one, you can use the copy function:
```bash
python -c 'from xetrack import copy; copy(source="db1.db", target="db2.db")'
```