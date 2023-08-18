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

## Installation
```bash
pip install xetrack
```

## Quickstart

```python
from xetrack import Tracker

tracker = Tracker('database.db',
                  params={'model': 'resnet18'},
                  verbose=False)
tracker.log(accuracy=0.9, loss=0.1, epoch=1)
tracker.last
{'accuracy': 0.9, 'loss': 0.1, 'epoch': 1, 'model': 'resnet18', 'timestamp': '18-08-2023 11:02:35.162360', 'track_id': 'cd8afc54-5992-4828-893d-a4cada28dba5'}

tracker.to_df(all=True) # all runs
                    timestamp                              track_id     model  accuracy  epoch  loss
0  18-08-2023 10:43:34.599687  961ae844-203f-4be2-ae3c-907afce3a1b0  resnet18       0.9      1   0.1

```
Params are values which are added to every future row:

```python
tracker.set_params({'model': 'resnet18', 'dataset': 'cifar10'})
```

You can also set a value to an entire run with *set_value* ("back in time"):

```python
tracker.set_value('test_accuracy', 0.9)
```

## Track functions
You can track any function.
* The function must return a dictionary or None

```python
tracker = Tracker('database.db', log_system_params=True, log_network_params=True, measurement_interval=0.1)
image = tracker.track(read_image, *args, **kwargs)
tracker.last
{'result': 571084, 'name': 'read_image', 'time': 0.30797290802001953, 'error': '', 'disk_percent': 0.6,
 'p_memory_percent': 0.496507, 'cpu': 0.0, 'memory_percent': 32.874608, 'bytes_sent': 0.0078125,
 'bytes_recv': 0.583984375}
```
Or with a wrapper:
```python

@tracker.wrap(name='foofoo')
def foo(a: int, b: str):
    return a + len(b)
{'function_result': 6, 'name': 'foofoo', 'time': 7.867813110351562e-06, 'error': '', 'args': "[1, 'hello']", 'kwargs': '{}', 'disk_percent': 0, 'p_memory_percent': 0, 'cpu': 0, 'memory_percent': 0, 'bytes_sent': 0.0, 'bytes_recv': 0.0, 'model': 'lightgbm', 'timestamp': '18-08-2023 10:59:26.011938', 'track_id': 'a6f99e21-dfd8-4056-98e5-46b2a76fab41'}
```


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

## Analysis
To get the data of all runs in the database for analysis:   
Use this for further analysis and plotting.
* This works even while a another process is writing to the database.

```python
from xetrack import Reader
df = Reader('database.db').to_df() 
```

## Merge two databases
If you have two databases, and you want to merge them into one, you can use the copy function:
```bash
python -c 'from xetrack import copy; copy(source="db1.db", target="db2.db")'
```