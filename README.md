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
tracker.track(accuracy=0.9, loss=0.1, epoch=1)
tracker.to_df(all=True)
                                    _id                              track_id                 date     model  loss  epoch  accuracy
0  3adfc3ca-e7a0-4955-a7c7-2dae1a2552e4  e5324c93-25b1-4344-a1e1-0b47e95761aa  16-08-2023 00:43:38  resnet18   0.1      1       0.9

```
Params are values which are added to every feature row:

```python
tracker.set_params({'model': 'resnet18', 'dataset': 'cifar10'})
```

You can also set a value to an entire run with *set_value*:

```python
tracker.set_value('test_accuracy', 0.9)
```
## Track functions
You can track any function.
* The function must return a dictionary or None
```python
tracker = Tracker('database.db', log_system_params=True, log_network_params=True, measurement_interval=0.1)
tracker.track_function(read_image, *args, **kwargs)
{'result': 571084, 'name': 'read_image', 'time': 0.30797290802001953, 'error': '', 'disk_percent': 0.6, 'p_memory_percent': 0.496507, 'cpu': 0.0, 'memory_percent': 32.874608, 'bytes_sent': 0.0078125, 'bytes_recv': 0.583984375}
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
```python
tracker.conn.execute(f"SELECT * FROM {tracker.table_name} WHERE accuracy > 0.8").fetchall()
```

## Analysis
To get the data of all runs in the database for analysis:   
Use this for further analysis and plotting.

```python
# retrieve all runs data
df = Tracker('database.db').to_df(all=True) 
```
### Export
```python
tracker.to_csv('tracker.csv', all=False)
tracker.to_parquet('tracker.parquet', all=True)
```
## Merge two databases
If you have two databases, and you want to merge them into one, you can use the copy function:
```bash
python -c 'from xetrack import copy; copy(source="db1.db", target="db2.db")'
```