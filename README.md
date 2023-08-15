# Xetrack
xetrack is a lightweight package to track experiments and benchmarks data using duckdb.
It looks and feels like pandas and is very easy to use.   
Each instance of the tracker has a "track_id" to filter by it later.

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

tracker = Tracker('database.db', params={'model': 'resnet18'}, verbose=False)
tracker.track(accuracy=0.9, loss=0.1, epoch=1)
tracker.run()
                                   _id                              track_id                 date  epoch  accuracy     model  loss
0  6b0f676e-15c0-4024-960b-e3911b3c7f15  ad287aaf-6704-474c-96a9-f748ddfed19b  15-08-2023 13:19:57     1       0.9  resnet18   0.1
```
Params are values which are added to every row and can be set with *set_params* method:
```python
tracker.set_params({'model': 'resnet18', 'dataset': 'cifar10'})
```
You can also set a value to an entire run with the *set_value* method:

```python
tracker.set_value('test_accuracy', 0.9)
```

### SQL-like
You can filter the data using SQL-like syntax:
```python
tracker.conn.execute('SELECT * FROM tracker WHERE accuracy > 0.8').fetchall()
```

## Analysis
To get the data of all runs in the database for analysis:   
Use this for further analysis and plotting.
```python
run_data = Tracker('database.db').run() # pandas dataframe
all_data = Tracker('database.db').all() # pandas dataframe

# Export to csv or parquet
tracker.to_csv('tracker.csv', all=False)
tracker.to_parquet('tracker.parquet', all=True)
```