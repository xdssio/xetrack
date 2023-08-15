# Xetrack
Xetrack is a lightweight pacakge to track experiments and benchmarks data using duckdb.
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

tracker = Tracker('database.db')
tracker.track(accuracy=0.9, loss=0.1, epoch=1, model="resnet18")
print(tracker)
                                   _id                              track_id                 date  epoch  accuracy     model  loss
0  6b0f676e-15c0-4024-960b-e3911b3c7f15  ad287aaf-6704-474c-96a9-f748ddfed19b  15-08-2023 13:19:57     1       0.9  resnet18   0.1    

tracker = Tracker('database.db', params={'model': 'resnet34'}, verbose=False)
tracker.track(accuracy=0.8, loss=0.2, epoch=2)
print(tracker)
                                   _id                              track_id                 date  epoch  accuracy     model  loss
0  11f73063-2488-497b-914f-4179093e3c18  c374374a-974f-4862-83c7-de152688c3e0  15-08-2023 13:25:17      2       0.8  resnet34   0.2

print(tracker.all())
                                   _id                              track_id                 date  epoch  accuracy     model  loss
0  6b0f676e-15c0-4024-960b-e3911b3c7f15  ad287aaf-6704-474c-96a9-f748ddfed19b  15-08-2023 13:19:57      1       0.9  resnet18   0.1
1  11f73063-2488-497b-914f-4179093e3c18  c374374a-974f-4862-83c7-de152688c3e0  15-08-2023 13:25:17      2       0.8  resnet34   0.2

# Update future tracks params
tracker.set_param('branch', 'experiment1')
tracker.track(accuracy=0.7, loss=0.3, epoch=3)
print(tracker)
                                   _id                              track_id                 date  epoch  accuracy     model  loss       branch
0  11f73063-2488-497b-914f-4179093e3c18  c374374a-974f-4862-83c7-de152688c3e0  15-08-2023 13:25:17      2       0.8  resnet34   0.2         None
1  0a8b1083-b710-4274-955e-ba67a10ff413  c374374a-974f-4862-83c7-de152688c3e0  15-08-2023 13:29:20      3       0.7  resnet34   0.3  experiment1

# Update tracks params in the past
tracker['model_path'] = '/path/to/model.pth'
print(tracker)
                                    _id                              track_id                 date  epoch  accuracy     model  loss       branch          model_path
0  e9b450ea-f0ae-4247-b155-dc99d7a705af  6de2cc93-bfcb-4d5f-ae27-beda333c2693  15-08-2023 13:31:50      2       0.8  resnet34   0.2         None  /path/to/model.pth
1  f0f37075-2e07-44ec-b5db-1df46d351d9b  6de2cc93-bfcb-4d5f-ae27-beda333c2693  15-08-2023 13:32:08      3       0.7  resnet34   0.3  experiment1  /path/to/model.pth


# Use this for further analysis and plotting
analysis = Tracker('database.db').all() # pandas dataframe

# Export to csv or parquet
tracker.to_csv('tracker.csv', all=False)
tracker.to_parquet('tracker.parquet', all=True)
```