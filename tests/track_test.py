from xetrack.tracker import Tracker
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


@pytest.fixture
def database():
    temp = NamedTemporaryFile()
    return temp.name


def test_track(database):
    tracker = Tracker(db=database, params={"model": 'lightgbm'}, reset=True)
    assert len(tracker) == 0

    data = {'accuracy': 0.9, 'data': "mnist", 'params': {'lr': 0.01, 'epochs': 10}}
    assert tracker.track(**data) == 7  # data size including the default params
    assert len(tracker) == len(tracker.to_df()) == len(tracker.to_df(all=True)) == 1

    tracker.set_params({"model": 'lightgbm', 'branch': 'main'})
    assert tracker.track(**data) == 8  # we added another default param

    assert tracker['model'].iloc[0] == 'lightgbm'
    assert len(tracker['model'].value_counts()) == 1

    tracker['model'] = 'xgboost'
    assert tracker['model'].iloc[0] == 'xgboost'
    assert len(tracker['model'].value_counts()) == 1

    for i in range(10):
        tracker.track(**data)
    assert tracker['model'].value_counts().to_dict() == {'xgboost': 2, 'lightgbm': 10}

    assert len(tracker.head()) == len(tracker.tail()) != len(tracker.run()) == len(tracker)

    new_tracker = Tracker(db=database)
    new_tracker.track_batch([data] * 10)

    assert len(new_tracker) == 10

    assert len(new_tracker.all()) == 22
    tracker._drop_table()
