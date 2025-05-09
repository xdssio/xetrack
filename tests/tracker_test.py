import pandas as pd
from xetrack import Tracker, Reader
from tempfile import NamedTemporaryFile, TemporaryDirectory

from xetrack.engine import SqliteEngine


def test_track_wrapper():
    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", params={
                      "model": 'lightgbm'}, reset=True)

    @tracker.wrap(params={'name': 'foo', 'function_time': 2})
    def foo(a: int, b: int):
        return a + b

    foo(1, 2)
    assert tracker.latest.get('name') == 'foo'


def test_track():
    database = NamedTemporaryFile().name

    tracker = Tracker(db=database, engine="sqlite", params={"model": 'lightgbm'})
    assert len(tracker) == 0

    data = {'accuracy': 0.9, 'data': "mnist",
            'params': {'lr': 0.01, 'epochs': 10}}
    results = tracker.log(data)
    assert len(results) == 6  # data size including the default params
    assert tracker.latest == results
    assert len(tracker) == len(tracker.to_df()) == len(
        tracker.to_df(all=True)) == 1
    tracker.set_params({"model": 'lightgbm', 'branch': 'main'})
    assert len(tracker.log(data)) == 7  # we added another default param

    assert tracker['model'].iloc[0] == 'lightgbm'
    assert len(tracker['model'].value_counts()) == 1

    tracker.set_value('model', 'xgboost')
    assert tracker['model'].iloc[0] == 'xgboost'
    assert len(tracker['model'].value_counts()) == 1

    for _ in range(10):
        tracker.log(data)
    assert tracker['model'].value_counts().to_dict(
    ) == {'xgboost': 2, 'lightgbm': 10}  # type: ignore

    assert len(tracker.head()) == len(tracker.tail()) != len(
        tracker.to_df()) == len(tracker)

    new_tracker = Tracker(db=database)
    new_tracker.log_batch([data] * 10)

    assert len(new_tracker) == 10

    assert len(new_tracker.to_df(all=True)) == 22


def test_track_batch():
    database = NamedTemporaryFile().name
    tracker = Tracker(db=database, engine="sqlite", params={"model": 'lightgbm'})
    n = 10
    tracker.log_batch([{'a': i} for i in range(n)])
    assert tracker.latest['a'] == 9
    assert len(tracker) == n
    assert len(Reader(database).to_df()) == n


def test_skip_insert():
    logs = TemporaryDirectory().name
    tracker = Tracker(db=Tracker.SKIP_INSERT, engine="sqlite", logs_path=logs)
    tracker.log({"a": 1, "b": 2})
    assert len(tracker.to_df()) == 0
    assert len(Reader.read_logs(logs)) == 1


def test_track_assets():

    class MLModel():
        def __init__(self, name):
            self.model = name
            self.func = lambda x: x * 2

        def predict(self, X: int) -> int:
            return self.func(X)

    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", logs_stdout=True)

    tracker.log({'model': MLModel('model')})
    tracker.log({'model': MLModel('model2')})

    df = tracker.to_df()
    model_hash = df['model'].iloc[0]
    assert tracker.get(model_hash).model == 'model'
    tracker.remove_asset(model_hash, 'model')
    assert tracker.get(model_hash) is None

    assert pd.isna(tracker.to_df()['model'].iloc[0])


def test_ignore_warnings():

    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", logs_stdout=True)
    assert tracker.warnings
    _ = tracker.log(data={})
    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", warnings=False)
    assert not tracker.warnings
    _ = tracker.log(data={})
    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", logs_stdout=False, warnings=True)
    assert not tracker.warnings


def test_git_info():
    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", logs_stdout=True, git_root='.')
    latest = tracker.log({'a': 1})
    assert latest.get('git_commit_hash')
    assert len(latest.get('git_commit_hash')) == 40

    tracker = Tracker(db=Tracker.IN_MEMORY, engine="sqlite", logs_stdout=True, git_root='..')


def test_tracker_context_manager():
    with SqliteEngine(Tracker.IN_MEMORY) as engine:
        df = engine.execute_sql("SELECT * FROM main.events")
    assert len(df) == 0
