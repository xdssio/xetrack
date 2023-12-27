from tempfile import TemporaryDirectory
from xetrack import Tracker, copy
import pandas as pd
import shutil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_copy():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    target = Tracker(db2)

    class Mock():
        def __init__(self, name: str) -> None:
            self.name = name

    result = source.log({"a": 1, "b": 2, 'mock': Mock('a')})
    result = source.log({"a": 1, "b": 2, 'mock': Mock('test')})
    target.log({"a": 1, "c": 2, 'other': Mock('c')})
    target.log({"a": 1, "c": 2, 'other': Mock('a')})
    copy(source=db1, target=db2)
    df = Tracker(db2).to_df(all=True)
    assert len(df) == 4
    assert len(list(source.assets.assets.keys())) == 2
    assert len(list(source.assets.counts.items())) == 2
    assert target.get(result['mock']).name == 'test'  # by hash
    assert len(list(target.assets.assets.keys())) == 3  # deduplication
    for column in ('mock', 'other'):
        for hash_value in df[column].dropna().tolist():
            assert target.get(hash_value).name


def test_merge_repetitions():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    source.log({"a": 1, "b": 1})

    shutil.copy(db1, db2)
    target = Tracker(db2)

    source.log({"a": 2, "c": 2})
    source.log({"a": 3, "c": 3})
    target.log({"a": 4, "c": 4})

    assert copy(source=db2, target=db1) == 1
    assert copy(source=db2, target=db1) == 0  # already copied
    assert copy(source=db1, target=db2) == 2
