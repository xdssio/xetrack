from tempfile import TemporaryDirectory
from xetrack import Tracker, copy
import pandas as pd
import time
import shutil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_merge():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    target = Tracker(db2)
    source.log({"a": 1, "b": 2})
    target.log({"a": 1, "c": 2})
    copy(source=db1, target=db2)
    analyses = Tracker(db2)
    assert len(analyses.to_df(all=True)) == 2


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
