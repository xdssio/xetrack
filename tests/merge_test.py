from tempfile import TemporaryDirectory
from xetrack import Tracker, copy
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_merge():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    target = Tracker(db2)
    source.track(a=1, b=2)
    target.track(a=1, c=2)
    copy(source=db1, target=db2)
    analyses = Tracker(db2)
    assert len(analyses.to_df(all=True)) == 2
