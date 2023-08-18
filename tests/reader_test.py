from xetrack import Tracker, Reader
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile
import multiprocessing as mp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_db(database):
    reader = Reader(database)
    assert len(reader.to_df()) == 1


def test_reader():
    tempfile = NamedTemporaryFile()
    database = tempfile.name
    tracker = Tracker(database)
    tracker.log(a=1, b=2)
    stats_process = mp.Process(target=read_db, args=(database,))
    stats_process.start()
    stats_process.join()
