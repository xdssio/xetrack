from xetrack import Tracker, Reader
from tempfile import TemporaryDirectory
import numpy as np


def test_bashplotlib_hist():
    tempdir = TemporaryDirectory()
    db: str = f'{tempdir.name}/db1.db'
    source = Tracker(db)
    y_data = np.random.normal(loc=1, scale=10, size=1000)
    x_data = np.random.normal(loc=2, scale=20, size=1000)
    for x, y in zip(x_data, y_data):
        _ = source.log({'x': float(x), 'y': float(y)})

    df = Reader(db).to_df()
