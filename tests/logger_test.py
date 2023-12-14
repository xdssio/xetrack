from xetrack import Tracker
from tempfile import TemporaryDirectory
from xetrack.logging import Logger
import pandas as pd


def test_loguru_helper_simple():
    logger = Logger()
    logger.info("test")
    logger.error("test")
    logger.debug("test")
    logger.warning("test")


def test_log_to_file():
    data = {'accuracy': 0.9, 'data': "mnist",
            'params': {'lr': 0.01, 'epochs': 10}, 'category': 'a'}
    tempdir = TemporaryDirectory().name

    logger = Logger(logs_path=tempdir, stdout=True)

    logger.info("test")
    logger.monitor(data)
    data['category'] = 'b'
    logger.monitor([data])
    data['category'] = 'c'
    logger.monitor(pd.DataFrame([data]))
    assert len(list(logger._iter_logs(tempdir))) == 1

    df = pd.DataFrame(logger.read_logs(tempdir))
    assert set(df['category']) == {'a', 'b', 'c'}

    data['category'] = 'd'
    logger.track(data)
    data['category'] = 'e'
    logger.experiment(data)

    df = pd.DataFrame(logger.read_logs(tempdir))
    assert set(df['category']) == {'a', 'b', 'c', 'd', 'e'}


def test_logs_to_file():
    """ Test loguru logger """
    tracker = Tracker(Tracker.IN_MEMROY, logs_path='logs', logs_stdout=True)
    tracker.log(a=1, b=2)
    tracker.track(lambda x: {'x': x * 2}, args=[2])


def test_logs_to_file():
    """ Test loguru logger """
    tracker = Tracker(Tracker.IN_MEMROY, logs_path='logs', logs_stdout=True)
    tracker.log(a=1, b=2)
    tracker.track(lambda x: {'x': x * 2}, args=[2])
