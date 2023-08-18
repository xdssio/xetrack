from loguru import logger
from xetrack import Tracker
from tempfile import NamedTemporaryFile
import pytest

def test_loguru():
    tmp = NamedTemporaryFile()
    logger.add(tmp, format="{time} {level} {message}", level="INFO")
    tmp = NamedTemporaryFile()
    tracker = Tracker(db=tmp.name, verbose=True, logger=logger)
    tracker.log(a=1, b=2)
    tracker.track(lambda x: {'x': x * 2}, args=[2])
