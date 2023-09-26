from xetrack import Tracker
import sys
from tempfile import NamedTemporaryFile


def test_loguru():
    """ Test loguru logger """
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level=15, format="{time} {level} {message}")
    logger.level("info", no=15, color="<green>", icon="🚀")
    tracker = Tracker(db=NamedTemporaryFile().name, logger=logger)
    tracker.log(a=1, b=2)
    tracker.track(lambda x: {'x': x * 2}, args=[2])


def test_logging():
    """ Test standard logging logger """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    tracker = Tracker(db=NamedTemporaryFile().name, logger=logger)
    tracker.log(a=1, b=2)
    tracker.track(lambda x: {'x': x * 2}, args=[2])
