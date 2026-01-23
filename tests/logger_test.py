from xetrack import Tracker, Reader
from tempfile import TemporaryDirectory
from xetrack.logging import Logger
import pandas as pd
import os


def test_loguru_helper_simple():
    logger = Logger()
    logger.info("test")
    logger.error("test")
    logger.debug("test")
    logger.warning("test")


def test_log_to_file():
    data = {
        "accuracy": 0.9,
        "data": "mnist",
        "params": {"lr": 0.01, "epochs": 10},
        "category": "a",
    }
    tempdir = TemporaryDirectory().name

    logger = Logger(logs_path=tempdir, stdout=True)
    logger = Logger(logs_path=tempdir, stdout=True)

    logger.info("test")
    logger.monitor(data)
    data["category"] = "b"
    logger.monitor([data])
    data["category"] = "c"
    logger.monitor(pd.DataFrame([data]))

    # Use a public method or create a helper method for testing instead of accessing protected method
    log_files = [f for f in logger.read_logs(tempdir)]
    assert len(log_files) > 0

    df = pd.DataFrame(logger.read_logs(tempdir))
    categories = set(df["category"].tolist())
    assert categories == {"a", "b", "c"}

    data["category"] = "d"
    logger.track(data)
    data["category"] = "e"
    logger.experiment(data)

    df = pd.DataFrame(logger.read_logs(tempdir))
    categories = set(df["category"].tolist())
    assert categories == {"a", "b", "c", "d", "e"}


def test_logs_to_file():
    """Test loguru logger"""
    tracker = Tracker(Tracker.IN_MEMORY, logs_path="logs", logs_stdout=False)
    tracker.log({"a": 2, "b": 2})

    tracker.track(lambda x: {"x": x * 2}, args=[2])  # type: ignore

    tracker = Tracker(Tracker.IN_MEMORY, logs_path="logs", logs_stdout=True)
    _ = tracker.log({"a": 1, "b": 2})

def test_multi_logggers():
    a = Tracker(Tracker.IN_MEMORY, logs_path="logs", logs_stdout=True)
    b = Tracker(Tracker.IN_MEMORY, logs_path="logs", logs_stdout=False)

    a.log({"a": 1, "b": 2})
    b.log({"a": 1, "b": 2})

    c = Tracker(Tracker.IN_MEMORY, logs_path="logs", logs_stdout=True)
    c.log({"a": 1, "b": 2})
    c.log({"a": 1, "b": 2})


def test_jsonl_logging():
    """Test JSONL logging functionality"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    jsonl_path = os.path.join(tempdir.name, "test.jsonl")

    # Create tracker with JSONL logging
    tracker = Tracker(
        db=db_path,
        jsonl=jsonl_path,
        params={'model': 'test-model'}
    )

    # Log some data
    tracker.log({"accuracy": 0.95, "loss": 0.05, "epoch": 1})
    tracker.log({"accuracy": 0.97, "loss": 0.03, "epoch": 2})
    tracker.log({"accuracy": 0.98, "loss": 0.02, "epoch": 3})

    # Verify JSONL file was created
    assert os.path.exists(jsonl_path), "JSONL file was not created"

    # Read JSONL file
    df = Reader.read_jsonl(jsonl_path)

    # Verify data integrity
    assert len(df) == 3, f"Expected 3 entries, got {len(df)}"
    assert 'timestamp' in df.columns, "Missing timestamp column"
    assert 'level' in df.columns, "Missing level column"
    assert 'accuracy' in df.columns, "Missing accuracy column"
    assert 'loss' in df.columns, "Missing loss column"
    assert 'epoch' in df.columns, "Missing epoch column"

    # Verify values
    assert df['accuracy'].tolist() == [0.95, 0.97, 0.98], "Accuracy values mismatch"
    assert df['loss'].tolist() == [0.05, 0.03, 0.02], "Loss values mismatch"
    assert df['epoch'].tolist() == [1, 2, 3], "Epoch values mismatch"

    # Verify all entries have TRACKING level
    assert all(df['level'] == 'TRACKING'), "Not all entries have TRACKING level"

    with open(jsonl_path, "r") as f:
        for line in f:
            assert "TRACKING" in line, "Not all entries have TRACKING level"
            assert "data" not in line, "Data should not be in the line"

    df = pd.read_json(jsonl_path, lines=True)  # validate that the jsonl file is valid
    assert len(df) == 3, f"Expected 3 entries, got {len(df)}"

    tempdir.cleanup()


def test_jsonl_with_logger():
    """Test JSONL logging with Logger class directly"""
    tempdir = TemporaryDirectory()
    jsonl_path = os.path.join(tempdir.name, "logger_test.jsonl")

    logger = Logger(jsonl=jsonl_path, stdout=False)

    # Log different types of data
    logger.monitor({"cpu": 50.5, "memory": 1024})
    logger.track({"step": 1, "value": 0.95})
    logger.experiment({"param1": "value1", "param2": 100})

    # Verify file exists and has content
    assert os.path.exists(jsonl_path), "JSONL file was not created"

    df = Reader.read_jsonl(jsonl_path)

    # Should have 3 entries (monitor, track, experiment)
    assert len(df) == 3, f"Expected 3 entries, got {len(df)}"

    # Verify different levels were logged
    levels = set(df['level'].tolist())
    assert 'MONITOR' in levels or 'TRACKING' in levels or 'EXPERIMENT' in levels

    tempdir.cleanup()


def test_reader_read_db_classmethod():
    """Test Reader.read_db() classmethod"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_readdb.db")

    # Create tracker and log data
    tracker = Tracker(db=db_path, params={'model': 'test'})
    tracker.log({"accuracy": 0.95, "epoch": 1})
    tracker.log({"accuracy": 0.97, "epoch": 2})

    # Test Reader.read_db() classmethod
    df = Reader.read_db(db_path, engine='sqlite')

    assert len(df) == 2, f"Expected 2 entries, got {len(df)}"
    assert 'accuracy' in df.columns
    assert 'epoch' in df.columns

    # Test with head parameter
    df_head = Reader.read_db(db_path, engine='sqlite', head=1)
    assert len(df_head) == 1, f"Expected 1 entry with head=1, got {len(df_head)}"

    # Test with tail parameter
    df_tail = Reader.read_db(db_path, engine='sqlite', tail=1)
    assert len(df_tail) == 1, f"Expected 1 entry with tail=1, got {len(df_tail)}"

    tempdir.cleanup()


def test_jsonl_filters_non_tracking_logs():
    """Test that JSONL only captures MONITOR/TRACKING/EXPERIMENT levels"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_filter.db")
    jsonl_path = os.path.join(tempdir.name, "test_filter.jsonl")

    tracker = Tracker(db=db_path, jsonl=jsonl_path)

    # Log tracking data (should appear in JSONL)
    tracker.log({"value": 1})

    # Log warnings/errors via logger (should NOT appear in JSONL)
    if tracker.logger:
        tracker.logger.warning("This is a warning")
        tracker.logger.error("This is an error")
        tracker.logger.info("This is info")

    # Log more tracking data
    tracker.log({"value": 2})

    # Read JSONL - should only have the 2 tracking logs, not the warning/error/info
    df = Reader.read_jsonl(jsonl_path)

    assert len(df) == 2, f"Expected only 2 tracking entries, got {len(df)}"
    assert all(df['level'].isin(['MONITOR', 'TRACKING', 'EXPERIMENT'])), \
        "Found non-tracking levels in JSONL"

    tempdir.cleanup()


def test_jsonl_timestamp_format():
    """Test that JSONL uses ISO 8601 timestamp format from loguru"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_timestamp.db")
    jsonl_path = os.path.join(tempdir.name, "test_timestamp.jsonl")

    tracker = Tracker(db=db_path, jsonl=jsonl_path)
    tracker.log({"value": 1})

    # Read JSONL and check timestamp format
    df = Reader.read_jsonl(jsonl_path)

    timestamp_str = df['timestamp'].iloc[0]

    # ISO 8601 format should contain 'T' (date-time separator)
    # and typically '+' or 'Z' for timezone
    assert 'T' in timestamp_str, \
        f"Timestamp should be ISO 8601 format with 'T', got: {timestamp_str}"

    # Verify it's loguru's timestamp, not tracker's custom format
    # Tracker's format: "2026-01-23 13:00:44.614170" (has space, no 'T')
    # Loguru's format: "2026-01-23T13:00:44.614170+01:00" (has 'T')
    # If we have 'T', it means it's ISO format from loguru

    tempdir.cleanup()
