"""Tests for the DataFrame backend abstraction layer.

Verifies that core xetrack operations work under both pandas and polars backends.
"""
from tempfile import NamedTemporaryFile

from xetrack import Tracker
from xetrack._dataframe import (
    get_backend, set_backend, PANDAS, POLARS,
    cursor_to_dataframe, dataframe_from_dicts, empty_dataframe,
    df_sort, df_to_dict_records, df_to_markdown, df_columns,
    df_is_empty, df_filter_eq, df_row_to_dict, df_column_max,
    df_column_min, df_describe, df_select_columns, df_dropna_all,
)


def test_backend_detection():
    """Default backend should be pandas when both are installed."""
    set_backend("auto")
    assert get_backend() == PANDAS


def test_set_backend_polars():
    set_backend(POLARS)
    assert get_backend() == POLARS
    set_backend("auto")


def test_set_backend_invalid():
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("invalid")


def test_tracker_log_and_read(df_backend):
    """Core tracker log + read works under both backends."""
    db = NamedTemporaryFile().name
    t = Tracker(db=db, params={"model": "test"})
    t.log({"accuracy": 0.9, "loss": 0.1})
    t.log({"accuracy": 0.95, "loss": 0.05})

    df = t.to_df()
    assert len(df) == 2

    records = df_to_dict_records(df)
    assert len(records) == 2
    assert all("accuracy" in r for r in records)


def test_tracker_head_tail(df_backend):
    db = NamedTemporaryFile().name
    t = Tracker(db=db, params={"model": "test"})
    for i in range(10):
        t.log({"step": i})

    head_df = t.head(3)
    assert len(head_df) == 3

    tail_df = t.tail(3)
    assert len(tail_df) == 3


def test_tracker_getitem(df_backend):
    db = NamedTemporaryFile().name
    t = Tracker(db=db, params={"model": "test"})
    t.log({"accuracy": 0.9})

    row = t[1]
    assert row is not None
    assert "accuracy" in row


def test_reader_to_df(df_backend):
    from xetrack.reader import Reader

    db = NamedTemporaryFile().name
    t = Tracker(db=db, params={"model": "lgb"})
    t.log({"accuracy": 0.9})

    reader = Reader(db)
    df = reader.to_df()
    assert len(df) == 1

    cols = df_columns(df)
    assert "accuracy" in cols
    assert "model" in cols


def test_reader_latest(df_backend):
    from xetrack.reader import Reader

    db = NamedTemporaryFile().name
    Tracker(db=db, params={"model": "a"}).log({"acc": 0.1})
    Tracker(db=db, params={"model": "b"}).log({"acc": 0.2})

    reader = Reader(db)
    latest = reader.latest()
    assert not df_is_empty(latest)


def test_dataframe_from_dicts(df_backend):
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df = dataframe_from_dicts(data)
    assert len(df) == 2
    assert df_columns(df) == ["a", "b"]


def test_empty_dataframe(df_backend):
    df = empty_dataframe(columns=["x", "y"])
    assert df_is_empty(df)
    assert "x" in df_columns(df)


def test_df_sort(df_backend):
    data = [{"x": 3}, {"x": 1}, {"x": 2}]
    df = dataframe_from_dicts(data)
    sorted_df = df_sort(df, "x")
    records = df_to_dict_records(sorted_df)
    assert records[0]["x"] == 1
    assert records[2]["x"] == 3


def test_df_to_markdown(df_backend):
    data = [{"a": 1, "b": 2}]
    df = dataframe_from_dicts(data)
    md = df_to_markdown(df)
    assert "a" in md
    assert "1" in md


def test_df_filter_eq(df_backend):
    data = [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]
    df = dataframe_from_dicts(data)
    filtered = df_filter_eq(df, "name", "alice")
    assert len(filtered) == 1


def test_df_row_to_dict(df_backend):
    data = [{"x": 10, "y": 20}]
    df = dataframe_from_dicts(data)
    row = df_row_to_dict(df, 0)
    assert row["x"] == 10
    assert row["y"] == 20


def test_df_column_max_min(df_backend):
    data = [{"v": 1}, {"v": 5}, {"v": 3}]
    df = dataframe_from_dicts(data)
    assert df_column_max(df, "v") == 5
    assert df_column_min(df, "v") == 1


def test_df_select_columns(df_backend):
    data = [{"a": 1, "b": 2, "c": 3}]
    df = dataframe_from_dicts(data)
    selected = df_select_columns(df, ["a", "c"])
    cols = df_columns(selected)
    assert "a" in cols
    assert "c" in cols
    assert "b" not in cols


def test_df_dropna_all(df_backend):
    data = [{"a": 1, "b": 2}, {"a": None, "b": None}]
    df = dataframe_from_dicts(data)
    cleaned = df_dropna_all(df)
    assert len(cleaned) == 1


def test_df_describe(df_backend):
    data = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    df = dataframe_from_dicts(data)
    desc = df_describe(df)
    assert not df_is_empty(desc)


def test_execute_sql(df_backend):
    db = NamedTemporaryFile().name
    t = Tracker(db=db, params={"model": "test"})
    t.log({"accuracy": 0.9})

    df = t.engine.execute_sql(f"SELECT * FROM {t.engine.table_name}")
    assert len(df) == 1
