from xetrack._json import dumps, loads


def test_dumps_returns_str():
    result = dumps({"a": 1})
    assert isinstance(result, str)
    assert '"a"' in result


def test_loads_from_str():
    result = loads('{"a": 1}')
    assert result == {"a": 1}


def test_loads_from_bytes():
    result = loads(b'{"a": 1}')
    assert result == {"a": 1}


def test_dumps_with_indent():
    result = dumps({"a": 1}, indent=2)
    assert isinstance(result, str)
    assert "\n" in result


def test_roundtrip():
    data = {"key": "value", "num": 42, "nested": {"x": [1, 2, 3]}}
    assert loads(dumps(data)) == data


def test_dumps_indent_always_two_spaces():
    result = dumps({"a": {"b": 1}}, indent=4)
    assert result == '{\n  "a": {\n    "b": 1\n  }\n}'


def test_dumps_nan_becomes_null():
    result = loads(dumps({"loss": float("nan")}))
    assert result["loss"] is None


def test_dumps_datetime():
    import datetime
    result = dumps({"t": datetime.datetime(2024, 1, 1)})
    assert "2024-01-01" in result


def test_json_decode_error_importable():
    from xetrack._json import JSONDecodeError
    import pytest
    with pytest.raises(JSONDecodeError):
        loads("not valid json")
