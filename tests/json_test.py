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
