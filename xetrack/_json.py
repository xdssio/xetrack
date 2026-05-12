"""Thin orjson wrapper matching stdlib json interface (str in, str out)."""
from typing import Any, Optional, Union

import orjson
from orjson import JSONDecodeError

__all__ = ["dumps", "loads", "JSONDecodeError"]


def dumps(obj: Any, *, indent: Optional[int] = None) -> str:
    """Serialize *obj* to a JSON string (not bytes).

    Note: orjson only supports 2-space indentation. Any non-None ``indent``
    value produces 2-space indented output regardless of the value passed.
    """
    opts = orjson.OPT_NON_STR_KEYS
    if indent is not None:
        opts |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, option=opts).decode()


def loads(s: Union[str, bytes]) -> Any:
    """Deserialize a JSON string or bytes to a Python object."""
    return orjson.loads(s)
