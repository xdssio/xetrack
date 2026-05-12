"""Thin orjson wrapper matching stdlib json interface (str in, str out)."""
from __future__ import annotations

from typing import Any, Optional, Union

import orjson


def dumps(obj: Any, *, indent: Optional[int] = None) -> str:
    """Serialize *obj* to a JSON string (not bytes)."""
    opts = orjson.OPT_NON_STR_KEYS
    if indent is not None:
        opts |= orjson.OPT_INDENT_2  # orjson only supports 2-space indent
    return orjson.dumps(obj, option=opts).decode()


def loads(s: Union[str, bytes]) -> Any:
    """Deserialize a JSON string or bytes to a Python object."""
    return orjson.loads(s)
