"""DataFrame backend abstraction for xetrack.

Provides a thin layer that returns either pandas or polars DataFrames
based on what's installed. Pandas is preferred when both are available
for backward compatibility.

Usage:
    from xetrack._dataframe import get_backend, set_backend, cursor_to_dataframe

    # Auto-detection (pandas preferred)
    df = cursor_to_dataframe(cursor)

    # Force polars
    set_backend('polars')
    df = cursor_to_dataframe(cursor)
"""
from __future__ import annotations

from typing import Any, List, Optional


_BACKEND: Optional[str] = None


def _detect_backend() -> str:
    """Detect available DataFrame backend.

    Priority: pandas (if installed) > polars > error.

    Returns:
        'pandas' or 'polars'

    Raises:
        ImportError: If neither pandas nor polars is installed.
    """
    try:
        import pandas  # noqa: F401
        return "pandas"
    except ImportError:
        pass
    try:
        import polars  # noqa: F401
        return "polars"
    except ImportError:
        raise ImportError(
            "Either pandas or polars must be installed. "
            "Install with: pip install pandas  OR  pip install polars"
        )


def get_backend() -> str:
    """Get the current DataFrame backend ('polars' or 'pandas')."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _detect_backend()
    return _BACKEND


def set_backend(backend: str) -> None:
    """Override the DataFrame backend.

    Args:
        backend: 'polars', 'pandas', or 'auto' to re-detect.
    """
    global _BACKEND
    if backend == "auto":
        _BACKEND = _detect_backend()
    elif backend in ("polars", "pandas"):
        _BACKEND = backend
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'polars', 'pandas', or 'auto'.")


def cursor_to_dataframe(cursor: Any) -> Any:
    """Convert a database cursor result to a DataFrame.

    Handles DuckDB cursors (which have .df()/.pl()) and SQLite cursors
    (which have .description and .fetchall()).

    Args:
        cursor: A database cursor or DuckDB result object.

    Returns:
        A pandas or polars DataFrame depending on the configured backend.
    """
    backend = get_backend()

    if backend == "polars":
        import polars as pl
        if hasattr(cursor, 'pl'):
            return cursor.pl()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        if not rows:
            return pl.DataFrame(schema={col: pl.Utf8 for col in columns})
        return pl.DataFrame(rows, schema=columns, orient="row")
    else:
        import pandas as pd
        if hasattr(cursor, 'df'):
            return cursor.df()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return pd.DataFrame.from_records(rows, columns=columns)


def dataframe_from_dicts(data: list[dict]) -> Any:
    """Create a DataFrame from a list of dictionaries.

    Args:
        data: List of row dictionaries.

    Returns:
        A pandas or polars DataFrame.
    """
    backend = get_backend()
    if backend == "polars":
        import polars as pl
        return pl.DataFrame(data) if data else pl.DataFrame()
    else:
        import pandas as pd
        return pd.DataFrame(data)


def empty_dataframe(columns: Optional[List[str]] = None) -> Any:
    """Create an empty DataFrame.

    Args:
        columns: Optional column names.

    Returns:
        An empty pandas or polars DataFrame.
    """
    backend = get_backend()
    if backend == "polars":
        import polars as pl
        if columns:
            return pl.DataFrame(schema={col: pl.Utf8 for col in columns})
        return pl.DataFrame()
    else:
        import pandas as pd
        return pd.DataFrame(columns=columns)


def df_sort(df: Any, by: str | list[str]) -> Any:
    """Sort a DataFrame by column(s).

    Args:
        df: A pandas or polars DataFrame.
        by: Column name(s) to sort by.

    Returns:
        Sorted DataFrame.
    """
    backend = get_backend()
    if backend == "polars":
        return df.sort(by)
    return df.sort_values(by=by)


def df_to_dict_records(df: Any) -> list[dict]:
    """Convert a DataFrame to a list of dicts (record-oriented).

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        List of row dictionaries.
    """
    backend = get_backend()
    if backend == "polars":
        return df.to_dicts()
    return df.to_dict(orient='records')


def df_to_markdown(df: Any) -> str:
    """Convert a DataFrame to a markdown table string.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        Markdown-formatted table string.
    """
    backend = get_backend()
    if backend == "polars":
        from tabulate import tabulate
        return tabulate(df.to_dicts(), headers="keys", tablefmt="pipe")
    return df.to_markdown()


def df_to_csv(df: Any, path: str) -> None:
    """Write a DataFrame to a CSV file.

    Args:
        df: A pandas or polars DataFrame.
        path: Output file path.
    """
    backend = get_backend()
    if backend == "polars":
        df.write_csv(path)
    else:
        df.to_csv(path, index=False)


def df_to_parquet(df: Any, path: str) -> None:
    """Write a DataFrame to a Parquet file.

    Args:
        df: A pandas or polars DataFrame.
        path: Output file path.
    """
    backend = get_backend()
    if backend == "polars":
        df.write_parquet(path)
    else:
        df.to_parquet(path, index=False)


def df_is_empty(df: Any) -> bool:
    """Check if a DataFrame is empty.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        True if the DataFrame has no rows.
    """
    backend = get_backend()
    if backend == "polars":
        return len(df) == 0
    return df.empty


def df_columns(df: Any) -> list[str]:
    """Get column names from a DataFrame.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        List of column names.
    """
    backend = get_backend()
    if backend == "polars":
        return df.columns
    return df.columns.tolist()


def df_filter_eq(df: Any, column: str, value: Any) -> Any:
    """Filter DataFrame rows where column equals value.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.
        value: Value to match.

    Returns:
        Filtered DataFrame.
    """
    backend = get_backend()
    if backend == "polars":
        import polars as pl
        return df.filter(pl.col(column) == value)
    return df[df[column] == value]


def df_dropna_all(df: Any) -> Any:
    """Drop rows where all values are null.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        DataFrame with all-null rows removed.
    """
    backend = get_backend()
    if backend == "polars":
        import polars as pl
        return df.filter(~pl.all_horizontal(pl.all().is_null()))
    return df.dropna(how='all')


def df_column_is_float(df: Any, column: str) -> bool:
    """Check if a column has float dtype.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.

    Returns:
        True if the column dtype is float.
    """
    backend = get_backend()
    if backend == "polars":
        import polars as pl
        return df[column].dtype in (pl.Float32, pl.Float64)
    else:
        from pandas.api.types import is_float_dtype
        return is_float_dtype(df[column])


def df_unique_series(df: Any, column: str) -> Any:
    """Get unique values of a column as a Series.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.

    Returns:
        A pandas Series or polars Series with unique values.
    """
    backend = get_backend()
    if backend == "polars":
        return df[column].unique()
    else:
        import pandas as pd
        return pd.Series(df[column].unique(), name=column)


def df_describe(df: Any) -> Any:
    """Get descriptive statistics of a DataFrame.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        A DataFrame with descriptive statistics.
    """
    return df.describe()


def df_row_to_dict(df: Any, index: int) -> dict:
    """Get a single row as a dictionary.

    Args:
        df: A pandas or polars DataFrame.
        index: Row index.

    Returns:
        Dictionary of column names to values.
    """
    backend = get_backend()
    if backend == "polars":
        return df.row(index, named=True)
    return df.iloc[index].to_dict()


def df_column_max(df: Any, column: str) -> Any:
    """Get the maximum value in a column.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.

    Returns:
        The maximum value.
    """
    backend = get_backend()
    if backend == "polars":
        return df[column].max()
    return df[column].max()


def df_column_min(df: Any, column: str) -> Any:
    """Get the minimum value in a column.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.

    Returns:
        The minimum value.
    """
    backend = get_backend()
    if backend == "polars":
        return df[column].min()
    return df[column].min()


def df_select_columns(df: Any, columns: list[str]) -> Any:
    """Select specific columns from a DataFrame.

    Args:
        df: A pandas or polars DataFrame.
        columns: List of column names to select.

    Returns:
        DataFrame with only the selected columns.
    """
    backend = get_backend()
    if backend == "polars":
        return df.select(columns)
    return df[columns]
