"""DataFrame backend abstraction for xetrack.

Provides a thin layer that returns either pandas or polars DataFrames
based on what's installed. Pandas is preferred when both are available
for backward compatibility.

Creation functions (cursor_to_dataframe, dataframe_from_dicts, empty_dataframe)
use the configured backend. Operation functions (df_sort, df_filter_eq, etc.)
dispatch on the actual DataFrame type so they work correctly even if the
backend setting changes between creation and use.

Usage:
    from xetrack._dataframe import get_backend, set_backend, cursor_to_dataframe

    # Auto-detection (pandas preferred)
    df = cursor_to_dataframe(cursor)

    # Force polars
    set_backend(POLARS)
    df = cursor_to_dataframe(cursor)
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, List, Optional

PANDAS = "pandas"
POLARS = "polars"

_BACKEND: ContextVar[Optional[str]] = ContextVar("_BACKEND", default=None)


def _detect_backend() -> str:
    """Detect available DataFrame backend.

    Priority: pandas (if installed) > polars > error.

    Returns:
        PANDAS or POLARS

    Raises:
        ImportError: If neither pandas nor polars is installed.
    """
    try:
        import pandas  # noqa: F401
        return PANDAS
    except ImportError:
        pass
    try:
        import polars  # noqa: F401
        return POLARS
    except ImportError:
        raise ImportError(
            "Either pandas or polars must be installed. "
            "Install with: pip install pandas  OR  pip install polars"
        )


def get_backend() -> str:
    """Get the current DataFrame backend (PANDAS or POLARS)."""
    backend = _BACKEND.get()
    if backend is None:
        backend = _detect_backend()
        _BACKEND.set(backend)
    return backend


def set_backend(backend: str) -> None:
    """Override the DataFrame backend.

    Args:
        backend: POLARS, PANDAS, or 'auto' to re-detect.
    """
    if backend == "auto":
        _BACKEND.set(_detect_backend())
    elif backend in (POLARS, PANDAS):
        _BACKEND.set(backend)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'polars', 'pandas', or 'auto'.")


def _is_polars(df: Any) -> bool:
    """Check if a DataFrame/Series is a polars object by module name.

    Uses module inspection instead of isinstance to avoid importing polars
    when it may not be installed.

    Args:
        df: A DataFrame or Series object.

    Returns:
        True if the object is from the polars library.
    """
    return type(df).__module__.startswith("polars")


# ---------------------------------------------------------------------------
# Creation functions — use get_backend() to decide which library to use
# ---------------------------------------------------------------------------

def cursor_to_dataframe(cursor: Any) -> Any:
    """Convert a database cursor result to a DataFrame.

    Handles DuckDB cursors (which have .df()/.pl()) and SQLite cursors
    (which have .description and .fetchall()).

    Args:
        cursor: A database cursor or DuckDB result object.

    Returns:
        A pandas or polars DataFrame depending on the configured backend.
    """
    if get_backend() == POLARS:
        import polars as pl
        if hasattr(cursor, 'pl'):
            return cursor.pl()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        if not rows:
            return pl.DataFrame(schema=columns)
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
    if get_backend() == POLARS:
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
    if get_backend() == POLARS:
        import polars as pl
        if columns:
            return pl.DataFrame(schema=columns)
        return pl.DataFrame()
    else:
        import pandas as pd
        return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Operation functions — dispatch on actual DataFrame type via _is_polars()
# ---------------------------------------------------------------------------

def df_sort(df: Any, by: str | list[str]) -> Any:
    """Sort a DataFrame by column(s).

    Args:
        df: A pandas or polars DataFrame.
        by: Column name(s) to sort by.

    Returns:
        Sorted DataFrame.
    """
    if _is_polars(df):
        return df.sort(by)
    return df.sort_values(by=by)


def df_to_dict_records(df: Any) -> list[dict]:
    """Convert a DataFrame to a list of dicts (record-oriented).

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        List of row dictionaries.
    """
    if _is_polars(df):
        return df.to_dicts()
    return df.to_dict(orient='records')


def df_to_markdown(df: Any) -> str:
    """Convert a DataFrame or Series to a markdown table string.

    Args:
        df: A pandas/polars DataFrame or Series.

    Returns:
        Markdown-formatted table string.
    """
    if _is_polars(df):
        from tabulate import tabulate
        # Handle both DataFrame and Series
        if hasattr(df, 'to_dicts'):
            return tabulate(df.to_dicts(), headers="keys", tablefmt="pipe")
        # Polars Series — convert to DataFrame first
        return tabulate(df.to_frame().to_dicts(), headers="keys", tablefmt="pipe")
    return df.to_markdown(index=False)


def df_to_csv(df: Any, path: str) -> None:
    """Write a DataFrame to a CSV file.

    Args:
        df: A pandas or polars DataFrame.
        path: Output file path.
    """
    if _is_polars(df):
        df.write_csv(path)
    else:
        df.to_csv(path, index=False)


def df_to_parquet(df: Any, path: str) -> None:
    """Write a DataFrame to a Parquet file.

    Args:
        df: A pandas or polars DataFrame.
        path: Output file path.
    """
    if _is_polars(df):
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
    return len(df) == 0


def df_columns(df: Any) -> list[str]:
    """Get column names from a DataFrame.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        List of column names.
    """
    if _is_polars(df):
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
    if _is_polars(df):
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
    if _is_polars(df):
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
    if _is_polars(df):
        import polars as pl
        return df[column].dtype in (pl.Float32, pl.Float64)
    else:
        from pandas.api.types import is_float_dtype
        return is_float_dtype(df[column])


def df_unique_series(df: Any, column: str) -> Any:
    """Get unique values of a column as a Series with stable ordering.

    Args:
        df: A pandas or polars DataFrame.
        column: Column name.

    Returns:
        A pandas Series or polars Series with unique values.
    """
    if _is_polars(df):
        return df[column].unique(maintain_order=True)
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
    if _is_polars(df):
        return df.row(index, named=True)
    return df.iloc[index].to_dict()


def df_column_max(df: Any, column: str) -> Any:
    """Get the maximum value in a column."""
    return df[column].max()


def df_column_min(df: Any, column: str) -> Any:
    """Get the minimum value in a column."""
    return df[column].min()


def df_select_columns(df: Any, columns: list[str]) -> Any:
    """Select specific columns from a DataFrame.

    Args:
        df: A pandas or polars DataFrame.
        columns: List of column names to select.

    Returns:
        DataFrame with only the selected columns.
    """
    if _is_polars(df):
        return df.select(columns)
    return df[columns]
