from typing import Optional
from xetrack import Reader
from tempfile import NamedTemporaryFile
from bashplotlib.histogram import plot_hist as bashplot_hist
from bashplotlib.scatterplot import plot_scatter as bashplot_scatter


def _to_csv(db: str, columns: list[str], path: str) -> str:
    from xetrack._dataframe import (
        df_column_is_float, df_columns, df_select_columns, df_to_dict_records,
    )

    df = Reader(db).to_df()
    for column in columns:
        if column not in df_columns(df):
            raise ValueError(f"Column {column} not in the database")
        if not df_column_is_float(df, column):
            raise ValueError(f"Column {column} is not a float type")

    # Write CSV without headers (required by bashplotlib)
    import csv
    selected = df_select_columns(df, columns)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in df_to_dict_records(selected):
            writer.writerow([row[c] for c in columns])
    return path


def plot_scatter(db: str,
                 x: str,
                 y: str,
                 pch: str = 'o',
                 size: Optional[int] = 20,
                 title: Optional[str] = None,
                 colour: str = 'white'):
    tmp = NamedTemporaryFile()
    temp_path = _to_csv(db, [x, y], tmp.name)
    if title is None:
        title = f'{x} vs {y}'
    if size is None:
        size = 20
    scatter_args = [('size', size), ('pch', pch),
                    ('colour', colour), ('title', title), ('xs', 0), ('ys', 0)]
    kwargs = {arg[0]: arg[1] for arg in scatter_args if arg[1] is not None}
    bashplot_scatter(temp_path, **kwargs)


def plot_hist(db: str,
              x: str,
              bins: Optional[int] = None,
              width: Optional[int] = None,
              height: Optional[int] = None,
              pch: str = 'o',
              colour: str = 'white',
              xlab: bool = False,
              summary: bool = True,
              title: Optional[str] = None):
    tmp = NamedTemporaryFile()
    temp_path = _to_csv(db, [x], tmp.name)
    if title is None:
        title = f'{x} histogram'
    hist_args = [('height', height), ('bincount', bins), ('binwidth', width), ('pch', pch),
                 ('colour', colour), ('title', title), ('xlab', xlab), ('showSummary', summary)]
    kwargs = {arg[0]: arg[1] for arg in hist_args if arg[1] is not None}
    bashplot_hist(temp_path, **kwargs)
