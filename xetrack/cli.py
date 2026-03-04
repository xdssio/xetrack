from __future__ import annotations

from typing import Literal, Union, List
import typer
from xetrack import Reader, copy as copy_db
from json import dumps
from xetrack._dataframe import (
    df_columns, df_describe, df_filter_eq, df_column_max, df_column_min,
    df_select_columns, df_to_dict_records, df_to_markdown, df_unique_series,
)

app = typer.Typer()


def _parse_engine(engine: str) -> Literal['duckdb', 'sqlite']:
    """Normalize an engine string to a typed literal."""
    return 'duckdb' if engine == 'duckdb' else 'sqlite'


@app.command()
def head(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show"),
         json: bool = typer.Option(False, help="Prettify the output as json"),
         engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
         table: str = typer.Option("default", help="Name of the table to read from")):
    """Show the first lines of the database events table"""
    df = Reader(db, engine=_parse_engine(engine), table=table).to_df(head=n)
    result = dumps(df_to_dict_records(df), indent=4) if json else df_to_markdown(df)
    typer.echo(result)


@app.command()
def tail(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show"),
         json: bool = typer.Option(False, help="Prettify the output as json"),
         engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
         table: str = typer.Option("default", help="Name of the table to read from")):
    """Show the last lines of the database events table"""
    df = Reader(db, engine=_parse_engine(engine), table=table).to_df(tail=n)
    result = dumps(df_to_dict_records(df), indent=4) if json else df_to_markdown(df)
    typer.echo(result)


@app.command()
def columns(db: str = typer.Argument(help='path to database'),
            engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
            table: str = typer.Option("default", help="Name of the table to read from")):
    """List the columns in the database"""
    columns_list = df_columns(Reader(db, engine=_parse_engine(engine), table=table).to_df(head=1))
    typer.echo(f"Columns in {db} table '{table}':\n")
    typer.echo(columns_list)


@app.command()
def copy(source: str = typer.Argument(help='path to source database - will not be modified'),
         target: str = typer.Argument(help='path to target database'),
         assets: bool = typer.Option(True, help='copy the assets'),
         table: List[str] = typer.Option(None, help='table(s) to copy - can be specified multiple times (e.g., --table=default --table=other). If not provided, defaults to "default" table')):
    """Copy the data from one database to another. Supports copying specific tables or defaults to 'default' table."""
    tables = table if table else None
    total_copied = copy_db(source=source, target=target, assets=assets, tables=tables)
    typer.echo(f"Successfully copied {total_copied} events")


@app.command()
def delete(db: str = typer.Argument(help='path to database'),
           track_id: str = typer.Argument(help='track ID to delete'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
           table: str = typer.Option("default", help="Name of the table to read from")):
    """Delete a specific run from the database"""
    Reader(db, engine=_parse_engine(engine), table=table).delete_run(track_id)


@app.command()
def set(db: str = typer.Argument(help='path to database'),
        key: str = typer.Argument(help='key to set'),
        value: str = typer.Argument(help='value to set'),
        track_id: Union[str , None] = typer.Option(
            None, help='If provided, only events with this track id value would be changed.'),
        where_key: Union[str, None] = typer.Option(None, help='key to set'),
        where_value: Union[str, None] = typer.Option(None, help='value to set'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
        table: str = typer.Option("default", help="Name of the table to read from")):
    """
    Set a value in the database.

    This function sets a value in the specified database using the provided key and value.
    Optionally, a track ID can be specified to associate the value with a specific track.
    Additionally, a where clause can be specified to set the value only for records that match the given key-value pair.


    Examples:

        >>> xt set "path/to/database.db" "name" "John Doe" --track-id "cool-name" --where-key "age" --where-value 30
    """
    reader = Reader(db, engine=_parse_engine(engine), table=table)
    if where_key is not None:
        reader.set_where(key, value, where_key, where_value, track_id)
    else:
        reader.set_value(key, value, track_id=track_id)


@app.command()
def ls(db: str = typer.Argument(help='path to database'),
       column: Union[str, None] = typer.Argument(None, help='column values to list'),
       unique: bool = typer.Option(False, help='list unique values only'),
       track_id: str = typer.Option(None, help='track ID to list'),
       engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
       table: str = typer.Option("default", help="Name of the table to read from")):
    """List all the track IDs in the database"""
    df = Reader(db, engine=_parse_engine(engine), table=table).to_df(track_id=track_id)
    if column is None:
        typer.echo(df_columns(df))
        return
    values = df[column]
    if unique:
        values = df_unique_series(df, column)
    typer.echo(df_to_markdown(values))


@app.command()
def sql(db: str = typer.Argument(help='path to database'),
        query: str = typer.Argument(help='SQL query to execute'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
        table: str = typer.Option("default", help="Name of the table to read from")):
    """Execute a SQL query on the database and show the results as a markdown table - consider that the default table is 'db.default' or 'default'\n\nExample: xt sql path/to/database.db 'SELECT * FROM db.default LIMIT 5' """
    reader = Reader(db, engine=_parse_engine(engine), table=table)
    df = reader.engine.execute_sql(query)
    typer.echo(df_to_markdown(df))


cache_app = typer.Typer(short_help="Cache management commands")
app.add_typer(cache_app, name="cache")


@cache_app.command(name="delete")
def cache_delete(
    cache_path: str = typer.Argument(help="Path to the cache directory"),
    track_id: str = typer.Argument(help="Track ID whose cache entries should be deleted"),
):
    """Delete all cache entries associated with a specific track_id."""
    deleted = Reader.delete_cache_by_track_id(cache_path, track_id)
    typer.echo(f"Deleted {deleted} cache entries for track_id '{track_id}'")


@cache_app.command(name="ls")
def cache_ls(
    cache_path: str = typer.Argument(help="Path to the cache directory"),
):
    """List all cached entries with their track_id lineage."""
    entries = list(Reader.scan_cache(cache_path))
    if not entries:
        typer.echo("Cache is empty")
        return
    typer.echo(f"Cache entries in {cache_path}:\n")
    for key, cached_data in entries:
        track_id = cached_data.get("cache", "N/A") if isinstance(cached_data, dict) else "N/A"
        typer.echo(f"  track_id: {track_id}  key: {key}")


assets_app = typer.Typer(short_help="A helper for assets managments")
app.add_typer(assets_app, name="assets")


@assets_app.command()
def export(db: str = typer.Argument(help='path to database'),
           key: str = typer.Argument(help='hash of model to retrive'),
           output: str = typer.Argument(help='output to save the file to'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Export an asset from the database to a file."""
    tracker = Reader(db, engine=_parse_engine(engine))
    asset = tracker.assets.assets.get(key)
    if asset is None:
        typer.echo(f'Asset {key} not found in database {db}')
    else:
        with open(output, 'wb') as f:
            f.write(asset)  # type: ignore
        typer.echo(f'Asset {key} saved to {output}')


@assets_app.command(name="delete")
def assets_delete(db: str = typer.Argument(help='path to database'),
                  asset: str = typer.Argument(help='Asset hash to delete'),
                  column: str = typer.Option(
    None, help='Column to set to None'),
    remove_keys: bool = typer.Option(
    True, help='Remove the keys associated with the asset'),
    engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')
):
    """Delete a specific run from the database"""
    Reader(db, engine=_parse_engine(engine)).remove_asset(asset, column, remove_keys)


@assets_app.command(name="ls")
def assets_ls(db: str = typer.Argument(help='path to database'),
              engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """list all the assets in the database"""
    typer.echo(f"Assets in {db}:\n")
    typer.echo("Key: Hash")
    typer.echo("----")
    for key, hash in Reader(db, engine=_parse_engine(engine)).assets.keys.items():
        typer.echo(f"{key}: {hash}")


plot_app = typer.Typer(
    short_help="A helper for plots using bashplotlib (requires bashplotlib)")
app.add_typer(plot_app, name="plot")


@plot_app.command()
def hist(db: str = typer.Argument(help='path to database'),
         column: str = typer.Argument(help='column to plot'),
         bins: int = typer.Option(None, help='number of bins'),
         width: int = typer.Option(None, help='width of the plot'),
         height: int = typer.Option(None, help='height of the plot'),
         pch: str = typer.Option('o', help='point character'),
         colour: str = typer.Option('white', help='colour of the plot'),
         xlab: bool = typer.Option(False, help='show x label'),
         summary: bool = typer.Option(True, help='show summary'),
         title: str = typer.Option(None, help='title of the plot')):
    """Plot a histogram of the given column"""
    from xetrack.bashplotlib import plot_hist
    plot_hist(db, column, bins, width, height,
              pch, colour, xlab, summary, title)


@plot_app.command()
def scatter(db: str = typer.Argument(help='path to database'),
            x: str = typer.Argument(help='x column'),
            y: str = typer.Argument(help='y column'),
            pch: str = typer.Option('o', help='point character'),
            size: int = typer.Option(20, help='size of the points'),
            title: str = typer.Option(None, help='title of the plot'),
            colour: str = typer.Option('white', help='colour of the plot')):
    """Plot a scatter plot of the given columns"""
    from xetrack.bashplotlib import plot_scatter
    plot_scatter(db, x, y, pch, size, title, colour)


stats_app = typer.Typer(
    short_help="A helper for basic stats")
app.add_typer(stats_app, name="stats")


def _get_df(db: str, columns: str = "", engine: str = "sqlite"):
    """Get a DataFrame from the database, optionally selecting specific columns."""
    df = Reader(db, engine=_parse_engine(engine)).to_df()
    if columns != '':
        columns_list = columns.split(',')
        df = df_select_columns(df, columns_list)
    return df


@stats_app.command()
def describe(db: str = typer.Argument(help='path to database'),
             columns: str = typer.Option('', help='columns to describe - comma separated list (e.g. "col1,col2,col3")'),
             engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Describe columns in the database - use either numeric or categorical columns."""
    df = _get_df(db, columns, engine)
    typer.echo(df_to_markdown(df_describe(df)))


@stats_app.command()
def top(db: str = typer.Argument(help='path to database'),
        column: str = typer.Argument(help='Entry with best value'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Get the maximum value in the column"""
    df = _get_df(db, engine=engine)
    df = df_filter_eq(df, column, df_column_max(df, column))
    typer.echo(df_to_markdown(df))


@stats_app.command()
def bottom(db: str = typer.Argument(help='path to database'),
           column: str = typer.Argument(help='Entry with best value'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Get the minimum value in the column"""
    df = _get_df(db, engine=engine)
    df = df_filter_eq(df, column, df_column_min(df, column))
    typer.echo(df_to_markdown(df))
