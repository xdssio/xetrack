from typing import Literal, Union
from pandas.core.frame import DataFrame
from pip._internal.utils.misc import hide_value
import typer
from xetrack import Reader, copy as copy_db
from json import dumps
app = typer.Typer()


@app.command()
def head(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show"),
         json: bool = typer.Option(False, help="Prettify the output as json"),
         engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
         table_name: str = typer.Option("default", help="Name of the table to read from")):
    """Show the first lines of the database events table"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    df = Reader(db, engine=engine_literal, table_name=table_name).to_df(head=n)
    result = dumps(df.to_dict(orient='records'),
                   indent=4) if json else df.to_markdown()
    typer.echo(result)


@app.command()
def tail(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show"),
         json: bool = typer.Option(False, help="Prettify the output as json"),
         engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
         table_name: str = typer.Option("default", help="Name of the table to read from")):
    """Show the last lines of the database events table"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    df = Reader(db, engine=engine_literal, table_name=table_name).to_df(tail=n)
    result = dumps(df.to_dict(orient='records'),
                   indent=4) if json else df.to_markdown()
    typer.echo(result)


@app.command()
def columns(db: str = typer.Argument(help='path to database'),
            engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
            table_name: str = typer.Option("default", help="Name of the table to read from")):
    """List the columns in the database"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    columns_list = list(Reader(db, engine=engine_literal, table_name=table_name).to_df(head=1).columns)
    typer.echo(f"Columns in {db} table '{table_name}':\n")
    typer.echo(columns_list)


@app.command()
def copy(source: str = typer.Argument(help='path to source database - will not be modified'),
         target: str = typer.Argument(help='path to target database'),
         assets: bool = typer.Option(True, help='copy the assets')):
    """Copy the data from one database to another"""
    copy_db(source=source, target=target, assets=assets)


@app.command()
def delete(db: str = typer.Argument(help='path to database'),
           track_id: str = typer.Argument(help='track ID to delete'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
           table_name: str = typer.Option("default", help="Name of the table to read from")):
    """Delete a specific run from the database"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    Reader(db, engine=engine_literal, table_name=table_name).delete_run(track_id)


@app.command()
def set(db: str = typer.Argument(help='path to database'),
        key: str = typer.Argument(help='key to set'),
        value: str = typer.Argument(help='value to set'),
        track_id: Union[str , None] = typer.Option(
            None, help='If provided, only events with this track id value would be changed.'),
        where_key: Union[str, None] = typer.Option(None, help='key to set'),
        where_value: Union[str, None] = typer.Option(None, help='value to set'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
        table_name: str = typer.Option("default", help="Name of the table to read from")):
    """
    Set a value in the database.

    This function sets a value in the specified database using the provided key and value.
    Optionally, a track ID can be specified to associate the value with a specific track.
    Additionally, a where clause can be specified to set the value only for records that match the given key-value pair.


    Examples:

        >>> xt set "path/to/database.db" "name" "John Doe" --track-id "cool-name" --where-key "age" --where-value 30
    """
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    reader = Reader(db, engine=engine_literal, table_name=table_name)
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
       table_name: str = typer.Option("default", help="Name of the table to read from")):
    """List all the track IDs in the database"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    df = Reader(db, engine=engine_literal, table_name=table_name).to_df(track_id=track_id)
    if column is None:
        typer.echo(df.columns.tolist())
        return
    values = df[column]
    if unique:
        import pandas as pd
        values = pd.Series(values.unique(), name=column)
    typer.echo(values.to_markdown())


@app.command()
def sql(db: str = typer.Argument(help='path to database'),
        query: str = typer.Argument(help='SQL query to execute'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"'),
        table_name: str = typer.Option("default", help="Name of the table to read from")):
    """Execute a SQL query on the database and show the results as a markdown table - consider that the default table is 'db.default' or 'default'\n\nExample: xt sql path/to/database.db 'SELECT * FROM db.default LIMIT 5' """
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    reader = Reader(db, engine=engine_literal, table_name=table_name)
    df: DataFrame = reader.engine.execute_sql(query)
    typer.echo(df.to_markdown())


assets_app = typer.Typer(short_help="A helper for assets managments")
app.add_typer(assets_app, name="assets")


@assets_app.command()
def export(db: str = typer.Argument(help='path to database'),
           key: str = typer.Argument(help='hash of model to retrive'),
           output: str = typer.Argument(help='output to save the file to'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Export an asset from the database to a file."""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    tracker = Reader(db, engine=engine_literal)
    asset = tracker.assets.assets.get(key)
    if asset is None:
        typer.echo(f'Asset {key} not found in database {db}')
    else:
        with open(output, 'wb') as f:
            f.write(asset)  # type: ignore
        typer.echo(f'Asset {key} saved to {output}')


@assets_app.command()
def delete(db: str = typer.Argument(help='path to database'),
           asset: str = typer.Argument(help='Asset hash to delete'),
           column: str = typer.Option(
    None, help='Column to set to None'),
    remove_keys: bool = typer.Option(
    True, help='Remove the keys associated with the asset'),
    engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')
):
    """Delete a specific run from the database"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    Reader(db, engine=engine_literal).remove_asset(asset, column, remove_keys)


@assets_app.command()
def ls(db: str = typer.Argument(help='path to database'),
       engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """list all the assets in the database"""
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    typer.echo(f"Assets in {db}:\n")
    typer.echo("Key: Hash")
    typer.echo("----")
    for key, hash in Reader(db, engine=engine_literal).assets.keys.items():
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


def _get_df(db: str, columns: str = '', engine: str = "sqlite"):
    engine_literal: Literal['duckdb', 'sqlite'] = 'sqlite' if engine == 'sqlite' else 'duckdb'
    df = Reader(db, engine=engine_literal).to_df()
    if columns != '':
        columns_list = columns.split(',')
        df = df[columns_list]
    return df


@stats_app.command()
def describe(db: str = typer.Argument(help='path to database'),
             columns: str = typer.Option('', help='columns to describe - comma separated list (e.g. "col1,col2,col3")'),
             engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Describe columns in the database - use either numeric or categorical columns."""
    df = _get_df(db, columns, engine)
    typer.echo(df.describe().to_markdown())


@stats_app.command()
def top(db: str = typer.Argument(help='path to database'),
        column: str = typer.Argument(help='Entry with best value'),
        engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Get the maximum value in the column"""
    df = _get_df(db, engine=engine)
    df = df[df[column] == df[column].max()]
    typer.echo(str(df.to_markdown()))


@stats_app.command()
def bottom(db: str = typer.Argument(help='path to database'),
           column: str = typer.Argument(help='Entry with best value'),
           engine: str = typer.Option("sqlite", help='database engine to use: "duckdb" or "sqlite"')):
    """Get the maximum value in the column"""
    df = _get_df(db, engine=engine)
    df = df[df[column] == df[column].min()]
    typer.echo(str(df.to_markdown()))
