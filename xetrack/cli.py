import typer
from xetrack import Reader, copy as copy_db
from xetrack.tracker import Tracker
app = typer.Typer()


@app.command()
def head(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show")):
    """Show the first lines of the database evetns table"""
    typer.echo(f"Showing the first {n} lines")
    df = Reader(db).to_df(head=n)
    typer.echo(df.to_markdown())


@app.command()
def tail(db: str = typer.Argument(help='path to database'),
         n: int = typer.Option(5, help="Number of lines to show")):
    """Show the last lines of the database evetns table"""
    df = Reader(db).to_df(tail=n)
    typer.echo(df.to_markdown())


@app.command()
def copy(source: str = typer.Argument(help='path to source database - will not be modified'),
         target: str = typer.Argument(help='path to target database')):
    """Copy the data from one database to another"""
    copy_db(source, target)


@app.command()
def delete(db: str = typer.Argument(help='path to database'),
           track_id: str = typer.Argument(help='track ID to delete')):
    """Delete a specific run from the database"""
    Reader(db).delete_run(track_id)


@app.command()
def set(db: str = typer.Argument(help='path to database'),
        key: str = typer.Argument(help='key to set'),
        value: str = typer.Argument(help='value to set'),
        track_id: str = typer.Option(
            None, help='If provided, only events with this track id value would be changed.'),
        where_key: str = typer.Option(None, help='key to set'),
        where_value: str = typer.Option(None, help='value to set')):
    """
    Set a value in the database.

    This function sets a value in the specified database using the provided key and value.
    Optionally, a track ID can be specified to associate the value with a specific track.
    Additionally, a where clause can be specified to set the value only for records that match the given key-value pair.


    Examples:

        >>> xt set "path/to/database.db" "name" "John Doe" --track-id "cool-name" --where-key "age" --where-value 30
    """
    reader = Reader(db)
    if where_key is not None:
        reader.set_where(key, value, where_key, where_value, track_id)
    else:
        Reader(db).set_value(key, value, track_id=track_id)


@app.command()
def ls(db: str = typer.Argument(help='path to database'),
       column: str = typer.Argument(None, help='column values to list'),
       unique: bool = typer.Option(False, help='list unique values only'),
       track_id: str = typer.Option(None, help='track ID to list')):
    """List all the track IDs in the database"""
    df = Reader(db).to_df(track_id=track_id)
    if column is None:
        typer.echo(df.columns.tolist())
        return
    values = df[column]
    if unique:
        import pandas as pd
        values = pd.Series(values.unique(), name=column)
    typer.echo(values.to_markdown())


@app.command()
def get(db: str = typer.Argument(help='path to database'),
        key: str = typer.Argument(help='hash of model to retrive'),
        output: str = typer.Argument(help='output to save the file to')):
    tracker = Tracker(db)
    asset = tracker.assets.assets.get(key)
    if asset is None:
        typer.echo(f'Asset {key} not found in database {db}')
    else:
        with open(output, 'wb') as f:
            f.write(asset)
        typer.echo(f'Asset {key} saved to {output}')


@app.command()
def sql(db: str = typer.Argument(help='path to database'),
        query: str = typer.Argument(help='SQL query to execute')):
    reader = Reader(db)
    df = reader.conn.execute(query).df()
    typer.echo(df.to_markdown())
