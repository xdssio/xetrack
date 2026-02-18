from tempfile import NamedTemporaryFile, TemporaryDirectory
from xetrack.cli import app
from typer.testing import CliRunner
from xetrack import Reader, Tracker
import json
import cloudpickle
import pandas as pd
runner = CliRunner()

# Helper function for checking set values
def _check_reader_values(reader: Reader, expected_values: list) -> pd.DataFrame:
    df = reader.to_df()
    values = sorted(df['accuracy'].tolist())
    assert set(values) == set(expected_values)
    return df

def test_cli_head():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    source = Tracker(db1)
    for i in range(10):
        source.log({i: i})
    result = runner.invoke(app, args=['head', db1])
    assert 'timestamp' in result.output and 'track_id' in result.output and 'i' in result.output

    result = runner.invoke(app, args=['head', db1, '--json'])
    assert 'timestamp' in json.loads(result.output)[0]


def test_cli_tail():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    source = Tracker(db1)
# sourcery skip: no-loop-in-tests
    for i in range(10):
        source.log({i: i})

    result = runner.invoke(app, args=['tail', db1])
    assert 'timestamp' in result.output and 'track_id' in result.output and 'i' in result.output

    result = runner.invoke(app, args=['tail', db1, '--json'])
    assert 'timestamp' in json.loads(result.output)[0]


def test_cli_columns():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    source = Tracker(db1)
    for i in range(10):
        source.log({i: i})
    result = runner.invoke(app, args=['columns', db1])
    assert 'timestamp' in result.output and 'track_id' in result.output and '0' in result.output


def test_cli_copy():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    db2 = f'{tempdir.name}/db2.db'
    db3 = f'{tempdir.name}/db3.db'
    
    source = Tracker(db1)    
    source.log({'a': 1, 'b': 1})    
    # Test copy with default behavior (assets=True)
    result = runner.invoke(app, args=['copy', db1, db2])
    assert result.exit_code == 0
        
    reader2 = Reader(db2)
    
    assert 1 in set(reader2.to_df()['a'].tolist())
        
    result = runner.invoke(app, args=['copy', db1, db3, '--no-assets'])
    assert result.exit_code == 0
        
    reader3 = Reader(db3)
    assert 1 in set(reader3.to_df()['a'].tolist())
    
    source.log({'a': 2, 'c': 2})
    source.log({'a': 3, 'c': 3})
    
    target = Tracker(db2)
    target.log({'a': 4, 'c': 4})
    
    result = runner.invoke(app, args=['copy', db2, db1])
    assert result.exit_code == 0
    
    result = runner.invoke(app, args=['copy', db1, db2])
    assert result.exit_code == 0
    
    # Verify that both databases contain the expected records by checking 'a' values
    db1_values = set(Reader(db1).to_df()['a'].tolist()) # type: ignore
    db2_values = set(Reader(db2).to_df()['a'].tolist()) # type: ignore
        
    for i in range(1, 5):
        assert i in db1_values
        assert i in db2_values


def test_cli_copy_with_table_param():
    """Test CLI copy command with --table parameter"""
    tempdir = TemporaryDirectory()
    db1 = f"{tempdir.name}/db1.db"
    db2 = f"{tempdir.name}/db2.db"

    # Create data in multiple tables
    exp_tracker = Tracker(db1, table="experiments")
    exp_tracker.log({"accuracy": 0.95, "epoch": 1})
    exp_tracker.log({"accuracy": 0.97, "epoch": 2})

    val_tracker = Tracker(db1, table="validation")
    val_tracker.log({"loss": 0.5, "epoch": 1})

    # Test copying single table
    result = runner.invoke(app, args=["copy", db1, db2, "--table=experiments"])
    assert result.exit_code == 0

    # Verify only experiments table was copied
    exp_reader = Reader(db2, table="experiments")
    assert len(exp_reader.to_df()) == 2

    # Test copying multiple tables
    db3 = f"{tempdir.name}/db3.db"
    result = runner.invoke(
        app, args=["copy", db1, db3, "--table=experiments", "--table=validation"]
    )
    assert result.exit_code == 0

    # Verify both tables were copied
    exp_reader3 = Reader(db3, table="experiments")
    val_reader3 = Reader(db3, table="validation")
    assert len(exp_reader3.to_df()) == 2
    assert len(val_reader3.to_df()) == 1


def test_cli_delete():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({'accuracy': 0.9})
    Tracker(db=database, params={"model": 'xgboost'}).log({'accuracy': 0.9})
    reader = Reader(database)
    latest = reader.latest().to_dict('records')[0]
    assert len(reader) == 2
    result = runner.invoke(app, args=['delete', database, latest['track_id']])
    print(result.output)

    assert len(reader) == 1
    assert reader.latest().to_dict('records')[
        0]['track_id'] != latest['track_id']


def test_cli_set_value():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9})
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9})

    reader = Reader(database)
    latest = reader.latest().to_dict('records')[0]
    assert latest['accuracy'] == 0.9

    result = runner.invoke(
        app, args=['set', database, 'accuracy', '0.8', '--track-id', latest['track_id']])
    records = reader.to_df().to_dict('records')
    for record in records:
        if record['track_id'] == latest['track_id']:
            assert record['accuracy'] == 0.8
        else:
            assert record['accuracy'] == 0.9


def test_cli_where():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9})
    Tracker(db=database, params={"model": 'xgboost'}).log({"accuracy": 0.9})

    reader = Reader(database)
    df = reader.to_df()
    originals = df.to_dict('records')

    latest = reader.latest().to_dict('records')[0]
    result = runner.invoke(
        app, args=['set', database, 'accuracy', '0.8', '--where-key', 'model', '--where-value', 'lightgbm'])
    edited = _check_reader_values(reader, [0.8, 0.9])

    result = runner.invoke(
        app, args=['set', database, 'accuracy', '0.5'])
    edited = _check_reader_values(reader, [0.5, 0.5])


def test_cli_sql():
    database = NamedTemporaryFile().name
    result = Tracker(db=database, params={"model": 'lightgbm'}).log(
        {"accuracy": 0.9})
    result = runner.invoke(
        app, args=['sql', database, 'SELECT * FROM "default";'])
    assert result.exit_code == 0
    assert 'accuracy' in result.output
    assert '0.9' in result.output


def test_cli_assets_export():
    class MockObject:
        def __init__(self, name):
            self.name = name
    database = NamedTemporaryFile().name
    model_path = NamedTemporaryFile().name

    result = Tracker(db=database, params={"model": 'lightgbm'}).log(
        {"mock": MockObject('mock')})
    result = runner.invoke(
        app, args=['assets', 'export', database, result['mock'], model_path])
    assert result.exit_code == 0

    with open(model_path, 'rb') as f:
        obj = cloudpickle.load(f)
    assert obj.name == 'mock'
    assert isinstance(obj, MockObject)


def test_cli_assets_delete():
    class MockObject:
        def __init__(self, name: str):
            self.name: str = name

    tempdir = TemporaryDirectory()
    db: str = f'{tempdir.name}/db1.db'
    source = Tracker(db)
    log = source.log({'o1': MockObject('o1'), 'o2': MockObject('o2')})
    assert len(source.assets) == 2
    result = runner.invoke(app, args=['assets', 'delete', db, log['o1']])
    assert result.exit_code == 0
    assert len(Reader(db).assets) == len(source.assets) == 1


def test_cli_assets_ls():
    class MockObject:
        def __init__(self, name: str):
            self.name: str = name

    tempdir = TemporaryDirectory()
    db: str = f'{tempdir.name}/db1.db'
    source = Tracker(db)
    log = source.log({'o1': MockObject('o1'), 'o2': MockObject('o2')})
    assert len(source.assets) == 2
    result = runner.invoke(app, args=['assets', 'ls', db])
    result.exit_code == 0


def test_cli_cache_ls():
    """Test xt cache ls lists entries with track_id lineage."""
    tempdir = TemporaryDirectory()
    db_path = f'{tempdir.name}/db.db'
    cache_path = f'{tempdir.name}/cache'

    tracker = Tracker(db=db_path, cache=cache_path)

    def add(a: int, b: int) -> int:
        return a + b

    tracker.track(add, args=[1, 2])
    tracker.track(add, args=[3, 4])

    result = runner.invoke(app, args=['cache', 'ls', cache_path])
    assert result.exit_code == 0
    assert 'track_id' in result.output
    assert tracker.track_id in result.output
    tempdir.cleanup()


def test_cli_cache_ls_empty():
    """Test xt cache ls on empty cache."""
    tempdir = TemporaryDirectory()
    cache_path = f'{tempdir.name}/cache'

    # Create empty cache dir
    from diskcache import Cache
    Cache(cache_path).close()

    result = runner.invoke(app, args=['cache', 'ls', cache_path])
    assert result.exit_code == 0
    assert 'empty' in result.output.lower()
    tempdir.cleanup()


def test_cli_cache_delete():
    """Test xt cache delete removes entries for a specific track_id."""
    tempdir = TemporaryDirectory()
    db_path = f'{tempdir.name}/db.db'
    cache_path = f'{tempdir.name}/cache'

    tracker = Tracker(db=db_path, cache=cache_path)

    def mul(a: int, b: int) -> int:
        return a * b

    def div(a: int, b: int) -> float:
        return a / b

    # Cache two functions under same track_id
    tracker.track(mul, args=[2, 3])
    tracker.track(div, args=[10, 2])

    # Verify 2 entries exist
    entries = list(Reader.scan_cache(cache_path))
    assert len(entries) == 2

    # Delete via CLI
    result = runner.invoke(app, args=['cache', 'delete', cache_path, tracker.track_id])
    assert result.exit_code == 0
    assert 'Deleted 2' in result.output

    # Verify cache is empty
    entries_after = list(Reader.scan_cache(cache_path))
    assert len(entries_after) == 0
    tempdir.cleanup()


def test_cli_cache_delete_partial():
    """Test xt cache delete only removes entries for the specified track_id."""
    tempdir = TemporaryDirectory()
    db_path = f'{tempdir.name}/db.db'
    cache_path = f'{tempdir.name}/cache'

    tracker = Tracker(db=db_path, cache=cache_path)
    first_id = tracker.track_id

    def compute(x: int) -> int:
        return x * 2

    # Cache under first track_id
    tracker.track(compute, args=[5])

    # Switch track_id, cache different args
    tracker.track_id = Tracker.generate_track_id()
    second_id = tracker.track_id
    tracker.track(compute, args=[10])

    assert len(list(Reader.scan_cache(cache_path))) == 2

    # Delete only first
    result = runner.invoke(app, args=['cache', 'delete', cache_path, first_id])
    assert result.exit_code == 0
    assert 'Deleted 1' in result.output

    # Second should remain
    remaining = list(Reader.scan_cache(cache_path))
    assert len(remaining) == 1
    assert remaining[0][1]["cache"] == second_id
    tempdir.cleanup()


def test_cli_bashplotlib():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9, 'x': 1.0, 'y': 2.0})
    result = runner.invoke(app, args=['plot', 'scatter', database, 'x', 'y'])
    assert result.exit_code == 0
    assert 'x vs y' in result.output
