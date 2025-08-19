from tempfile import NamedTemporaryFile, TemporaryDirectory
from xetrack.cli import app
from typer.testing import CliRunner
from xetrack import Reader, Tracker
import json
import numpy as np
import shutil
import cloudpickle
import pytest
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
    result.exit_code == 0
    len(Reader(db).assets) == len(source.assets) == 1


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


def test_cli_bashplotlib():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9, 'x': 1.0, 'y': 2.0})
    result = runner.invoke(app, args=['plot', 'scatter', database, 'x', 'y'])
    assert result.exit_code == 0
    assert 'x vs y' in result.output
