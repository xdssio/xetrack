from tempfile import NamedTemporaryFile, TemporaryDirectory
from tests.reader_test import _extracted_from_test_reader_set_where
from xetrack.cli import app
from typer.testing import CliRunner
from xetrack import Reader, Tracker
import json
import numpy as np
import shutil
import cloudpickle

runner = CliRunner()


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
    source = Tracker(db1)
    source.log({'a': 1, 'b': 1})

    shutil.copy(db1, db2)
    target = Tracker(db2)

    source.log({'a': 2, 'c': 2})
    source.log({'a': 3, 'c': 3})
    target.log({'a': 4, 'c': 4})

    source_size = len(source)
    target_size = len(target)

    result = runner.invoke(app, args=['copy', db2, db1])
    assert result.exit_code == 0

    result = runner.invoke(app, args=['copy', db1, db2])
    assert result.exit_code == 0

    assert len(Reader(db1)) == len(Reader(db2)) == 4

    source.log({'object': object()})
    assert len(source.assets) == 1
    result = runner.invoke(app, args=['copy', db1, db2, '--no-assets'])
    target_reader = Reader(db2)
    assert len(Reader(db1)) == len(target_reader) == 5
    assert len(target_reader.assets) == 0
    result = runner.invoke(app, args=['copy', db1, db2, '--assets'])
    assert len(target_reader.assets) == 1


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
    edited = _extracted_from_test_reader_set_where(reader, 0.8, 0.9)

    result = runner.invoke(
        app, args=['set', database, 'accuracy', '0.5'])
    edited = _extracted_from_test_reader_set_where(reader, 0.5, 0.5)


def test_cli_sql():
    database = NamedTemporaryFile().name
    result = Tracker(db=database, params={"model": 'lightgbm'}).log(
        {"accuracy": 0.9})
    result = runner.invoke(
        app, args=['sql', database, "SELECT * FROM db.events;"])
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
    tempdir = TemporaryDirectory()
    db: str = f'{tempdir.name}/db1.db'
    source = Tracker(db)
    y_data = np.random.normal(loc=1, scale=10, size=1000)
    x_data = np.random.normal(loc=2, scale=20, size=1000)
    for x, y in zip(x_data, y_data):
        _ = source.log({'x': float(x), 'y': float(y)})
    result = runner.invoke(app, args=['plot', 'hist', db, 'x'])
    assert result.exit_code == 0
    assert 'Summary' in result.output

    result = runner.invoke(app, args=['plot', 'scatter', db, 'x', 'y'])
    assert result.exit_code == 0
    assert 'x vs y ' in result.output
