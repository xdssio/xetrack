from tempfile import NamedTemporaryFile, TemporaryDirectory
from tests.reader_test import _extracted_from_test_reader_set_where
from xetrack.cli import app, copy, head, set, tail
from typer.testing import CliRunner
from xetrack import Reader, Tracker
import shutil
import cloudpickle

runner = CliRunner()


def test_cli_head():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    source = Tracker(db1)
    for i in range(10):
        source.log({i: i})
    result = runner.invoke(app, args=['tail', db1])
    assert 'timestamp' in result.output and 'track_id' in result.output and 'i' in result.output


def test_cli_tail():
    tempdir = TemporaryDirectory()
    db1 = f'{tempdir.name}/db1.db'
    source = Tracker(db1)
# sourcery skip: no-loop-in-tests
    for i in range(10):
        source.log({i: i})

    result = runner.invoke(app, args=['tail', db1])
    assert 'timestamp' in result.output and 'track_id' in result.output and 'i' in result.output


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
    assert 'Copied 1 events' in result.output
    assert 'total is 4 events' in result.output

    result = runner.invoke(app, args=['copy', db1, db2])
    assert 'Copied 2 events' in result.output
    assert 'total is 4 events' in result.output

    assert len(Reader(db1)) == len(Reader(db2)) == 4


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


def test_cli_get():
    class MockObject:
        def __init__(self, name):
            self.name = name
    database = NamedTemporaryFile().name
    model_path = NamedTemporaryFile().name

    result = Tracker(db=database, params={"model": 'lightgbm'}).log(
        {"mock": MockObject('mock')})
    result = runner.invoke(
        app, args=['get', database, result['mock'], model_path])
    assert result.exit_code == 0

    with open(model_path, 'rb') as f:
        obj = cloudpickle.load(f)
    assert obj.name == 'mock'
    assert isinstance(obj, MockObject)


def test_cli_sql():
    database = NamedTemporaryFile().name
    result = Tracker(db=database, params={"model": 'lightgbm'}).log(
        {"accuracy": 0.9})
    result = runner.invoke(
        app, args=['sql', database, "SELECT * FROM db.events;"])
    assert result.exit_code == 0
    assert 'accuracy' in result.output
    assert '0.9' in result.output
