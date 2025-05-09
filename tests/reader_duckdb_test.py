from xetrack import Tracker, Reader
import pandas as pd
from tempfile import NamedTemporaryFile, TemporaryDirectory
import multiprocessing as mp
from xetrack.cli import tail

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_reader_to_df():
    tempfile = NamedTemporaryFile()
    database = tempfile.name
    tracker = Tracker(database)
    tracker2 = Tracker(database)

    for i in range(5):
        tracker.log({"i": i})

    reader = Reader(database, engine="duckdb")
    assert len(reader.to_df(track_id=tracker2.track_id)) == 0

    head = reader.to_df(head=2)['i']
    for i in range(2):
        assert head.iloc[i] == i

    tail = reader.to_df(tail=2)['i']
    for i in range(2):
        assert tail.iloc[i] == 3+i


def read_db(database: str):
    reader = Reader(database, engine="duckdb")
    assert len(reader.to_df()) == 2
    assert len(reader.latest()) == 1


def test_reader():
    tempfile = NamedTemporaryFile()
    database = tempfile.name
    tracker = Tracker(database)
    tracker.log({"a": 1, "b": 2})
    tracker = Tracker(database)
    tracker.log({"a": 1, "c": 2})
    stats_process = mp.Process(target=read_db, args=(database,))
    stats_process.start()
    stats_process.join()


def test_reader_set_value():
    database = NamedTemporaryFile().name
    Tracker(db=database,engine='duckdb', params={"model": 'lightgbm'}).log({"accuracy": 0.9})
    Tracker(db=database,engine='duckdb', params={"model": 'lightgbm'}).log({"accuracy": 0.9})

    editor = Reader(database, engine="duckdb")
    latest = editor.latest().to_dict('records')[0]
    assert latest['accuracy'] == 0.9

    editor.set_value('accuracy', 0.8, latest['track_id'])
    records = editor.to_df().to_dict('records')
    for record in records:
        if record['track_id'] == latest['track_id']:
            assert record['accuracy'] == 0.8
        else:
            assert record['accuracy'] == 0.9


def test_reader_set_where():
    database = NamedTemporaryFile().name
    Tracker(db=database,engine='duckdb', params={"model": 'lightgbm'}).log({"accuracy": 0.9})
    Tracker(db=database,engine='duckdb', params={"model": 'xgboost'}).log({"accuracy": 0.9})

    editor = Reader(database, engine="duckdb")
    df = editor.to_df()
    originals = df.to_dict('records')

    editor.set_where('accuracy', 0.8, 'model',
                     'lightgbm', originals[0]['track_id'])
    # Check that one record has accuracy 0.8 and one has 0.9
    df = editor.to_df()
    assert set(df['accuracy'].tolist()) == {0.8, 0.9}
    
    Tracker(db=database, params={"model": 'lightgbm'},
            reset=True).log({"accuracy": 0.9})
    Tracker(db=database, params={"model": 'xgboost'}).log({"accuracy": 0.9})
    editor = Reader(database, engine="duckdb")
    editor.set_where('accuracy', 0.8, 'model', 'xgboost')
    
    # Check that one record has model='lightgbm' and accuracy=0.9, and one has model='xgboost' and accuracy=0.8
    df = editor.to_df()
    values = [(row['model'], row['accuracy']) for row in df.to_dict('records')]
    assert ('lightgbm', 0.9) in values
    assert ('xgboost', 0.8) in values


def test_reader_delete_run():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log({"accuracy": 0.9})
    Tracker(db=database, params={"model": 'xgboost'}).log({"accuracy": 0.9})
    editor = Reader(database, engine="duckdb")
    latest = editor.latest().to_dict('records')[0]
    assert len(editor) == 2
    editor.delete_run(latest['track_id'])
    editor.to_df()
    assert len(editor) == 1
    assert editor.latest().to_dict('records')[
        0]['track_id'] != latest['track_id']


def test_reader_logs():
    tempdir = TemporaryDirectory()
    Tracker(db=Tracker.IN_MEMORY, logs_path=tempdir.name).log(
        {"accuracy": 0.9})
    assert len(Reader.read_logs(tempdir.name)) == 1
