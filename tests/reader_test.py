from xetrack import Tracker, Reader
import pandas as pd
from tempfile import NamedTemporaryFile
import multiprocessing as mp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_db(database):
    reader = Reader(database)
    assert len(reader.to_df()) == 2
    assert len(reader.latest()) == 1


def test_reader():
    tempfile = NamedTemporaryFile()
    database = tempfile.name
    tracker = Tracker(database)
    tracker.log(a=1, b=2)
    tracker = Tracker(database)
    tracker.log(a=1, c=2)
    stats_process = mp.Process(target=read_db, args=(database,))
    stats_process.start()
    stats_process.join()


def test_reader_set_value():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log(accuracy=0.9)
    Tracker(db=database, params={"model": 'lightgbm'}).log(accuracy=0.9)

    editor = Reader(database)
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
    Tracker(db=database, params={"model": 'lightgbm'}).log(accuracy=0.9)
    Tracker(db=database, params={"model": 'xgboost'}).log(accuracy=0.9)

    editor = Reader(database)
    df = editor.to_df()
    originals = df.to_dict('records')

    editor.set_where('accuracy', 0.8, 'model', 'lightgbm', originals[0]['track_id'])
    edited = editor.to_df().to_dict('records')
    assert edited[0]['accuracy'] == 0.8
    assert edited[1]['accuracy'] == 0.9

    Tracker(db=database, params={"model": 'lightgbm'}, reset=True).log(accuracy=0.9)
    Tracker(db=database, params={"model": 'xgboost'}).log(accuracy=0.9)
    editor = Reader(database)
    editor.set_where('accuracy', 0.8, 'model', 'xgboost')
    edited = editor.to_df().to_dict('records')
    assert edited[0]['accuracy'] == 0.9
    assert edited[1]['accuracy'] == 0.8


def test_reader_delete_run():
    database = NamedTemporaryFile().name
    Tracker(db=database, params={"model": 'lightgbm'}).log(accuracy=0.9)
    Tracker(db=database, params={"model": 'xgboost'}).log(accuracy=0.9)
    editor = Reader(database)
    latest = editor.latest().to_dict('records')[0]
    assert len(editor) == 2
    editor.delete_run(latest['track_id'])
    editor.to_df()
    assert len(editor) == 1
    assert editor.latest().to_dict('records')[0]['track_id'] != latest['track_id']
