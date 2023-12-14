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
        tracker.log(i=i)

    reader = Reader(database)
    assert len(reader.to_df(track_id=tracker2.track_id)) == 0

    head = reader.to_df(head=2)['i']
    for i in range(2):
        assert head.iloc[i] == i

    tail = reader.to_df(tail=2)['i']
    for i in range(2):
        assert tail.iloc[i] == 3+i


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

    editor.set_where('accuracy', 0.8, 'model',
                     'lightgbm', originals[0]['track_id'])
    edited = _extracted_from_test_reader_set_where(editor, 0.8, 0.9)
    Tracker(db=database, params={"model": 'lightgbm'},
            reset=True).log(accuracy=0.9)
    Tracker(db=database, params={"model": 'xgboost'}).log(accuracy=0.9)
    editor = Reader(database)
    editor.set_where('accuracy', 0.8, 'model', 'xgboost')
    edited = _extracted_from_test_reader_set_where(editor, 0.9, 0.8)


# TODO Rename this here and in `test_reader_set_where`
def _extracted_from_test_reader_set_where(editor, arg1, arg2):
    result = editor.to_df().to_dict('records')
    assert result[0]['accuracy'] == arg1
    assert result[1]['accuracy'] == arg2

    return result


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
    assert editor.latest().to_dict('records')[
        0]['track_id'] != latest['track_id']


def test_reader_logs():
    tempdir = TemporaryDirectory()
    Tracker(db=Tracker.IN_MEMROY, logs_path=tempdir.name).log(accuracy=0.9)
    assert len(Reader.read_logs(tempdir.name)) == 1
