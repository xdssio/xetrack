from tempfile import TemporaryDirectory
from xetrack import Tracker, copy
import pandas as pd
import shutil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_copy():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    target = Tracker(db2)

    class Mock():
        def __init__(self, name: str) -> None:
            self.name = name

    result = source.log({"a": 1, "b": 2, 'mock': Mock('a')})
    result = source.log({"a": 1, "b": 2, 'mock': Mock('test')})
    target.log({"a": 1, "c": 2, 'other': Mock('c')})
    target.log({"a": 1, "c": 2, 'other': Mock('a')})
    copy(source=db1, target=db2)
    df = Tracker(db2).to_df(all=True)
    assert len(df) == 4
    assert len(list(source.assets.assets.keys())) == 2
    assert len(list(source.assets.counts.items())) == 2
    assert target.get(result['mock']).name == 'test'  # by hash
    assert len(list(target.assets.assets.keys())) == 3  # deduplication
    for column in ('mock', 'other'):
        for hash_value in df[column].dropna().tolist():
            assert target.get(hash_value).name


def test_merge_repetitions():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'
    source = Tracker(db1)
    source.log({"a": 1, "b": 1})

    shutil.copy(db1, db2)
    target = Tracker(db2)

    source.log({"a": 2, "c": 2})
    source.log({"a": 3, "c": 3})
    target.log({"a": 4, "c": 4})

    assert copy(source=db2, target=db1) == 1
    assert copy(source=db2, target=db1) == 0  # already copied
    assert copy(source=db1, target=db2) == 2


def test_copy_datatypes():
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'

    source = Tracker(db1)
    source.log({"int_val": 42, "float_val": 3.14, "bool_val": True, "str_val": "test_string"})
    copy(source=db1, target=db2)
    target = Tracker(db2)
    source_df = source.to_df()
    target_df = target.to_df(all=True)
    assert (target_df.dtypes == source_df.dtypes).all()
    assert len(target_df) > 0
    tempdir.cleanup()


def test_copy_single_table():
    """Test copying a single specific table"""
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'

    # Create data in custom table
    source = Tracker(db1, table='experiments')
    source.log({"metric": 1.5, "epoch": 1})
    source.log({"metric": 2.0, "epoch": 2})

    # Copy specific table
    copy(source=db1, target=db2, tables=['experiments'])

    # Verify data was copied
    target = Tracker(db2, table='experiments')
    df = target.to_df(all=True)
    assert len(df) == 2
    assert 'metric' in df.columns
    assert 'epoch' in df.columns
    tempdir.cleanup()


def test_copy_multiple_tables():
    """Test copying multiple tables in one operation"""
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'

    # Create data in multiple tables
    source_exp = Tracker(db1, table='experiments')
    source_exp.log({"accuracy": 0.95})
    source_exp.log({"accuracy": 0.97})

    source_val = Tracker(db1, table='validation')
    source_val.log({"loss": 0.5})

    # Copy both tables
    total = copy(source=db1, target=db2, tables=['experiments', 'validation'])
    assert total == 3  # 2 from experiments + 1 from validation

    # Verify both tables were copied
    target_exp = Tracker(db2, table='experiments')
    df_exp = target_exp.to_df(all=True)
    assert len(df_exp) == 2

    target_val = Tracker(db2, table='validation')
    df_val = target_val.to_df(all=True)
    assert len(df_val) == 1
    tempdir.cleanup()


def test_copy_default_table():
    """Test that copy defaults to 'default' table when no tables specified"""
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'

    # Create data in default table
    source = Tracker(db1)  # Uses 'default' table
    source.log({"value": 100})

    # Copy without specifying tables - should use default
    copy(source=db1, target=db2)

    # Verify data was copied to default table
    target = Tracker(db2)  # Uses 'default' table
    df = target.to_df(all=True)
    assert len(df) == 1
    assert df.iloc[0]['value'] == 100
    tempdir.cleanup()


def test_copy_assets_with_multiple_tables():
    """Test that assets are copied once when copying multiple tables"""
    tempdir = TemporaryDirectory()
    db1 = tempdir.name + '/db1.db'
    db2 = tempdir.name + '/db2.db'

    class Mock():
        def __init__(self, name: str) -> None:
            self.name = name

    # Create data with assets in multiple tables
    source_t1 = Tracker(db1, table='table1')
    source_t1.log({"value": 1, "model": Mock("model1")})

    source_t2 = Tracker(db1, table='table2')
    source_t2.log({"value": 2, "model": Mock("model2")})

    # Copy both tables with assets
    copy(source=db1, target=db2, assets=True, tables=['table1', 'table2'])

    # Verify assets are accessible from both tables
    target_t1 = Tracker(db2, table='table1')
    target_t2 = Tracker(db2, table='table2')

    df1 = target_t1.to_df(all=True)
    df2 = target_t2.to_df(all=True)

    # Get model hashes and verify they can be retrieved
    model1_hash = df1.iloc[0]['model']
    model2_hash = df2.iloc[0]['model']

    assert target_t1.get(model1_hash).name == 'model1'
    assert target_t2.get(model2_hash).name == 'model2'
    tempdir.cleanup()
