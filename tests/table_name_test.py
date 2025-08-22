import os
import tempfile
import pytest
from xetrack import Tracker, Reader
from xetrack.config import SCHEMA_PARAMS


class TestTableName:
    """Test suite for custom table names functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    def test_default_table_name(self, temp_db):
        """Test that default table name is 'default'"""
        tracker = Tracker(db=temp_db, engine="sqlite")
        assert tracker.table_name == "default"
        assert tracker.engine.table_name == "default"
    
    def test_custom_table_name_sqlite(self, temp_db):
        """Test custom table names with SQLite engine"""
        # Create tracker with custom table name
        tracker = Tracker(db=temp_db, engine="sqlite", table="experiments")
        assert tracker.table_name == "experiments"
        assert tracker.engine.table_name == "experiments"
        
        # Log some data
        tracker.log({"accuracy": 0.9, "epoch": 1})
        tracker.log({"accuracy": 0.92, "epoch": 2})
        
        # Verify data is stored in the custom table
        reader = Reader(db=temp_db, engine="sqlite", table="experiments")
        df = reader.to_df()
        assert len(df) == 2
        assert "accuracy" in df.columns
        assert df["accuracy"].tolist() == [0.9, 0.92]
    
    def test_custom_table_name_duckdb(self, temp_db):
        """Test custom table names with DuckDB engine"""
        try:
            # Create tracker with custom table name
            tracker = Tracker(db=temp_db, engine="duckdb", table="ml_experiments")
            # DuckDB should add db. prefix automatically
            assert tracker.table_name == "db.ml_experiments"
            assert tracker.engine.table_name == "db.ml_experiments"
            
            # Log some data
            tracker.log({"loss": 0.1, "epoch": 1})
            tracker.log({"loss": 0.08, "epoch": 2})
            
            # Verify data is stored in the custom table
            reader = Reader(db=temp_db, engine="duckdb", table="ml_experiments")
            df = reader.to_df()
            assert len(df) == 2
            assert "loss" in df.columns
            assert df["loss"].tolist() == [0.1, 0.08]
        except ImportError:
            pytest.skip("DuckDB not available")
    
    def test_multiple_tables_same_db(self, temp_db):
        """Test multiple experiment types in the same database"""
        # Create two different experiment types
        model_tracker = Tracker(db=temp_db, engine="sqlite", table="model_experiments")
        data_tracker = Tracker(db=temp_db, engine="sqlite", table="data_experiments")
        
        # Log data to different tables
        model_tracker.log({"model": "resnet", "accuracy": 0.9})
        data_tracker.log({"dataset": "cifar10", "size": 50000})
        
        # Verify each table has its own data
        model_reader = Reader(db=temp_db, engine="sqlite", table="model_experiments")
        data_reader = Reader(db=temp_db, engine="sqlite", table="data_experiments")
        
        model_df = model_reader.to_df()
        data_df = data_reader.to_df()
        
        assert len(model_df) == 1
        assert len(data_df) == 1
        assert "model" in model_df.columns
        assert "dataset" in data_df.columns
        assert "accuracy" in model_df.columns
        assert "size" in data_df.columns
    
    def test_reader_with_custom_table(self, temp_db):
        """Test Reader class with custom table names"""
        # Create some data
        tracker = Tracker(db=temp_db, engine="sqlite", table="test_table")
        tracker.log({"x": 1, "y": 2})
        tracker.log({"x": 3, "y": 4})
        
        # Read with custom table name
        reader = Reader(db=temp_db, engine="sqlite", table="test_table")
        
        # Test various reader methods
        df = reader.to_df()
        assert len(df) == 2
        
        # Test head/tail
        head_df = reader.to_df(head=1)
        assert len(head_df) == 1
        
        tail_df = reader.to_df(tail=1)
        assert len(tail_df) == 1
        
        # Test columns
        assert "x" in reader.columns
        assert "y" in reader.columns
    
    def test_set_value_with_custom_table(self, temp_db):
        """Test set_value functionality with custom tables"""
        tracker = Tracker(db=temp_db, engine="sqlite", table="custom_table")
        tracker.log({"score": 0.8})
        
        reader = Reader(db=temp_db, engine="sqlite", table="custom_table")
        
        # Set a new value
        reader.set_value("final_score", 0.95)
        
        df = reader.to_df()
        assert "final_score" in df.columns
        assert df["final_score"].iloc[0] == 0.95
    
    def test_delete_run_with_custom_table(self, temp_db):
        """Test delete_run functionality with custom tables"""
        tracker = Tracker(db=temp_db, engine="sqlite", table="delete_test")
        tracker.log({"value": 1})
        track_id = tracker.track_id
        
        reader = Reader(db=temp_db, engine="sqlite", table="delete_test")
        assert len(reader.to_df()) == 1
        
        # Delete the run
        reader.delete_run(track_id)
        assert len(reader.to_df()) == 0
    
    def test_table_name_validation(self, temp_db):
        """Test that table names are properly handled"""
        # Test with different table name formats
        valid_names = ["experiments", "test_table", "model_v1", "data123"]
        
        for table_name in valid_names:
            tracker = Tracker(db=temp_db, engine="sqlite", table=table_name)
            assert tracker.table_name == table_name
            
            # Should be able to log data
            tracker.log({"test": 1})
            
            reader = Reader(db=temp_db, engine="sqlite", table=table_name)
            df = reader.to_df()
            assert len(df) == 1
    
    def test_assets_with_custom_tables(self, temp_db):
        """Test that assets work with custom table names"""
        try:
            # Create tracker with custom table and some object
            tracker = Tracker(db=temp_db, engine="sqlite", table="assets_test")
            
            # Create a simple object to store
            test_obj = {"model": "test", "params": [1, 2, 3]}
            tracker.log({"obj": test_obj, "accuracy": 0.9})
            
            reader = Reader(db=temp_db, engine="sqlite", table="assets_test")
            df = reader.to_df()
            
            # Should have the object stored as hash
            assert len(df) == 1
            assert "obj" in df.columns
            
            # The object should be stored in assets
            if reader.assets:
                obj_hash = df["obj"].iloc[0]
                retrieved_obj = reader.assets.get(obj_hash)
                assert retrieved_obj == test_obj
        except ImportError:
            pytest.skip("sqlitedict not available for assets functionality")
    
    def test_backward_compatibility(self, temp_db):
        """Test that old code still works without specifying table_name"""
        # This should still work and use 'default' table
        tracker = Tracker(db=temp_db, engine="sqlite")
        tracker.log({"test": "backward_compat"})
        
        reader = Reader(db=temp_db, engine="sqlite")
        df = reader.to_df()
        
        assert len(df) == 1
        assert df["test"].iloc[0] == "backward_compat"
        assert tracker.table_name == "default"
        assert reader.table_name == "default"


if __name__ == "__main__":
    pytest.main([__file__])