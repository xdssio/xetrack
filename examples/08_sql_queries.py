"""
Example 8: SQL Queries (SQLite and DuckDB)

Demonstrates:
- Direct SQL queries
- Using SQLite engine
- Using DuckDB engine
- Complex queries and aggregations
- Joining with assets table
"""

from xetrack import Tracker, Reader
import random


def main():
    print("=" * 60)
    print("Example 8: SQL Queries")
    print("=" * 60)
    
    # Generate data with SQLite
    print("\n1. Setting up data (SQLite engine):")
    tracker_sqlite = Tracker("examples_data/sql_demo_sqlite.db", engine="sqlite")
    
    models = ['resnet', 'vgg', 'efficientnet']
    for i in range(15):
        tracker_sqlite.log({
            'model': random.choice(models),
            'accuracy': random.uniform(0.7, 0.95),
            'loss': random.uniform(0.05, 0.3),
            'epoch': random.randint(1, 100),
            'lr': random.choice([0.001, 0.01, 0.1])
        })
    
    print(f"   Created {len(tracker_sqlite.to_df(all=True))} experiments")
    
    # Example 2: Direct SQL with SQLite
    print("\n2. SQL queries with SQLite:")
    
    # Get top 5 by accuracy
    query = """
    SELECT model, accuracy, loss, epoch 
    FROM "default"
    ORDER BY accuracy DESC 
    LIMIT 5
    """
    result = tracker_sqlite.conn.execute(query).fetchall()
    print("   Top 5 by accuracy:")
    for row in result:
        print(f"     {row[0]}: acc={row[1]:.4f}, loss={row[2]:.4f}, epoch={row[3]}")
    
    # Aggregation query
    query_agg = """
    SELECT model, 
           COUNT(*) as experiments,
           AVG(accuracy) as avg_acc,
           MAX(accuracy) as max_acc
    FROM "default"
    GROUP BY model
    ORDER BY avg_acc DESC
    """
    result_agg = tracker_sqlite.conn.execute(query_agg).fetchall()
    print("\n   Aggregation by model:")
    for row in result_agg:
        print(f"     {row[0]}: {row[1]} runs, avg={row[2]:.4f}, max={row[3]:.4f}")
    
    # Example 3: DuckDB for analytics
    print("\n3. Using DuckDB engine for analytics:")
    try:
        # Create tracker with DuckDB (separate database for demo purposes)
        tracker_duckdb = Tracker("examples_data/sql_demo_duckdb.db", engine="duckdb")

        # Copy some data to DuckDB database
        for i in range(15):
            tracker_duckdb.log(
                {
                    "model": random.choice(models),
                    "accuracy": random.uniform(0.7, 0.95),
                    "loss": random.uniform(0.05, 0.3),
                    "epoch": random.randint(1, 100),
                    "lr": random.choice([0.001, 0.01, 0.1]),
                }
            )
        
        # DuckDB queries
        query_duckdb = """
        SELECT 
            model,
            COUNT(*) as total_runs,
            AVG(accuracy) as avg_accuracy,
            STDDEV(accuracy) as std_accuracy,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY accuracy) as median_accuracy
        FROM db.default
        GROUP BY model
        ORDER BY avg_accuracy DESC
        """
        
        result_duckdb = tracker_duckdb.conn.execute(query_duckdb).fetchall()
        print("   Advanced analytics with DuckDB:")
        for row in result_duckdb:
            print(f"     {row[0]}:")
            print(f"       Runs: {row[1]}, Avg: {row[2]:.4f}, Std: {row[3]:.4f}, Median: {row[4]:.4f}")
        
        # Window functions
        query_window = """
        SELECT 
            model,
            accuracy,
            ROW_NUMBER() OVER (PARTITION BY model ORDER BY accuracy DESC) as rank_in_model
        FROM db.default
        QUALIFY rank_in_model <= 3
        ORDER BY model, rank_in_model
        """
        
        result_window = tracker_duckdb.conn.execute(query_window).fetchall()
        print("\n   Top 3 per model (window function):")
        current_model = None
        for row in result_window:
            if row[0] != current_model:
                current_model = row[0]
                print(f"     {current_model}:")
            print(f"       #{row[2]}: {row[1]:.4f}")

    except ImportError:
        print("   ⚠ DuckDB not installed")
        print("     Install with: pip install xetrack[duckdb]")
    
    # Example 4: Complex filtering
    print("\n4. Complex SQL filtering:")
    query_filter = """
    SELECT model, COUNT(*) as high_acc_runs
    FROM "default"
    WHERE accuracy > 0.85 AND epoch > 50
    GROUP BY model
    HAVING COUNT(*) > 0
    """
    result_filter = tracker_sqlite.conn.execute(query_filter).fetchall()
    print("   High accuracy (>0.85) + long training (>50 epochs):")
    for row in result_filter:
        print(f"     {row[0]}: {row[1]} runs")
    
    # Example 5: Using Reader for filtering
    print("\n5. Filtering with Reader:")
    reader = Reader("examples_data/sql_demo_sqlite.db")
    
    # Reader supports head/tail/track_id filtering
    df_all = reader.to_df()
    df_high_acc = df_all[df_all['accuracy'] > 0.8]
    print(f"   Experiments with accuracy > 0.8: {len(df_high_acc)}")
    
    # Or use direct SQL on the connection
    import pandas as pd
    df_sql = pd.read_sql_query('SELECT * FROM "default" WHERE accuracy > 0.8', 
                                tracker_sqlite.conn)
    print(f"   Via pd.read_sql_query: {len(df_sql)} experiments")
    
    print("\n" + "=" * 60)
    print("✓ SQL queries example complete!")
    print("  SQLite database: examples_data/sql_demo_sqlite.db")
    print("  DuckDB database: examples_data/sql_demo_duckdb.db")
    print("=" * 60)


if __name__ == "__main__":
    main()
