"""
Example 6: Logging Integration

Demonstrates:
- Logging to stdout
- Logging to files
- JSONL format for data synthesis
- Logger-only mode (no database)
- Reading logs back
"""

from xetrack import Tracker, Reader
import os


def main():
    print("=" * 60)
    print("Example 6: Logging Integration")
    print("=" * 60)
    
    # Example 1: Log to stdout
    print("\n1. Logging to stdout:")
    tracker_stdout = Tracker(
        db=Tracker.IN_MEMORY,
        logs_stdout=True
    )
    
    tracker_stdout.log({"accuracy": 0.95, "loss": 0.05, "epoch": 1})
    print("   ↑ Structured log output above")
    
    # Example 2: Log to file
    print("\n2. Logging to file:")
    tracker_file = Tracker(
        db='examples_data/logging.db',
        logs_path='examples_data/logs'
    )
    
    for epoch in range(1, 4):
        tracker_file.log({
            "epoch": epoch,
            "train_loss": 0.1 / epoch,
            "val_loss": 0.12 / epoch,
            "lr": 0.001 * (0.9 ** epoch)
        })
    
    print(f"   Logs written to: examples_data/logs/")
    print(f"   Log files: {os.listdir('examples_data/logs') if os.path.exists('examples_data/logs') else '(none yet)'}")
    
    # Example 3: JSONL logging for ML datasets
    print("\n3. JSONL logging (for LLM training data):")
    tracker_jsonl = Tracker(
        db='examples_data/logging.db',
        jsonl='examples_data/logs/training_data.jsonl'
    )
    
    # Simulate logging prompts and responses for LLM training
    conversations = [
        {"subject": "python", "prompt": "How to sort a list?", "response": "Use list.sort()"},
        {"subject": "ml", "prompt": "What is gradient descent?", "response": "Optimization algorithm"},
        {"subject": "python", "prompt": "How to read a file?", "response": "Use open()"},
    ]
    
    for conv in conversations:
        tracker_jsonl.log(conv)
    
    print("   JSONL file created: examples_data/logs/training_data.jsonl")
    
    # Read back JSONL
    if os.path.exists('examples_data/logs/training_data.jsonl'):
        df_jsonl = Reader.read_jsonl('examples_data/logs/training_data.jsonl')
        print(f"   Read {len(df_jsonl)} entries from JSONL")
        print("\n   Sample entries:")
        print(df_jsonl[['subject', 'prompt', 'response']].head(2).to_string(index=False))
    
    # Example 4: Logger-only mode (no database writes)
    print("\n4. Logger-only mode (model monitoring):")
    tracker_monitor = Tracker(
        db=Tracker.SKIP_INSERT,  # Don't write to database
        logs_path='examples_data/logs',
        logs_stdout=True
    )
    
    print("   Logging inference metrics:")
    tracker_monitor.log({
        "model_version": "v1.2",
        "inference_time": 0.045,
        "input_tokens": 150,
        "output_tokens": 50,
        "user_id": "user_123"
    })
    print("   ↑ Logged to file only, no database insert")
    
    # Example 5: Reading logs back
    print("\n5. Reading logs for analysis:")
    if os.path.exists('examples_data/logs'):
        try:
            df_logs = Reader.read_logs(path='examples_data/logs')
            print(f"   Total log entries: {len(df_logs)}")
            if len(df_logs) > 0:
                print("\n   Recent logs:")
                print(df_logs.tail(3).to_string(index=False))
        except Exception as e:
            print(f"   Note: {e}")
    
    # Example 6: Custom log file format
    print("\n6. Custom log file naming:")
    tracker_custom = Tracker(
        db=Tracker.IN_MEMORY,
        logs_path='examples_data/logs',
        logs_file_format='{time:YYYY-MM-DD}_experiment.log'  # Daily logs
    )
    
    tracker_custom.log({"experiment": "A/B test", "variant": "A", "conversion": 0.12})
    print("   Custom log file format: YYYY-MM-DD_experiment.log")
    
    print("\n" + "=" * 60)
    print("✓ Logging integration example complete!")
    print("  Logs directory: examples_data/logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
