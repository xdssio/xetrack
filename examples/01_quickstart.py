"""
Example 1: Quickstart - Basic usage of xetrack

Demonstrates:
- Creating a tracker
- Setting params
- Logging metrics
- Accessing latest results
- Converting to DataFrame
"""

from xetrack import Tracker

def main():
    print("=" * 60)
    print("Example 1: Quickstart - Basic Usage")
    print("=" * 60)
    
    # Create a tracker with default params
    tracker = Tracker('examples_data/quickstart.db', params={'model': 'resnet18'})
    
    # Log some metrics
    print("\n1. Logging single metrics:")
    tracker.log({"accuracy": 0.9, "loss": 0.1, "epoch": 1})
    print(f"   Latest: {tracker.latest}")
    
    # Set additional params
    print("\n2. Setting params for all future logs:")
    tracker.set_params({'dataset': 'cifar10'})
    tracker.log({"accuracy": 0.92, "loss": 0.08, "epoch": 2})
    print(f"   Latest: {tracker.latest}")
    
    # Set value for entire run (retroactively)
    print("\n3. Setting value for entire run:")
    tracker.set_value('test_accuracy', 0.89)
    
    # View as DataFrame
    print("\n4. Converting to DataFrame:")
    df = tracker.to_df()
    print(df[['epoch', 'accuracy', 'loss', 'test_accuracy']])
    
    # Multiple experiment types in same database
    print("\n5. Using different tables for different experiments:")
    model_tracker = Tracker('examples_data/quickstart.db', table='model_experiments')
    data_tracker = Tracker('examples_data/quickstart.db', table='data_experiments')
    
    model_tracker.log({'model': 'resnet18', 'params': 11_000_000})
    data_tracker.log({'dataset': 'imagenet', 'size': 1_000_000})
    
    print(f"   Model experiments: {len(model_tracker.to_df(all=True))} rows")
    print(f"   Data experiments: {len(data_tracker.to_df(all=True))} rows")
    
    print("\n" + "=" * 60)
    print("✓ Quickstart example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
