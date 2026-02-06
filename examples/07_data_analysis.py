"""
Example 7: Data Analysis and Pandas Integration

Demonstrates:
- Converting to DataFrame
- Pandas-like operations
- Filtering by track_id
- Head/tail operations
- Column access
"""

from xetrack import Tracker, Reader
import random


def main():
    print("=" * 60)
    print("Example 7: Data Analysis with Pandas")
    print("=" * 60)
    
    # Create some experiment data
    tracker = Tracker('examples_data/analysis.db', params={'project': 'image_classification'})
    
    print("\n1. Generating experiment data...")
    models = ['resnet18', 'resnet50', 'vgg16', 'efficientnet']
    datasets = ['cifar10', 'cifar100', 'imagenet']
    
    for i in range(20):
        tracker.log({
            'model': random.choice(models),
            'dataset': random.choice(datasets),
            'accuracy': random.uniform(0.7, 0.95),
            'loss': random.uniform(0.05, 0.3),
            'epoch': random.randint(1, 50)
        })
    
    print(f"   Generated {len(tracker.to_df(all=True))} experiments")
    
    # Example 2: Convert to DataFrame
    print("\n2. Converting to pandas DataFrame:")
    df = tracker.to_df(all=True)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Example 3: Pandas-like operations
    print("\n3. Using Pandas operations:")
    print("\n   Top 5 by accuracy:")
    top_df = df.nlargest(5, 'accuracy')[['model', 'dataset', 'accuracy', 'loss']]
    print(top_df.to_string(index=False))
    
    print("\n   Average accuracy by model:")
    avg_by_model = df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
    for model, acc in avg_by_model.items():
        print(f"     {model}: {acc:.4f}")
    
    print("\n   Average accuracy by dataset:")
    avg_by_dataset = df.groupby('dataset')['accuracy'].mean().sort_values(ascending=False)
    for dataset, acc in avg_by_dataset.items():
        print(f"     {dataset}: {acc:.4f}")
    
    # Example 4: Column access (pandas-like)
    print("\n4. Column access:")
    accuracies = tracker['accuracy']
    print(f"   Accuracy stats:")
    print(f"     Mean: {accuracies.mean():.4f}")
    print(f"     Std: {accuracies.std():.4f}")
    print(f"     Min: {accuracies.min():.4f}")
    print(f"     Max: {accuracies.max():.4f}")
    
    # Example 5: Head and tail
    print("\n5. Head and tail operations:")
    print("\n   First 3 experiments:")
    print(tracker.head(3)[['model', 'accuracy', 'loss']].to_string(index=False))
    
    print("\n   Last 3 experiments:")
    print(tracker.tail(3)[['model', 'accuracy', 'loss']].to_string(index=False))
    
    # Example 6: Filtering and analysis
    print("\n6. Custom filtering:")
    high_accuracy = df[df['accuracy'] > 0.85]
    print(f"   Experiments with accuracy > 0.85: {len(high_accuracy)}")
    print(f"   Best model in this group: {high_accuracy.loc[high_accuracy['accuracy'].idxmax(), 'model']}")
    
    # Example 7: Using Reader for read-only access
    print("\n7. Read-only access with Reader:")
    reader = Reader('examples_data/analysis.db')
    df_readonly = reader.to_df()
    print(f"   Read {len(df_readonly)} experiments")
    
    # Filter by track_id
    specific_track = df['track_id'].iloc[0]
    df_track = reader.to_df(track_id=specific_track)
    print(f"   Experiments for track_id={specific_track}: {len(df_track)}")
    
    # Example 8: Statistical summaries
    print("\n8. Statistical summary:")
    print(df[['accuracy', 'loss', 'epoch']].describe().to_string())
    
    print("\n" + "=" * 60)
    print("✓ Data analysis example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
