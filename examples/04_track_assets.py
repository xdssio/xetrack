"""
Example 4: Track Assets (ML Models)

Demonstrates:
- Storing ML models as assets
- Hash-based deduplication
- Retrieving models
- Using hashes directly for efficiency
"""

from xetrack import Tracker
import os

# Check if assets are available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ sklearn not installed. Install with: pip install xetrack[dev]")


def main():
    print("=" * 60)
    print("Example 4: Track Assets (ML Models)")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("\n✗ This example requires scikit-learn")
        print("  Install with: pip install scikit-learn")
        return
    
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Create tracker with assets enabled
    tracker = Tracker('examples_data/assets.db', params={'dataset': 'iris'})
    
    # Example 1: Store model as asset
    print("\n1. Storing ML model as asset:")
    lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
    accuracy = lr.score(X_test, y_test)
    
    tracker.log({
        'model': 'logistic_regression',
        'accuracy': float(accuracy),
        'lr_model': lr  # Model is automatically stored as asset
    })
    
    model_hash = tracker.latest['lr_model']
    print(f"   Model accuracy: {accuracy:.4f}")
    print(f"   Model hash: {model_hash}")
    print(f"   Model stored as asset with deduplication")
    
    # Example 2: Retrieve model
    print("\n2. Retrieving model from assets:")
    retrieved_model = tracker.get(model_hash)
    retrieved_accuracy = retrieved_model.score(X_test, y_test)
    print(f"   Retrieved model accuracy: {retrieved_accuracy:.4f}")
    print(f"   Models match: {accuracy == retrieved_accuracy}")
    
    # Example 3: Using hash directly for efficiency
    print("\n3. Reusing model hash (efficient):")
    # Train another model with same params
    lr2 = LogisticRegression(max_iter=200).fit(X_train, y_train)
    
    # Use the hash directly instead of storing the model again
    tracker.log({
        'model': 'logistic_regression',
        'accuracy': float(lr2.score(X_test, y_test)),
        'lr_model': model_hash  # Reuse hash - saves encoding/hashing time
    })
    print(f"   Logged with existing hash: {model_hash}")
    print(f"   No duplicate storage - deduplication works!")
    
    # Example 4: Asset deduplication
    print("\n4. Testing asset deduplication:")
    # Train identical model
    lr3 = LogisticRegression(max_iter=200).fit(X_train, y_train)
    tracker.log({
        'model': 'logistic_regression_v3',
        'accuracy': float(lr3.score(X_test, y_test)),
        'lr_model': lr3
    })
    
    hash3 = tracker.latest['lr_model']
    print(f"   New model hash: {hash3}")
    print(f"   Same as first hash: {hash3 == model_hash}")
    print(f"   ✓ Deduplication prevents storing identical models multiple times")
    
    # Example 5: View all tracked experiments
    print("\n5. All tracked experiments:")
    df = tracker.to_df(all=True)
    print(df[['model', 'accuracy', 'lr_model']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ Assets tracking example complete!")
    print(f"Database: examples_data/assets.db")
    print("=" * 60)


if __name__ == "__main__":
    main()
