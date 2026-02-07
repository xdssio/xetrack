"""
sklearn Model Benchmark Template

Demonstrates benchmarking multiple sklearn models with hyperparameter search.
"""

from dataclasses import dataclass
from typing import Optional
from xetrack import Tracker, Reader
import time
import numpy as np


@dataclass(frozen=True, slots=True)
class ModelParams:
    """All parameters that affect the result."""
    model_type: str  # 'logistic', 'random_forest', 'svm'
    regularization: float = 1.0
    max_iter: int = 100
    random_state: int = 42


def train_and_evaluate(X_train, y_train, X_test, y_test, params: ModelParams) -> dict:
    """
    Single-execution function: trains model and returns all metrics.

    Returns everything you might need later:
    - train/test accuracy
    - training time
    - inference latency
    - model object (saved as asset)
    - error if failed
    """
    import time

    start_time = time.time()
    error = None
    model = None

    try:
        # Import based on model type
        if params.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=params.regularization,
                                      max_iter=params.max_iter,
                                      random_state=params.random_state)
        elif params.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(max_depth=int(params.regularization * 10),
                                          max_iter=params.max_iter,
                                          random_state=params.random_state)
        elif params.model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(C=params.regularization,
                       max_iter=params.max_iter,
                       random_state=params.random_state)
        else:
            raise ValueError(f"Unknown model type: {params.model_type}")

        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        inference_start = time.time()
        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test))
        inference_time = (time.time() - inference_start) / len(X_test)  # Per sample

    except Exception as e:
        error = str(e)
        train_time = time.time() - start_time
        train_accuracy = None
        test_accuracy = None
        inference_time = None

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'inference_time_per_sample': inference_time,
        'model': model,  # xetrack will save as asset with deduplication
        'error': error
    }


def run_benchmark():
    """Run full benchmark comparing multiple models and hyperparameters."""

    # Generate synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                              random_state=42)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Define parameter grid
    param_grid = [
        ModelParams(model_type='logistic', regularization=0.1),
        ModelParams(model_type='logistic', regularization=1.0),
        ModelParams(model_type='logistic', regularization=10.0),
        ModelParams(model_type='random_forest', regularization=0.5),
        ModelParams(model_type='random_forest', regularization=1.0),
        ModelParams(model_type='svm', regularization=0.1),
        ModelParams(model_type='svm', regularization=1.0),
    ]

    # Initialize trackers for both tables
    predictions_tracker = Tracker(
        db='sklearn_benchmark.db',
        engine='duckdb',
        cache='sklearn_cache',
        table='predictions',  # Individual results
        params={'experiment_id': 'sklearn-comparison-v1'}
    )

    metrics_tracker = Tracker(
        db='sklearn_benchmark.db',
        engine='duckdb',
        table='metrics',  # Aggregated metrics
        params={'experiment_id': 'sklearn-comparison-v1'}
    )

    # Run benchmark
    print(f"Running benchmark for {len(param_grid)} configurations...")
    for i, params in enumerate(param_grid, 1):
        print(f"  [{i}/{len(param_grid)}] {params.model_type} (C={params.regularization})...")

        # Track individual execution
        result = predictions_tracker.track(
            train_and_evaluate,
            args=[X_train, y_train, X_test, y_test, params]
        )

        # Log aggregated metrics to metrics table
        if result['error'] is None:
            metrics_tracker.log({
                'model_type': params.model_type,
                'regularization': params.regularization,
                'train_accuracy': result['train_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'train_time': result['train_time'],
                'inference_time': result['inference_time_per_sample'],
            })

        print(f"     Test Accuracy: {result.get('test_accuracy', 'FAILED')}")

    print("\n✅ Benchmark complete!")
    print("\nAnalyze results:")
    print("  python -c \"from xetrack import Reader; print(Reader('sklearn_benchmark.db', table='metrics', engine='duckdb').to_df())\"")


if __name__ == "__main__":
    run_benchmark()
