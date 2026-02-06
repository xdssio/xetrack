"""
Example 9: Model Monitoring (Production Use Case)

Demonstrates:
- In-memory tracking for monitoring
- Logging to files without database
- Real-time inference monitoring
- Drift detection preparation
- Structured logging for analysis
"""

from xetrack import Tracker, Reader
import random
import time


def simulate_model_inference(input_data: dict, model_version: str) -> dict:
    """Simulate a model inference call"""
    time.sleep(random.uniform(0.01, 0.05))  # Simulate processing
    
    return {
        "prediction": random.choice(["class_a", "class_b", "class_c"]),
        "confidence": random.uniform(0.6, 0.99),
        "inference_time": random.uniform(0.01, 0.05)
    }


def main():
    print("=" * 60)
    print("Example 9: Model Monitoring (Production)")
    print("=" * 60)
    
    # Setup: Logger-only mode for production monitoring
    monitor = Tracker(
        db=Tracker.SKIP_INSERT,  # No database writes
        logs_path='examples_data/monitoring',
        logs_stdout=False,  # Don't spam stdout in production
        jsonl='examples_data/monitoring/inference.jsonl'  # Structured logging
    )
    
    print("\n1. Production inference monitoring:")
    print("   (Logging to file only, no database overhead)")
    
    # Simulate production traffic
    model_version = "v2.1.0"
    
    for i in range(10):
        # Simulate incoming request
        input_data = {
            "user_id": f"user_{random.randint(100, 999)}",
            "input_tokens": random.randint(50, 500),
            "request_id": f"req_{i:04d}"
        }
        
        # Run inference
        start = time.time()
        result = simulate_model_inference(input_data, model_version)
        inference_time = time.time() - start
        
        # Log everything
        monitor.log({
            "model_version": model_version,
            "user_id": input_data["user_id"],
            "request_id": input_data["request_id"],
            "input_tokens": input_data["input_tokens"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "inference_time": inference_time,
            "timestamp": time.time()
        })
    
    print(f"   Logged {10} inference requests")
    
    # Example 2: Read logs for analysis
    print("\n2. Analyzing monitoring logs:")
    import os
    if os.path.exists('examples_data/monitoring/inference.jsonl'):
        df = Reader.read_jsonl('examples_data/monitoring/inference.jsonl')
        
        print(f"   Total requests logged: {len(df)}")
        print(f"\n   Inference time statistics:")
        print(f"     Mean: {df['inference_time'].mean():.4f}s")
        print(f"     P50: {df['inference_time'].median():.4f}s")
        print(f"     P95: {df['inference_time'].quantile(0.95):.4f}s")
        print(f"     Max: {df['inference_time'].max():.4f}s")
        
        print(f"\n   Confidence statistics:")
        print(f"     Mean: {df['confidence'].mean():.4f}")
        print(f"     Min: {df['confidence'].min():.4f}")
        
        print(f"\n   Prediction distribution:")
        pred_dist = df['prediction'].value_counts()
        for pred, count in pred_dist.items():
            print(f"     {pred}: {count} ({count/len(df)*100:.1f}%)")
    
    # Example 3: Drift detection setup
    print("\n3. Setting up for drift detection:")
    print("   Logged data can be used for:")
    print("     - Input distribution monitoring")
    print("     - Confidence score drift")
    print("     - Prediction distribution changes")
    print("     - Performance degradation alerts")
    
    # Example 4: High-frequency monitoring
    print("\n4. High-frequency monitoring (fast logging):")
    
    fast_monitor = Tracker(
        db=Tracker.IN_MEMORY,  # Ultra-fast
        logs_path='examples_data/monitoring'
    )
    
    start = time.time()
    for i in range(100):
        fast_monitor.log({
            "request_id": f"fast_{i}",
            "latency": random.uniform(0.001, 0.01),
            "status": "success"
        })
    elapsed = time.time() - start
    
    print(f"   Logged 100 requests in {elapsed:.4f}s")
    print(f"   Throughput: {100/elapsed:.0f} logs/second")
    
    # Example 5: Structured fields for analytics
    print("\n5. Structured logging for easy analysis:")
    
    structured_monitor = Tracker(
        db=Tracker.SKIP_INSERT,
        jsonl='examples_data/monitoring/structured.jsonl'
    )
    
    # Log with consistent structure
    structured_monitor.log({
        "model_id": "fraud_detector_v1",
        "input": {"transaction_amount": 150.0, "merchant_category": "online"},
        "output": {"fraud_probability": 0.05, "decision": "approve"},
        "metadata": {"latency_ms": 12, "cache_hit": False}
    })
    
    print("   Structured format enables:")
    print("     - Easy querying with pandas/duckdb")
    print("     - Dashboard integration")
    print("     - Anomaly detection")
    print("     - A/B test analysis")
    
    print("\n" + "=" * 60)
    print("✓ Model monitoring example complete!")
    print("  Logs: examples_data/monitoring/")
    print("=" * 60)


if __name__ == "__main__":
    main()
