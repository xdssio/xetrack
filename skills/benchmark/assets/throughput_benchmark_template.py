"""
Throughput / Load Testing Benchmark Template

Simulates concurrent API requests to measure throughput, latency percentiles,
and system resource usage. Useful for benchmarking inference servers or APIs.
"""

from dataclasses import dataclass
from typing import Optional
from xetrack import Tracker, Reader
import time
import random
from multiprocessing import Pool, cpu_count
from functools import partial


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """Load testing configuration."""
    api_name: str  # 'inference-api-v1', 'classifier-v2'
    concurrency: int  # Number of parallel workers
    requests_per_worker: int
    timeout_sec: float = 5.0
    model_name: str = 'default'


def simulate_api_call(request_id: int, config: LoadConfig) -> dict:
    """
    Simulate a single API request.

    Returns:
    - request_id
    - latency (seconds)
    - status_code
    - response_size (bytes)
    - error (if failed)
    """
    start_time = time.time()
    error = None
    status_code = 200

    try:
        # Simulate API latency (varies by load)
        # Higher concurrency → higher latency (simulate contention)
        base_latency = 0.05  # 50ms
        contention_factor = 1 + (config.concurrency / 10) * random.uniform(0, 0.5)
        latency_sec = base_latency * contention_factor

        # Add some random variation
        latency_sec += random.gauss(0, 0.01)  # 10ms stddev
        latency_sec = max(0.001, latency_sec)  # Min 1ms

        # Simulate occasional timeouts
        if latency_sec > config.timeout_sec:
            status_code = 504  # Gateway timeout
            error = "Request timeout"

        # Simulate occasional errors
        elif random.random() < 0.01:  # 1% error rate
            status_code = random.choice([500, 503])
            error = "Server error"

        # Actually sleep to simulate work
        time.sleep(latency_sec / 10)  # Speed up for demo

        # Simulate response size
        response_size = random.randint(100, 5000)

    except Exception as e:
        error = str(e)
        status_code = 500
        response_size = 0

    actual_latency = time.time() - start_time

    return {
        'request_id': request_id,
        'latency': round(actual_latency, 4),
        'status_code': status_code,
        'response_size': response_size,
        'success': error is None and status_code == 200,
        'error': error
    }


def worker_function(worker_id: int, config: LoadConfig, db_path: str):
    """
    Worker function for parallel execution.

    Each worker makes N requests and tracks them.
    """
    # Each worker creates its own tracker (safe for multiprocessing)
    tracker = Tracker(
        db=db_path,
        engine='duckdb',
        table='requests',
        log_system_params=True,  # Track CPU/memory
        measurement_interval=0.1,
        params={'worker_id': worker_id}
    )

    results = []
    for i in range(config.requests_per_worker):
        request_id = worker_id * config.requests_per_worker + i
        result = tracker.track(
            simulate_api_call,
            args=[request_id, config]
        )
        results.append(result)

    return results


def run_throughput_benchmark():
    """
    Run load testing benchmark with different concurrency levels.

    Tracks:
    - Individual requests → 'requests' table
    - Aggregated metrics → 'throughput_metrics' table
    """

    db_path = 'throughput_benchmark.db'

    # Test different concurrency levels
    load_configs = [
        LoadConfig(api_name='api-v1', concurrency=1, requests_per_worker=100),
        LoadConfig(api_name='api-v1', concurrency=4, requests_per_worker=100),
        LoadConfig(api_name='api-v1', concurrency=8, requests_per_worker=100),
        LoadConfig(api_name='api-v1', concurrency=16, requests_per_worker=100),
    ]

    metrics_tracker = Tracker(
        db=db_path,
        engine='duckdb',
        table='throughput_metrics'
    )

    for config in load_configs:
        print(f"\n{'='*60}")
        print(f"Load Test: {config.api_name}")
        print(f"Concurrency: {config.concurrency} workers")
        print(f"Requests per worker: {config.requests_per_worker}")
        print(f"Total requests: {config.concurrency * config.requests_per_worker}")
        print(f"{'='*60}")

        # Run parallel load test
        start_time = time.time()

        with Pool(processes=config.concurrency) as pool:
            worker_func = partial(worker_function, config=config, db_path=db_path)
            all_results = pool.map(worker_func, range(config.concurrency))

        total_time = time.time() - start_time

        # Flatten results
        all_results = [r for worker_results in all_results for r in worker_results]

        # Calculate metrics
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r.get('success', False))
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        latencies = [r['latency'] for r in all_results if r.get('success', False)]
        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            avg_latency = sum(latencies) / len(latencies)
        else:
            p50 = p95 = p99 = avg_latency = 0

        requests_per_sec = total_requests / total_time if total_time > 0 else 0

        # Log aggregated metrics
        metrics_tracker.log({
            'api_name': config.api_name,
            'concurrency': config.concurrency,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': round(success_rate, 4),
            'requests_per_sec': round(requests_per_sec, 2),
            'total_time_sec': round(total_time, 2),
            'avg_latency': round(avg_latency, 4),
            'p50_latency': round(p50, 4),
            'p95_latency': round(p95, 4),
            'p99_latency': round(p99, 4),
        })

        print(f"\n📊 Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/sec: {requests_per_sec:.2f}")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Latency p50: {p50*1000:.1f}ms")
        print(f"  Latency p95: {p95*1000:.1f}ms")
        print(f"  Latency p99: {p99*1000:.1f}ms")

    print("\n" + "="*60)
    print("✅ Throughput benchmark complete!")
    print("\nAnalyze results:")
    print("  # Summary metrics")
    print(f"  python -c \"from xetrack import Reader; print(Reader('{db_path}', table='throughput_metrics', engine='duckdb').to_df())\"")
    print("\n  # Individual request analysis")
    print(f"  python -c \"from xetrack import Reader; df = Reader('{db_path}', table='requests', engine='duckdb').to_df(); print(df.groupby('params_worker_id')['latency'].describe())\"")


if __name__ == "__main__":
    run_throughput_benchmark()
