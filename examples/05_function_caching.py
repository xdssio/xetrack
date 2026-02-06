"""
Example 5: Function Result Caching

Demonstrates:
- Enabling transparent result caching
- Cache hits and misses
- Cache lineage tracking
- Using with different params
- Handling unhashable arguments
"""

from xetrack import Tracker, Reader
from dataclasses import dataclass
import time


def expensive_computation(x: int, y: int) -> int:
    """Simulate expensive computation"""
    time.sleep(0.1)  # Simulate work
    return x ** y


@dataclass(frozen=True)
class Config:
    multiplier: int


def process_with_config(value: int, config: Config) -> int:
    """Function with hashable dataclass config"""
    time.sleep(0.05)
    return value * config.multiplier


def main():
    print("=" * 60)
    print("Example 5: Function Result Caching")
    print("=" * 60)
    
    # Example 1: Basic caching
    print("\n1. Basic function result caching:")
    tracker = Tracker(db='examples_data/cache.db', cache='examples_data/cache_dir')
    
    print("   First call (cache miss):")
    start = time.time()
    result1 = tracker.track(expensive_computation, args=[2, 10])
    time1 = time.time() - start
    print(f"     Result: {result1}, Time: {time1:.3f}s")
    print(f"     Cache status: {tracker.latest.get('cache', '(computed)')}")
    
    print("   Second call with same args (cache hit):")
    start = time.time()
    result2 = tracker.track(expensive_computation, args=[2, 10])
    time2 = time.time() - start
    print(f"     Result: {result2}, Time: {time2:.3f}s")
    print(f"     Cache status: {tracker.latest.get('cache', '(computed)')}")
    print(f"     Speedup: {time1/time2:.1f}x faster")
    
    print("   Third call with different args (cache miss):")
    start = time.time()
    result3 = tracker.track(expensive_computation, args=[3, 10])
    time3 = time.time() - start
    print(f"     Result: {result3}, Time: {time3:.3f}s")
    
    # Example 2: Cache with different params
    print("\n2. Cache with tracker params:")
    tracker_v1 = Tracker(db='examples_data/cache.db', cache='examples_data/cache_dir', 
                         params={'version': 'v1'})
    tracker_v2 = Tracker(db='examples_data/cache.db', cache='examples_data/cache_dir',
                         params={'version': 'v2'})
    
    result_v1 = tracker_v1.track(expensive_computation, args=[2, 10])
    print(f"   v1 result: {result_v1} (computed)")
    
    result_v2 = tracker_v2.track(expensive_computation, args=[2, 10])
    print(f"   v2 result: {result_v2} (computed - different params)")
    
    result_v1_again = tracker_v1.track(expensive_computation, args=[2, 10])
    print(f"   v1 again: {result_v1_again} (cache hit!)")
    
    # Example 3: Cache with frozen dataclass (hashable)
    print("\n3. Caching with hashable dataclass:")
    
    config1 = Config(multiplier=5)
    config2 = Config(multiplier=5)  # Same values, different instance
    
    print("   First call with config1:")
    result1 = tracker.track(process_with_config, args=[10, config1])
    print(f"     Result: {result1}")
    
    print("   Second call with config2 (different instance, same values):")
    result2 = tracker.track(process_with_config, args=[10, config2])
    print(f"     Result: {result2} (cache hit!)")
    print(f"     ✓ Frozen dataclasses with same values share cache")
    
    # Example 4: Cache lineage tracking
    print("\n4. Cache lineage tracking:")
    df = Reader(db='examples_data/cache.db').to_df()
    if 'cache' in df.columns:
        cache_df = df[['function_name', 'function_time', 'cache', 'track_id']].tail(5)
        print("   Recent executions:")
        print(cache_df.to_string(index=False))
        print("\n   Cache field legend:")
        print("     - Empty string: Result was computed (cache miss)")
        print("     - track_id value: Result from cache (references original execution)")
    
    # Example 5: Unhashable arguments
    print("\n5. Unhashable arguments (caching skipped):")
    tracker_warn = Tracker(db='examples_data/cache.db', cache='examples_data/cache_dir',
                          warnings=True)
    
    def process_list(data: list) -> int:
        return sum(data)
    
    print("   Calling with list (unhashable):")
    result = tracker_warn.track(process_list, args=[[1, 2, 3]])
    print(f"     Result: {result}")
    print("     ⚠ Warning issued once, caching skipped for this call")
    
    # Example 6: Reading cache directly
    print("\n6. Cache statistics:")
    cache_count = 0
    try:
        for key, cached_data in Reader.scan_cache('examples_data/cache_dir'):
            cache_count += 1
        print(f"   Total cached results: {cache_count}")
    except Exception:
        print("   Cache directory empty or not accessible")
    
    print("\n" + "=" * 60)
    print("✓ Function caching example complete!")
    print("  Cache directory: examples_data/cache_dir/")
    print("=" * 60)


if __name__ == "__main__":
    main()
