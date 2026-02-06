"""
Example 2: Track Functions

Demonstrates:
- Using tracker.track() to monitor function execution
- Using @tracker.wrap() decorator
- Tracking with system and network params
"""

from xetrack import Tracker
import time


def expensive_computation(n: int) -> int:
    """Simulate some work"""
    time.sleep(0.1)
    return sum(range(n))


def read_image(filename: str) -> dict:
    """Simulate reading an image"""
    time.sleep(0.05)
    return {"width": 1920, "height": 1080, "size": 571084}


def main():
    print("=" * 60)
    print("Example 2: Track Functions")
    print("=" * 60)
    
    # Basic function tracking
    print("\n1. Basic function tracking:")
    tracker = Tracker('examples_data/functions.db')
    result = tracker.track(expensive_computation, args=[1000])
    print(f"   Result: {result}")
    print(f"   Execution time: {tracker.latest['function_time']:.4f}s")
    print(f"   Function name: {tracker.latest['function_name']}")
    
    # Track with system monitoring
    print("\n2. Track with system monitoring:")
    tracker_sys = Tracker(
        'examples_data/functions.db',
        log_system_params=True,
        log_network_params=True,
        measurement_interval=0.05
    )
    
    image_data = tracker_sys.track(read_image, args=['photo.jpg'])
    print(f"   Image size: {image_data['size']} bytes")
    print(f"   CPU usage: {tracker_sys.latest.get('cpu_percent', 'N/A')}%")
    print(f"   Memory: {tracker_sys.latest.get('p_memory_percent', 'N/A')}%")
    print(f"   Network sent: {tracker_sys.latest.get('bytes_sent', 'N/A')} MB")
    
    # Using decorator
    print("\n3. Using @tracker.wrap() decorator:")
    tracker_wrap = Tracker('examples_data/functions.db')
    
    @tracker_wrap.wrap(params={'name': 'my_function'})
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    result = add_numbers(5, 3)
    print(f"   Result: {result}")
    print(f"   Function name: {tracker_wrap.latest['function_name']}")
    print(f"   Custom param: {tracker_wrap.latest['name']}")
    print(f"   Args: {tracker_wrap.latest['args']}")
    
    print("\n" + "=" * 60)
    print("✓ Function tracking example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
