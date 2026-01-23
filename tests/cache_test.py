from xetrack import Tracker, Reader
from tempfile import TemporaryDirectory
import os


def test_cache_basic():
    """Test basic cache functionality"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    # Create tracker with cache
    tracker = Tracker(db=db_path, cache=cache_path)

    # Define a function that will be tracked
    def expensive_function(x: int, y: int) -> int:
        return x * y + x ** y

    # First call - should execute function
    result1 = tracker.track(expensive_function, args=[2, 3])
    assert result1 == 14  # 2*3 + 2**3 = 6 + 8 = 14

    # Second call with same args - should hit cache
    result2 = tracker.track(expensive_function, args=[2, 3])
    assert result2 == 14
    assert result1 == result2

    # Different args - should execute function
    result3 = tracker.track(expensive_function, args=[3, 2])
    assert result3 == 15  # 3*2 + 3**2 = 6 + 9 = 15

    tempdir.cleanup()


def test_cache_with_kwargs():
    """Test cache with keyword arguments"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    # Call with different kwargs
    result1 = tracker.track(greet, args=["Alice"], kwargs={"greeting": "Hi"})
    assert result1 == "Hi, Alice!"

    result2 = tracker.track(greet, args=["Alice"], kwargs={"greeting": "Hi"})
    assert result2 == "Hi, Alice!"

    result3 = tracker.track(greet, args=["Alice"], kwargs={"greeting": "Hello"})
    assert result3 == "Hello, Alice!"

    tempdir.cleanup()


def test_cache_hit_logging():
    """Test that cache field tracks lineage (empty = computed, track_id = cache hit)"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def add(a: int, b: int) -> int:
        return a + b

    # First call - computed
    tracker.track(add, args=[1, 2])

    # Second call - cache hit
    tracker.track(add, args=[1, 2])

    # Check logs
    df = Reader(db_path).to_df()
    assert len(df) == 2

    # First call should have empty cache field (computed)
    assert df.iloc[0]['cache'] == ""

    # Second call should have cache field pointing to first track_id (cache hit)
    assert df.iloc[1]['cache'] == df.iloc[0]['track_id']  # Lineage tracking

    tempdir.cleanup()


def test_reader_read_cache():
    """Test Reader.read_cache() method"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def multiply(x: int, y: int) -> int:
        return x * y

    # Track function to populate cache
    result = tracker.track(multiply, args=[5, 7])
    assert result == 35

    # Get the cache key (need to pass params={} to match the signature)
    cache_key = tracker._generate_cache_key(multiply, [5, 7], {}, {})

    # Read from cache using Reader
    # Cache now stores dict with "result" and "cache" keys
    cached_data = Reader.read_cache(cache_path, cache_key)
    assert cached_data["result"] == 35
    assert cached_data["cache"] == tracker.track_id  # Should match current tracker's track_id

    tempdir.cleanup()


def test_reader_scan_cache():
    """Test Reader.scan_cache() method"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def func1(x: int) -> int:
        return x * 2

    def func2(x: int) -> int:
        return x * 3

    # Track multiple functions
    tracker.track(func1, args=[10])
    tracker.track(func2, args=[10])

    # Scan cache
    cache_entries = list(Reader.scan_cache(cache_path))

    # Should have 2 entries
    assert len(cache_entries) == 2

    # Check that values are correct (cache now stores dicts)
    results = [cached_data["result"] for _, cached_data in cache_entries]
    assert 20 in results  # func1(10) = 20
    assert 30 in results  # func2(10) = 30

    tempdir.cleanup()


def test_cache_with_exception():
    """Test that exceptions are not cached"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path, raise_on_error=False)

    call_count = 0

    def failing_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("First call fails")
        return x * 2

    # First call fails
    result1 = tracker.track(failing_function, args=[5])
    assert result1 is None  # Exception was caught

    # Second call should execute again (not cached)
    result2 = tracker.track(failing_function, args=[5])
    assert result2 == 10

    # Verify function was called twice
    assert call_count == 2

    tempdir.cleanup()


def test_cache_with_different_params():
    """Test that different tracker params create separate cache entries"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def compute(x: int) -> int:
        return x * 2

    # Same args but different params should create different cache entries
    result1 = tracker.track(compute, args=[5], params={"model": "v1"})
    result2 = tracker.track(compute, args=[5], params={"model": "v2"})
    result3 = tracker.track(compute, args=[5], params={"model": "v1"})  # Should hit cache

    assert result1 == 10
    assert result2 == 10
    assert result3 == 10

    # Check logs
    df = Reader(db_path).to_df()
    assert len(df) == 3

    # First and second should be computed (different params)
    assert df.iloc[0]['cache'] == ""
    assert df.iloc[1]['cache'] == ""

    # Third should be cache hit (same params as first)
    assert df.iloc[2]['cache'] == df.iloc[0]['track_id']

    tempdir.cleanup()


def test_cache_disabled():
    """Test that tracking works without cache and cache field is not present"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test.db")

    # Create tracker without cache
    tracker = Tracker(db=db_path)

    def simple_func(x: int) -> int:
        return x * 2

    result1 = tracker.track(simple_func, args=[5])
    result2 = tracker.track(simple_func, args=[5])

    assert result1 == 10
    assert result2 == 10

    # Cache field should not be present when cache is disabled
    df = Reader(db_path).to_df()
    assert len(df) == 2
    assert 'cache' not in df.columns

    tempdir.cleanup()
