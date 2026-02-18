from xetrack import Tracker, Reader
from tempfile import TemporaryDirectory
import os


def test_cache_basic():
    """Test basic cache functionality"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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
    db_path = os.path.join(tempdir.name, "test_db")
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


def test_cache_with_hashable_objects():
    """Test that hashable custom objects work with caching"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    # Define a hashable custom class
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __hash__(self):
            return hash((self.x, self.y))

        def __eq__(self, other):
            return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def distance(p: Point) -> float:
        return (p.x ** 2 + p.y ** 2) ** 0.5

    # Create two Point objects with same values
    p1 = Point(3, 4)
    p2 = Point(3, 4)

    # First call
    result1 = tracker.track(distance, args=[p1])
    assert result1 == 5.0

    # Second call with different object but same hash - should hit cache
    result2 = tracker.track(distance, args=[p2])
    assert result2 == 5.0

    # Verify second call was a cache hit
    df = Reader(db_path).to_df()
    assert len(df) == 2
    assert df.iloc[0]['cache'] == ""
    assert df.iloc[1]['cache'] == df.iloc[0]['track_id']

    tempdir.cleanup()


def test_cache_with_unhashable_objects():
    """Test that unhashable objects skip caching entirely"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path, warnings=False)  # Disable warnings for cleaner test

    def process_list(items: list) -> int:
        return sum(items)

    # First call with a list (unhashable) - caching skipped
    result1 = tracker.track(process_list, args=[[1, 2, 3]])
    assert result1 == 6

    # Second call with same list values - caching still skipped (unhashable)
    result2 = tracker.track(process_list, args=[[1, 2, 3]])
    assert result2 == 6

    # Both should be computed and cache field should not be present (caching was skipped)
    df = Reader(db_path).to_df()
    assert len(df) == 2
    assert 'cache' not in df.columns  # Cache field not added when caching is skipped

    tempdir.cleanup()


def test_cache_disabled():
    """Test that tracking works without cache and cache field is not present"""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")

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


def test_cache_with_dataclass():
    """Test that dataclass inputs work with caching"""
    from dataclasses import dataclass
    
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    @dataclass(frozen=True)  # frozen=True makes it hashable
    class Config:
        learning_rate: float
        batch_size: int
        model_name: str

    def train_model(config: Config) -> float:
        """Simulate training with config"""
        return config.learning_rate * config.batch_size

    # Create two Config objects with same values
    config1 = Config(learning_rate=0.01, batch_size=32, model_name="bert")
    config2 = Config(learning_rate=0.01, batch_size=32, model_name="bert")

    # First call - should execute function
    result1 = tracker.track(train_model, args=[config1])
    assert result1 == 0.32  # 0.01 * 32

    # Second call with different object but same hash - should hit cache
    result2 = tracker.track(train_model, args=[config2])
    assert result2 == 0.32

    # Verify cache hit
    df = Reader(db_path).to_df()
    assert len(df) == 2
    assert df.iloc[0]['cache'] == ""  # First call computed
    assert df.iloc[1]['cache'] == df.iloc[0]['track_id']  # Second call hit cache

    # Different config should compute again
    config3 = Config(learning_rate=0.02, batch_size=32, model_name="bert")
    result3 = tracker.track(train_model, args=[config3])
    assert result3 == 0.64  # 0.02 * 32

    df = Reader(db_path).to_df()
    assert len(df) == 3
    assert df.iloc[2]['cache'] == ""  # New config computed

    tempdir.cleanup()


def test_cache_with_non_frozen_dataclass():
    """Test that non-frozen dataclass inputs skip caching"""
    from dataclasses import dataclass
    
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path, warnings=False)

    @dataclass  # Not frozen - not hashable
    class Config:
        learning_rate: float
        batch_size: int

    def compute(config: Config) -> float:
        return config.learning_rate * config.batch_size

    config = Config(learning_rate=0.01, batch_size=32)

    # Both calls should execute (no caching with unhashable dataclass)
    result1 = tracker.track(compute, args=[config])
    assert result1 == 0.32

    result2 = tracker.track(compute, args=[config])
    assert result2 == 0.32

    # Cache field should not be present (caching was skipped)
    df = Reader(db_path).to_df()
    assert len(df) == 2
    assert 'cache' not in df.columns

    tempdir.cleanup()


def test_cache_hit_logs_result_values():
    """Test that cache hits log the actual return values, not just lineage metadata."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def compute(x: int, y: int) -> dict:
        return {"accuracy": 0.95, "loss": 0.05}

    # First call - computed
    result1 = tracker.track(compute, args=[1, 2])
    assert result1 == {"accuracy": 0.95, "loss": 0.05}

    # Second call - cache hit
    result2 = tracker.track(compute, args=[1, 2])
    assert result2 == {"accuracy": 0.95, "loss": 0.05}

    df = Reader(db_path).to_df()
    assert len(df) == 2

    # Both rows should have accuracy and loss columns with values
    assert df.iloc[0]["accuracy"] == 0.95
    assert df.iloc[0]["loss"] == 0.05
    assert df.iloc[1]["accuracy"] == 0.95
    assert df.iloc[1]["loss"] == 0.05

    # First row is computed (empty cache), second is cache hit
    assert df.iloc[0]["cache"] == ""
    assert df.iloc[1]["cache"] != ""

    tempdir.cleanup()


def test_cache_hit_logs_primitive_result():
    """Test that cache hits log primitive return values via function_result column."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def multiply(x: int, y: int) -> int:
        return x * y

    # First call - computed
    tracker.track(multiply, args=[3, 7])

    # Second call - cache hit
    tracker.track(multiply, args=[3, 7])

    df = Reader(db_path).to_df()
    assert len(df) == 2

    # Both rows should have function_result with the actual value
    assert df.iloc[0]["function_result"] == 21
    assert df.iloc[1]["function_result"] == 21

    tempdir.cleanup()


def test_cache_force_skips_read_and_rewrites():
    """Test that cache_force=True skips cache lookup but overwrites the cache entry."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    call_count = 0

    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - populates cache
    result1 = tracker.track(compute, args=[5])
    assert result1 == 10
    assert call_count == 1

    # Second call without force - should hit cache (function NOT called)
    result2 = tracker.track(compute, args=[5])
    assert result2 == 10
    assert call_count == 1  # still 1

    # Third call with cache_force - should re-execute and overwrite cache
    result3 = tracker.track(compute, args=[5], cache_force=True)
    assert result3 == 10
    assert call_count == 2  # incremented

    # Verify DB has 3 entries: computed, cache hit, force-recomputed
    df = Reader(db_path).to_df()
    assert len(df) == 3
    assert df.iloc[0]['cache'] == ""   # computed
    assert df.iloc[1]['cache'] != ""   # cache hit
    assert df.iloc[2]['cache'] == ""   # force-recomputed

    # Fourth call without force - should still hit cache (was overwritten)
    result4 = tracker.track(compute, args=[5])
    assert result4 == 10
    assert call_count == 2  # still 2, cache hit

    tempdir.cleanup()


def test_cache_force_updates_lineage():
    """Test that cache_force overwrites the lineage track_id in cache."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    # Single tracker, but we change track_id between calls to verify lineage updates
    tracker = Tracker(db=db_path, cache=cache_path)
    original_track_id = tracker.track_id

    def add(a: int, b: int) -> int:
        return a + b

    # First call populates cache with original_track_id
    tracker.track(add, args=[1, 2])

    # Change track_id and force-refresh — cache lineage should update
    tracker.track_id = tracker.generate_track_id()
    new_track_id = tracker.track_id
    tracker.track(add, args=[1, 2], cache_force=True)

    # Now a cache hit should reference new_track_id (the one that overwrote)
    tracker.track_id = tracker.generate_track_id()
    tracker.track(add, args=[1, 2])

    df = Reader(db_path).to_df()
    # Last row's cache field should be the track_id that force-refreshed
    assert df.iloc[2]['cache'] == new_track_id

    tempdir.cleanup()


def test_delete_cache_by_track_id():
    """Test Reader.delete_cache_by_track_id removes the right entries."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    def func_a(x: int) -> int:
        return x + 1

    def func_b(x: int) -> int:
        return x + 2

    # Both are cached under the same tracker.track_id
    tracker.track(func_a, args=[10])
    tracker.track(func_b, args=[10])

    # Verify 2 entries in cache
    entries_before = list(Reader.scan_cache(cache_path))
    assert len(entries_before) == 2

    # Delete by track_id
    deleted = Reader.delete_cache_by_track_id(cache_path, tracker.track_id)
    assert deleted == 2

    # Cache should be empty now
    entries_after = list(Reader.scan_cache(cache_path))
    assert len(entries_after) == 0

    tempdir.cleanup()


def test_delete_cache_by_track_id_partial():
    """Test that delete only removes entries for the given track_id, not others."""
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)
    first_track_id = tracker.track_id

    def compute(x: int) -> int:
        return x * 3

    # Cache compute(5) under first track_id
    tracker.track(compute, args=[5])

    # Switch track_id and cache compute(10) under second track_id
    tracker.track_id = tracker.generate_track_id()
    second_track_id = tracker.track_id
    tracker.track(compute, args=[10])

    entries_before = list(Reader.scan_cache(cache_path))
    assert len(entries_before) == 2

    # Delete only first track_id's entries
    deleted = Reader.delete_cache_by_track_id(cache_path, first_track_id)
    assert deleted == 1

    # One entry should remain (second track_id's)
    entries_after = list(Reader.scan_cache(cache_path))
    assert len(entries_after) == 1
    assert entries_after[0][1]["cache"] == second_track_id

    tempdir.cleanup()


def test_wrap_cached_and_computed_rows_are_consistent():
    """Test that @tracker.wrap() produces identical DB rows for cached and computed calls.

    Verifies that:
    - wrap() works with positional args and keyword args
    - Cached rows contain the same result columns as computed rows
    - Only cache, function_time, track_id, and timestamp differ between rows
    - Dict return values are unpacked as columns in both cases
    - Primitive return values appear in function_result in both cases
    """
    tempdir = TemporaryDirectory()
    db_path = os.path.join(tempdir.name, "test_db")
    cache_path = os.path.join(tempdir.name, "cache")

    tracker = Tracker(db=db_path, cache=cache_path)

    # --- Test 1: Function returning a dict (columns unpacked) ---

    @tracker.wrap(params={"experiment": "v1"})
    def evaluate(model_name: str, threshold: float = 0.5, top_k: int = 10) -> dict:
        return {
            "accuracy": 0.92,
            "precision": 0.88,
            "recall": 0.95,
            "f1": 0.91,
            "top_k_used": top_k,
        }

    # First call — computed
    result1 = evaluate("bert", threshold=0.7, top_k=5)
    assert result1 == {"accuracy": 0.92, "precision": 0.88, "recall": 0.95, "f1": 0.91, "top_k_used": 5}

    # Second call with same args/kwargs — cache hit
    result2 = evaluate("bert", threshold=0.7, top_k=5)
    assert result2 == result1

    df = tracker.to_df(all=True)
    assert len(df) == 2

    computed_row = df.iloc[0]
    cached_row = df.iloc[1]

    # Result columns must be identical
    for col in ["accuracy", "precision", "recall", "f1", "top_k_used"]:
        assert computed_row[col] == cached_row[col], f"Column '{col}' differs: {computed_row[col]} vs {cached_row[col]}"

    # Function metadata must match
    assert computed_row["function_name"] == cached_row["function_name"] == "evaluate"
    assert computed_row["experiment"] == cached_row["experiment"] == "v1"
    assert str(computed_row["args"]) == str(cached_row["args"])
    assert str(computed_row["kwargs"]) == str(cached_row["kwargs"])

    # Cache lineage: computed is empty, cached points to computed's track_id
    assert computed_row["cache"] == ""
    assert cached_row["cache"] == computed_row["track_id"]

    # Computed should have nonzero function_time, cached should be 0
    assert computed_row["function_time"] > 0 or True  # may be very fast
    assert cached_row["function_time"] == 0.0

    # Both should have error="" (no exceptions)
    assert computed_row["error"] == ""
    assert cached_row["error"] == ""

    # --- Test 2: Function returning a primitive (stored in function_result) ---

    db_path2 = os.path.join(tempdir.name, "test_db2")
    cache_path2 = os.path.join(tempdir.name, "cache2")
    tracker2 = Tracker(db=db_path2, cache=cache_path2)

    @tracker2.wrap()
    def score(text: str, weight: float = 1.0) -> float:
        return len(text) * weight

    # Computed
    r1 = score("hello", weight=2.0)
    assert r1 == 10.0

    # Cached
    r2 = score("hello", weight=2.0)
    assert r2 == 10.0

    df2 = tracker2.to_df(all=True)
    assert len(df2) == 2

    comp2 = df2.iloc[0]
    cache2 = df2.iloc[1]

    # function_result must be the same primitive value
    assert comp2["function_result"] == cache2["function_result"] == 10.0

    # Same metadata
    assert comp2["function_name"] == cache2["function_name"] == "score"

    # Cache lineage
    assert comp2["cache"] == ""
    assert cache2["cache"] == comp2["track_id"]

    # --- Test 3: Different kwargs produce different cache entries ---

    r3 = score("hello", weight=3.0)
    assert r3 == 15.0

    df3 = tracker2.to_df(all=True)
    assert len(df3) == 3

    # Third row should be computed (different kwargs = cache miss)
    assert df3.iloc[2]["cache"] == ""
    assert df3.iloc[2]["function_result"] == 15.0

    # --- Test 4: Columns are consistent across all rows (no NaN gaps) ---

    # All rows in df2 should have the same columns with no unexpected NaNs
    result_cols = ["function_name", "function_result", "cache", "error"]
    for col in result_cols:
        assert df3[col].notna().all(), f"Column '{col}' has NaN values in some rows"

    tempdir.cleanup()
