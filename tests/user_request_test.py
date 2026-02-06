"""
Test case demonstrating the exact use case from the user's request.
"""
from dataclasses import dataclass
from xetrack import Tracker


def test_user_requested_behavior():
    """
    Test the exact behavior requested by the user:
    
    @track()
    def function(d:dataclass, b:BaseModel=<>)
       return {"answer":1}
    
    → {"answer":1, "d_a":<>,"d_b":<>,"b_c":<>,...}
    """
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class DataConfig:
        a: int
        b: str
    
    # Test with just dataclass (simulating the user's example)
    @tracker.wrap()
    def function(d: DataConfig):
        return {"answer": 1}
    
    d = DataConfig(a=42, b="hello")
    result = function(d)
    
    # Verify the result contains the answer
    assert result == {"answer": 1}
    
    # Verify all fields are unpacked with dot notation
    assert tracker.latest["answer"] == 1
    assert tracker.latest["d_a"] == 42
    assert tracker.latest["d_b"] == "hello"
    
    # Verify the full structure matches the user's expectation
    expected_keys = {"answer", "d_a", "d_b", "function_name", "args", "kwargs", 
                     "error", "function_time", "timestamp", "track_id"}
    actual_keys = set(tracker.latest.keys())
    assert expected_keys.issubset(actual_keys)
    
    print("✓ User's requested behavior works correctly!")
    print(f"  Result: {result}")
    print(f"  Tracked: answer={tracker.latest['answer']}, d.a={tracker.latest['d_a']}, d.b={tracker.latest['d_b']}")


def test_user_requested_behavior_with_basemodel():
    """
    Test with both dataclass and BaseModel as in the user's example.
    """
    try:
        from pydantic import BaseModel
    except ImportError:
        print("⚠ Skipping BaseModel test (pydantic not installed)")
        return
    
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class DataConfig:
        a: int
        b: str
    
    class ModelConfig(BaseModel):
        c: float
        d: bool = True
    
    @tracker.wrap()
    def function(d: DataConfig, b: ModelConfig):
        return {"answer": 1}
    
    d = DataConfig(a=42, b="hello")
    b = ModelConfig(c=3.14, d=False)
    
    result = function(d, b)
    
    # Verify the result
    assert result == {"answer": 1}
    
    # Verify all fields are unpacked
    assert tracker.latest["answer"] == 1
    assert tracker.latest["d_a"] == 42
    assert tracker.latest["d_b"] == "hello"
    assert tracker.latest["b_c"] == 3.14
    assert tracker.latest["b_d"] == False
    
    print("✓ User's requested behavior with BaseModel works correctly!")
    print(f"  Result: {result}")
    print(f"  Tracked: answer={tracker.latest['answer']}, d.a={tracker.latest['d_a']}, "
          f"d.b={tracker.latest['d_b']}, b.c={tracker.latest['b_c']}, b.d={tracker.latest['b_d']}")


if __name__ == "__main__":
    test_user_requested_behavior()
    test_user_requested_behavior_with_basemodel()
    print("\n✅ All tests passed! Feature implemented as requested.")
