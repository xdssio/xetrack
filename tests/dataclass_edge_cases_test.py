"""
Test edge cases for dataclass extraction
"""
from dataclasses import dataclass
from typing import Optional
from xetrack import Tracker


def test_circular_reference_protection():
    """Test that circular references don't cause infinite recursion"""
    tracker = Tracker(db=Tracker.IN_MEMORY, warnings=False)
    
    @dataclass
    class Node:
        value: int
        next: Optional['Node'] = None
    
    @tracker.wrap()
    def process(node: Node):
        return {"result": "ok"}
    
    # Create circular reference
    node1 = Node(value=1)
    node2 = Node(value=2, next=node1)
    node1.next = node2  # Circular!
    
    # This should NOT crash
    result = process(node1)
    
    assert result == {"result": "ok"}
    # Should have extracted at least the top-level fields
    assert tracker.latest["node_value"] == 1
    print("✓ Circular reference handled gracefully")


def test_depth_limit():
    """Test that deeply nested structures are limited"""
    tracker = Tracker(db=Tracker.IN_MEMORY, warnings=False)
    
    @dataclass
    class Level:
        value: int
        nested: Optional['Level'] = None
    
    @tracker.wrap()
    def process(cfg: Level):
        return {"done": True}
    
    # Create 15 levels of nesting (exceeds MAX_DEPTH of 10)
    current = Level(value=15)
    for i in range(14, 0, -1):
        current = Level(value=i, nested=current)
    
    result = process(current)
    
    assert result == {"done": True}
    # Should have top level
    assert tracker.latest["cfg_value"] == 1
    # Should have some nested levels
    assert "cfg_nested_value" in tracker.latest
    print("✓ Depth limit prevents stack overflow")


def test_none_fields():
    """Test that None fields are handled correctly"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class Config:
        name: str
        optimizer: Optional['OptimizerConfig'] = None
    
    @dataclass(frozen=True)
    class OptimizerConfig:
        lr: float
    
    @tracker.wrap()
    def train(cfg: Config):
        return {"accuracy": 0.9}
    
    config = Config(name="test", optimizer=None)
    result = train(config)
    
    assert result == {"accuracy": 0.9}
    assert tracker.latest["cfg_name"] == "test"
    assert tracker.latest["cfg_optimizer"] is None
    print("✓ None fields handled correctly")


def test_pydantic_nested_in_dataclass():
    """Test Pydantic model nested in dataclass"""
    try:
        from pydantic import BaseModel
    except ImportError:
        print("⚠ Skipping Pydantic test")
        return
    
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    class OptimizerConfig(BaseModel):
        lr: float
        momentum: float
    
    @dataclass(frozen=True)
    class TrainingConfig:
        batch_size: int
        optimizer: OptimizerConfig
    
    @tracker.wrap()
    def train(cfg: TrainingConfig):
        return {"loss": 0.05}
    
    optimizer = OptimizerConfig(lr=0.001, momentum=0.9)
    config = TrainingConfig(batch_size=32, optimizer=optimizer)
    
    result = train(config)
    
    assert result == {"loss": 0.05}
    assert tracker.latest["cfg_batch_size"] == 32
    # Nested Pydantic should be unpacked
    assert tracker.latest["cfg_optimizer_lr"] == 0.001
    assert tracker.latest["cfg_optimizer_momentum"] == 0.9
    print("✓ Pydantic nested in dataclass works")


def test_very_wide_dataclass():
    """Test dataclass with many fields (not deep)"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class WideConfig:
        f1: int = 1
        f2: int = 2
        f3: int = 3
        f4: int = 4
        f5: int = 5
        f6: int = 6
        f7: int = 7
        f8: int = 8
        f9: int = 9
        f10: int = 10
    
    @tracker.wrap()
    def process(cfg: WideConfig):
        return {"sum": sum([cfg.f1, cfg.f2, cfg.f3, cfg.f4, cfg.f5,
                           cfg.f6, cfg.f7, cfg.f8, cfg.f9, cfg.f10])}
    
    config = WideConfig()
    result = process(config)
    
    assert result["sum"] == 55
    # All fields should be extracted
    for i in range(1, 11):
        assert tracker.latest[f"cfg_f{i}"] == i
    print("✓ Wide dataclass (many fields) works")


if __name__ == "__main__":
    test_circular_reference_protection()
    test_depth_limit()
    test_none_fields()
    test_pydantic_nested_in_dataclass()
    test_very_wide_dataclass()
    print("\n✅ All edge case tests passed!")
