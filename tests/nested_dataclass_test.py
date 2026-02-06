"""
Test nested dataclasses and non-primitive fields
"""
from dataclasses import dataclass
from xetrack import Tracker


def test_nested_dataclass():
    """Test that nested dataclasses are recursively unpacked"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class OptimizerConfig:
        lr: float
        momentum: float
    
    @dataclass(frozen=True)
    class TrainingConfig:
        batch_size: int
        epochs: int
        optimizer: OptimizerConfig
    
    @tracker.wrap()
    def train(config: TrainingConfig):
        return {"accuracy": 0.95}
    
    optimizer = OptimizerConfig(lr=0.001, momentum=0.9)
    config = TrainingConfig(batch_size=32, epochs=10, optimizer=optimizer)
    
    result = train(config)
    
    assert result == {"accuracy": 0.95}
    # Flat fields
    assert tracker.latest["config_batch_size"] == 32
    assert tracker.latest["config_epochs"] == 10
    # Nested fields - should be unpacked recursively
    assert tracker.latest["config_optimizer_lr"] == 0.001
    assert tracker.latest["config_optimizer_momentum"] == 0.9
    assert tracker.latest["accuracy"] == 0.95
    
    print("✓ Nested dataclass unpacking works!")
    print(f"  Tracked keys: {[k for k in tracker.latest.keys() if k.startswith('config')]}")


def test_dataclass_with_list_field():
    """Test dataclass with list fields - lists should be converted to strings"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class Config:
        name: str
        values: list
    
    @tracker.wrap()
    def process(cfg: Config):
        return {"count": len(cfg.values)}
    
    config = Config(name="test", values=[1, 2, 3, 4, 5])
    result = process(config)
    
    assert result["count"] == 5
    assert tracker.latest["cfg_name"] == "test"
    # List should be converted to string by _validate_data
    assert "cfg_values" in tracker.latest
    # The list will be stringified, so it should contain the values
    assert "1" in str(tracker.latest["cfg_values"])
    
    print("✓ List fields are handled correctly!")


def test_deeply_nested_dataclass():
    """Test deeply nested dataclass structures"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class LearningRate:
        initial: float
        decay: float
    
    @dataclass(frozen=True)
    class OptimizerConfig:
        name: str
        lr_config: LearningRate
    
    @dataclass(frozen=True)
    class ModelConfig:
        layers: int
        optimizer: OptimizerConfig
    
    @tracker.wrap()
    def build_model(cfg: ModelConfig):
        return {"status": "built"}
    
    lr = LearningRate(initial=0.001, decay=0.95)
    optimizer = OptimizerConfig(name="adam", lr_config=lr)
    config = ModelConfig(layers=12, optimizer=optimizer)
    
    result = build_model(config)
    
    assert result["status"] == "built"
    assert tracker.latest["cfg_layers"] == 12
    assert tracker.latest["cfg_optimizer_name"] == "adam"
    assert tracker.latest["cfg_optimizer_lr_config_initial"] == 0.001
    assert tracker.latest["cfg_optimizer_lr_config_decay"] == 0.95
    
    print("✓ Deeply nested dataclasses work!")
    print(f"  Deepest field: cfg_optimizer_lr_config_initial = {tracker.latest['cfg_optimizer_lr_config_initial']}")


def test_mixed_nested_pydantic_dataclass():
    """Test nested mix of dataclass and Pydantic BaseModel"""
    try:
        from pydantic import BaseModel
    except ImportError:
        print("⚠ Skipping nested Pydantic test (pydantic not installed)")
        return
    
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class DatasetConfig:
        name: str
        size: int
    
    class TrainingConfig(BaseModel):
        batch_size: int
        dataset: DatasetConfig
    
    @tracker.wrap()
    def train(cfg: TrainingConfig):
        return {"done": True}
    
    dataset = DatasetConfig(name="cifar10", size=50000)
    config = TrainingConfig(batch_size=32, dataset=dataset)
    
    result = train(config)
    
    assert result["done"] == True
    assert tracker.latest["cfg_batch_size"] == 32
    # This should recursively unpack the dataclass inside Pydantic
    assert tracker.latest["cfg_dataset_name"] == "cifar10"
    assert tracker.latest["cfg_dataset_size"] == 50000
    
    print("✓ Nested Pydantic + dataclass works!")


if __name__ == "__main__":
    test_nested_dataclass()
    test_dataclass_with_list_field()
    test_deeply_nested_dataclass()
    test_mixed_nested_pydantic_dataclass()
    print("\n✅ All nested dataclass tests passed!")
