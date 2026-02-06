"""
Example 3: Dataclass and Pydantic Auto-Unpacking

Demonstrates:
- Automatic unpacking of frozen dataclasses
- Pydantic BaseModel support
- Multiple dataclasses
- Nested structures
"""

from dataclasses import dataclass
from xetrack import Tracker


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    num_layers: int
    hidden_size: int


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    momentum: float


@dataclass(frozen=True)
class NestedConfig:
    batch_size: int
    optimizer: OptimizerConfig


def main():
    print("=" * 60)
    print("Example 3: Dataclass Auto-Unpacking")
    print("=" * 60)
    
    tracker = Tracker('examples_data/dataclass.db')
    
    # Example 1: Single dataclass
    print("\n1. Single dataclass unpacking:")
    
    @tracker.wrap()
    def train_model(config: TrainingConfig):
        return {"accuracy": 0.95, "loss": 0.05}
    
    config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)
    result = train_model(config)
    
    print(f"   Result: {result}")
    print(f"   Config fields extracted:")
    print(f"     - config_learning_rate: {tracker.latest['config_learning_rate']}")
    print(f"     - config_batch_size: {tracker.latest['config_batch_size']}")
    print(f"     - config_epochs: {tracker.latest['config_epochs']}")
    print(f"     - config_optimizer: {tracker.latest['config_optimizer']}")
    
    # Example 2: Multiple dataclasses
    print("\n2. Multiple dataclasses:")
    
    def evaluate_model(model_cfg: ModelConfig, train_cfg: TrainingConfig):
        return {"test_accuracy": 0.93}
    
    model_config = ModelConfig(model_type="transformer", num_layers=12, hidden_size=768)
    train_config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)
    
    result = tracker.track(evaluate_model, args=[model_config, train_config])
    
    print(f"   Result: {result}")
    print(f"   Model config:")
    print(f"     - model_cfg_model_type: {tracker.latest['model_cfg_model_type']}")
    print(f"     - model_cfg_num_layers: {tracker.latest['model_cfg_num_layers']}")
    print(f"   Training config:")
    print(f"     - train_cfg_learning_rate: {tracker.latest['train_cfg_learning_rate']}")
    print(f"     - train_cfg_batch_size: {tracker.latest['train_cfg_batch_size']}")
    
    # Example 3: Nested dataclasses
    print("\n3. Nested dataclasses (recursive unpacking):")
    
    @tracker.wrap()
    def train_with_optimizer(cfg: NestedConfig):
        return {"loss": 0.04}
    
    optimizer = OptimizerConfig(lr=0.001, momentum=0.9)
    nested_config = NestedConfig(batch_size=64, optimizer=optimizer)
    
    result = train_with_optimizer(nested_config)
    
    print(f"   Result: {result}")
    print(f"   Flat fields:")
    print(f"     - cfg_batch_size: {tracker.latest['cfg_batch_size']}")
    print(f"   Nested fields (flattened):")
    print(f"     - cfg_optimizer_lr: {tracker.latest['cfg_optimizer_lr']}")
    print(f"     - cfg_optimizer_momentum: {tracker.latest['cfg_optimizer_momentum']}")
    
    # Example 4: Pydantic (if available)
    print("\n4. Pydantic BaseModel (if available):")
    try:
        from pydantic import BaseModel
        
        class ExperimentConfig(BaseModel):
            experiment_name: str
            seed: int
            use_gpu: bool = True
        
        @tracker.wrap()
        def run_experiment(cfg: ExperimentConfig):
            return {"status": "completed"}
        
        pydantic_config = ExperimentConfig(experiment_name="exp_001", seed=42)
        result = run_experiment(pydantic_config)
        
        print(f"   Result: {result}")
        print(f"   Pydantic fields extracted:")
        print(f"     - cfg_experiment_name: {tracker.latest['cfg_experiment_name']}")
        print(f"     - cfg_seed: {tracker.latest['cfg_seed']}")
        print(f"     - cfg_use_gpu: {tracker.latest['cfg_use_gpu']}")
        
    except ImportError:
        print("   ⚠ Pydantic not installed, skipping Pydantic example")
    
    print("\n" + "=" * 60)
    print("✓ Dataclass auto-unpacking example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
