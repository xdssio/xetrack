from dataclasses import dataclass
from xetrack import Tracker


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int


def test_track_with_frozen_dataclass():
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @tracker.wrap()
    def train(config: TrainingConfig):
        return {"accuracy": 0.95}
    
    config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)
    result = train(config)
    
    assert result == {"accuracy": 0.95}
    assert tracker.latest["config_learning_rate"] == 0.001
    assert tracker.latest["config_batch_size"] == 32
    assert tracker.latest["config_epochs"] == 10
    assert tracker.latest["accuracy"] == 0.95


def test_track_with_dataclass_and_kwargs():
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class ModelConfig:
        model_name: str
        dropout: float
    
    @tracker.wrap()
    def evaluate(config: ModelConfig, dataset: str):
        return {"f1_score": 0.88, "dataset": dataset}
    
    config = ModelConfig(model_name="bert", dropout=0.1)
    result = evaluate(config, dataset="test_set")
    
    assert result["f1_score"] == 0.88
    assert tracker.latest["config_model_name"] == "bert"
    assert tracker.latest["config_dropout"] == 0.1
    assert tracker.latest["f1_score"] == 0.88
    assert tracker.latest["dataset"] == "test_set"


def test_track_with_positional_dataclass():
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class DataConfig:
        train_size: float
        val_size: float
    
    def process_data(data_cfg: DataConfig):
        return {"processed": True}
    
    config = DataConfig(train_size=0.8, val_size=0.2)
    result = tracker.track(process_data, args=[config])
    
    assert result == {"processed": True}
    assert tracker.latest["data_cfg_train_size"] == 0.8
    assert tracker.latest["data_cfg_val_size"] == 0.2
    assert tracker.latest["processed"] == True


def test_track_with_multiple_dataclasses():
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class ModelConfig:
        layers: int
        units: int
    
    @dataclass(frozen=True)
    class TrainConfig:
        lr: float
        epochs: int
    
    def train_model(model_cfg: ModelConfig, train_cfg: TrainConfig):
        return {"loss": 0.05}
    
    model_config = ModelConfig(layers=3, units=128)
    train_config = TrainConfig(lr=0.001, epochs=50)
    
    result = tracker.track(train_model, args=[model_config, train_config])
    
    assert result == {"loss": 0.05}
    assert tracker.latest["model_cfg_layers"] == 3
    assert tracker.latest["model_cfg_units"] == 128
    assert tracker.latest["train_cfg_lr"] == 0.001
    assert tracker.latest["train_cfg_epochs"] == 50
    assert tracker.latest["loss"] == 0.05


def test_track_with_pydantic_basemodel():
    """Test with Pydantic BaseModel if pydantic is available"""
    try:
        from pydantic import BaseModel
    except ImportError:
        # Skip test if pydantic not installed
        return
    
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    class ExperimentConfig(BaseModel):
        experiment_name: str
        seed: int
        use_gpu: bool = True
    
    @tracker.wrap()
    def run_experiment(cfg: ExperimentConfig):
        return {"status": "completed"}
    
    config = ExperimentConfig(experiment_name="exp_001", seed=42, use_gpu=True)
    result = run_experiment(config)
    
    assert result == {"status": "completed"}
    assert tracker.latest["cfg_experiment_name"] == "exp_001"
    assert tracker.latest["cfg_seed"] == 42
    assert tracker.latest["cfg_use_gpu"] == True
    assert tracker.latest["status"] == "completed"


def test_track_mixed_dataclass_and_basemodel():
    """Test with both dataclass and BaseModel if pydantic is available"""
    try:
        from pydantic import BaseModel
    except ImportError:
        # Skip test if pydantic not installed
        return
    
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class DataConfig:
        batch_size: int
    
    class ModelConfig(BaseModel):
        model_type: str
    
    def train(d: DataConfig, m: ModelConfig):
        return {"result": "success"}
    
    data_cfg = DataConfig(batch_size=64)
    model_cfg = ModelConfig(model_type="transformer")
    
    result = tracker.track(train, args=[data_cfg, model_cfg])
    
    assert result == {"result": "success"}
    assert tracker.latest["d_batch_size"] == 64
    assert tracker.latest["m_model_type"] == "transformer"
    assert tracker.latest["result"] == "success"


def test_track_non_frozen_dataclass():
    """Test that non-frozen dataclasses also work"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass  # Not frozen
    class Config:
        value: int
    
    def process(cfg: Config):
        return {"output": cfg.value * 2}
    
    config = Config(value=5)
    result = tracker.track(process, args=[config])
    
    assert result["output"] == 10
    assert tracker.latest["cfg_value"] == 5
    assert tracker.latest["output"] == 10


def test_track_dataclass_with_complex_types():
    """Test dataclass with nested structures"""
    tracker = Tracker(db=Tracker.IN_MEMORY)
    
    @dataclass(frozen=True)
    class Config:
        name: str
        values: tuple  # Complex type - will be stored as string
    
    def process(cfg: Config):
        return {"count": len(cfg.values)}
    
    config = Config(name="test", values=(1, 2, 3))
    result = tracker.track(process, args=[config])
    
    assert result["count"] == 3
    assert tracker.latest["cfg_name"] == "test"
    # tuple will be converted to string representation during validation
    assert "cfg_values" in tracker.latest
