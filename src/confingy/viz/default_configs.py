"""Default example configurations for the visualization server."""

from dataclasses import dataclass
from typing import Optional

from confingy import lazy, serialize_fingy, track


# Example classes for baseline and updated configs
@track
class DataProcessorV1:
    def __init__(self, batch_size: int, normalize: bool = True):
        self.batch_size = batch_size
        self.normalize = normalize

    def process(self, data):
        if self.normalize:
            return data / 255.0
        return data


@track
class DataProcessorV2:
    def __init__(self, batch_size: int, normalize: bool = True, dtype: str = "float32"):
        self.batch_size = batch_size
        self.normalize = normalize
        self.dtype = dtype

    def process(self, data):
        if self.normalize:
            data = (data - 128.0) / 128.0
        return data.astype(self.dtype)


@track
class Model:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


@dataclass
class DataConfig:
    path: str
    processor: Optional[object] = None
    shuffle: bool = True
    cache_enabled: bool = False


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    model: Model
    data: DataConfig
    optimizer: OptimizerConfig
    epochs: int


def create_baseline_config():
    """A baseline config for comparison to updated config"""
    processor = DataProcessorV1(batch_size=32, normalize=True)

    data_config = DataConfig(
        path="/data/train",
        processor=processor,
        shuffle=True,
        cache_enabled=False,
    )

    # Add a field that will be removed in the updated config
    data_config.max_samples = 10000

    model = lazy(
        Model,
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
    )

    optimizer = OptimizerConfig(
        name="adam",
        learning_rate=0.001,
        weight_decay=0.0,
    )

    config = TrainingConfig(
        model=model,
        data=data_config,
        optimizer=optimizer,
        epochs=50,
    )

    # Add a field that will be removed in the updated config
    config.early_stopping = 5

    return config


def create_updated_config():
    """Create updated configuration version of the baseline config"""

    # Dynamically modify the Model class to change its hash
    def new_forward(self, x):
        return x * self.hidden_dim

    Model.forward = new_forward
    Model._version = "v2"

    processor = DataProcessorV2(batch_size=32, normalize=True, dtype="float16")

    data_config = DataConfig(
        path="/data/train",
        processor=processor,
        shuffle=True,
        cache_enabled=True,
    )

    # Add a field that didn't exist in the baseline config
    data_config.prefetch_factor = 2

    model = lazy(
        Model,
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
    )

    optimizer = OptimizerConfig(
        name="adam",
        learning_rate=0.0005,
        weight_decay=0.0,
    )

    config = TrainingConfig(
        model=model,
        data=data_config,
        optimizer=optimizer,
        epochs=50,
    )

    # Add a field that didn't exist in the baseline config
    config.gradient_accumulation = 4

    return config


# Example classes for a more complex config
@track
class DataProcessor:
    def __init__(self, batch_size: int, normalize: bool = True):
        self.batch_size = batch_size
        self.normalize = normalize

    def process(self, data):
        return data


@track
class Augmentation:
    def __init__(self, rotation: float, flip: bool, scale: float = 1.0):
        self.rotation = rotation
        self.flip = flip
        self.scale = scale


@track
class ComplexModel:
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout


@dataclass
class ComplexDataConfig:
    path: str
    batch_size: int
    shuffle: bool = True
    processor: Optional[DataProcessor] = None
    augmentation: Optional[Augmentation] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ComplexOptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class ComplexTrainingConfig:
    model: ComplexModel
    data: ComplexDataConfig
    optimizer: ComplexOptimizerConfig
    epochs: int
    checkpoint_every: int = 10
    early_stopping_patience: int = 5
    mixed_precision: bool = False


def create_complex_config():
    """Create a complex nested configuration from visualize_interactive.py."""
    processor = DataProcessor(batch_size=32, normalize=True)
    augmentation = Augmentation(rotation=15.0, flip=True, scale=1.2)

    data_config = ComplexDataConfig(
        path="/data/train",
        batch_size=32,
        shuffle=True,
        processor=processor,
        augmentation=augmentation,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    model = lazy(
        ComplexModel, input_dim=784, hidden_dim=256, output_dim=10, dropout=0.1
    )

    optimizer = ComplexOptimizerConfig(
        name="adamw", learning_rate=0.001, weight_decay=0.01, betas=(0.9, 0.98)
    )

    config = ComplexTrainingConfig(
        model=model,
        data=data_config,
        optimizer=optimizer,
        epochs=100,
        checkpoint_every=20,
        early_stopping_patience=10,
        mixed_precision=True,
    )

    return config


def get_default_configs():
    """Get all default configurations as serialized configs."""
    configs = {}

    # Individual configs
    configs["Baseline Training Config"] = serialize_fingy(create_baseline_config())
    configs["Updated Training Config"] = serialize_fingy(create_updated_config())
    configs["Complex Training Config"] = serialize_fingy(create_complex_config())

    return configs


def get_default_comparisons():
    """Get default comparison pairs."""
    comparisons = {}

    # Baseline vs Updated comparison
    comparisons["Baseline vs Updated"] = {
        "config1": serialize_fingy(create_baseline_config()),
        "config2": serialize_fingy(create_updated_config()),
        "title": "Simple Configuration Comparison",
        "description": "Shows differences between baseline and updated training configurations",
    }

    return comparisons
