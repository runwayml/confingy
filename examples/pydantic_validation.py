"""
Example of using Pydantic dataclasses for validation with confingy.

Shows how Pydantic dataclasses work seamlessly with confingy's lazy loading
and serialization, with validation happening at creation time.
"""

from pydantic import Field
from pydantic.dataclasses import dataclass

from confingy import lazy, load_fingy, save_fingy, track


# Define validated configuration dataclasses
@dataclass
class TrainingConfig:
    """Training configuration with validation."""

    learning_rate: float = Field(
        gt=0, le=1.0, description="Learning rate between 0 and 1"
    )
    batch_size: int = Field(gt=0, le=1024, description="Batch size")
    num_epochs: int = Field(default=10, ge=1, le=1000)
    optimizer: str = Field(default="adam", pattern="^(adam|sgd|rmsprop)$")


@dataclass
class ModelConfig:
    """Model configuration with validation."""

    hidden_dims: list[int] = Field(min_length=1, max_length=10)
    dropout_rate: float = Field(default=0.1, ge=0, lt=1.0)
    activation: str = Field(default="relu")


# Define tracked classes that use the configs
@track
class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        # Config is always a dataclass now (after our fixes)
        for dim in self.config.hidden_dims:
            layers.append(f"Linear({dim})")
            if self.config.dropout_rate > 0:
                layers.append(f"Dropout({self.config.dropout_rate})")
        return layers


@track
class Trainer:
    def __init__(self, model: Model, config: TrainingConfig):
        self.model = model
        self.config = config

    def train(self):
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Model layers: {self.model.layers}")


def main():
    print("=" * 70)
    print("PYDANTIC VALIDATION WITH CONFINGY")
    print("=" * 70)

    # 1. Create validated configurations
    print("\n1. Creating validated configurations...")

    # This succeeds - all values are valid
    training_config = TrainingConfig(learning_rate=0.001, batch_size=32, num_epochs=100)
    print("✓ Valid training config created")

    # This would fail - validation happens immediately
    try:
        _ = TrainingConfig(
            learning_rate=1.5,  # > 1.0 - invalid!
            batch_size=32,
        )
    except Exception as e:
        print(f"✓ Invalid config rejected: {str(e)[:60]}...")

    model_config = ModelConfig(hidden_dims=[256, 128, 64], dropout_rate=0.2)
    print("✓ Valid model config created")

    # 2. Use with lazy loading
    print("\n2. Using with lazy loading...")

    # Create components with lazy instantiation
    model = Model(config=model_config)
    lazy_trainer = lazy(Trainer, model=model, config=training_config)

    print("✓ Created lazy trainer")

    # The configs remain as dataclasses (not converted to dicts!)
    trainer = lazy_trainer.instantiate()
    assert isinstance(trainer.config, TrainingConfig)
    assert isinstance(trainer.model.config, ModelConfig)
    print("✓ Configs preserved as dataclasses through lazy")

    # 3. Serialize and load
    print("\n3. Serializing and loading...")

    config_dict = {
        "training_config": training_config,
        "model_config": model_config,
        "lazy_trainer": lazy_trainer,
    }

    save_fingy(config_dict, "pydantic_example.json")
    print("✓ Saved configuration")

    loaded = load_fingy("pydantic_example.json")
    print("✓ Loaded configuration")

    # Configs are restored as dataclasses (but not Pydantic after deserialization)
    print("✓ Configs restored with values preserved:")
    print(f"  Training LR: {loaded['training_config'].learning_rate}")
    print(f"  Model dims: {loaded['model_config'].hidden_dims}")

    # Instantiate and use
    loaded_trainer = loaded["lazy_trainer"].instantiate()
    loaded_trainer.train()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key benefits of using Pydantic dataclasses with confingy:

1. **Validation at creation**: Errors caught immediately when creating configs
2. **Type preservation**: Dataclasses remain dataclasses (not converted to dicts)
3. **Seamless integration**: Works perfectly with lazy() and serialization
4. **Clear error messages**: Pydantic provides detailed validation errors
5. **Type safety**: Type hints work correctly throughout

Just use `from pydantic.dataclasses import dataclass` and add Field constraints!
""")


if __name__ == "__main__":
    main()
