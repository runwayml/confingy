#!/usr/bin/env python
"""
Demo script showing how to use the confingy transpiler.

This demonstrates:
1. Creating a configuration with confingy decorators
2. Serializing it to JSON
3. Transpiling it back to Python code
"""

import json
from dataclasses import dataclass

from confingy import Lazy, lazy, serialize_fingy, track
from confingy.fingy import transpile_fingy


# Example classes using confingy decorators
@track
class Model:
    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


@track
class Dataset:
    def __init__(self, path: str, batch_size: int, shuffle: bool = True):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle


@track
class Optimizer:
    def __init__(self, learning_rate: float, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


@dataclass
class TrainingConfig:
    """Main training configuration."""

    model: Lazy[Model]
    train_dataset: Dataset
    optimizer: Lazy[Optimizer]
    epochs: int
    save_path: str


def main():
    print("=== Confingy Transpiler Demo ===\n")

    # 1. Create a configuration
    print("1. Creating configuration...")
    config = TrainingConfig(
        model=lazy(Model, hidden_size=512, num_layers=12, dropout=0.1),
        train_dataset=Dataset(path="/data/train.json", batch_size=32, shuffle=True),
        optimizer=lazy(Optimizer, learning_rate=0.001, weight_decay=0.01),
        epochs=10,
        save_path="/models/checkpoint",
    )
    print("✓ Configuration created\n")

    # 2. Serialize to JSON
    print("2. Serializing to JSON...")
    serialized = serialize_fingy(config)
    json_str = json.dumps(serialized, indent=2)
    print("✓ Serialized config (JSON):")
    print("-" * 50)
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    print("-" * 50)
    print()

    # 3. Transpile to Python code
    print("3. Transpiling to Python code...")
    python_code = transpile_fingy(serialized)
    print("✓ Transpiled config (Python):")
    print("-" * 50)
    print(python_code)
    print("-" * 50)
    print()

    # Save the transpiled code to a file
    output_file = "transpiled_config.py"
    with open(output_file, "w") as f:
        f.write(python_code)
    print(f"✓ Transpiled code saved to {output_file}")

    print("\n=== Benefits of Transpilation ===")
    print("1. Type hints are preserved in the Python code")
    print("2. The code is human-readable and editable")
    print("3. It can be imported and reused in other Python modules")
    print("4. No need for JSON deserialization at runtime")
    print("5. IDE features like autocomplete work with the transpiled code")


if __name__ == "__main__":
    main()
