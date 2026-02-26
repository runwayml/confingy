import json
from argparse import ArgumentParser
from dataclasses import dataclass

import torch

from confingy import Lazy, deserialize_fingy, lazy, serialize_fingy, track

# Models to pick from


@track
class LinearModel(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)


@track
class NeuralNet(torch.nn.Module):
    def __init__(self, num_features: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(num_features, hidden_units))
        self.layers.append(torch.nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_units, hidden_units))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_units, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@track
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, num_features: int):
        self.x = torch.randn(num_samples, num_features)
        self.y = torch.randn(num_samples, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@dataclass
class GlobalConfig:
    num_features: int


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class Config:
    model: Lazy[torch.nn.Module]
    dataset: torch.utils.data.Dataset
    optimizer: type[torch.optim.Optimizer]
    global_config: GlobalConfig
    training_config: TrainingConfig


global_config = GlobalConfig(num_features=10)
training_config = TrainingConfig(num_epochs=10, batch_size=32, learning_rate=0.001)
linear_config = Config(
    model=lazy(LinearModel, num_features=global_config.num_features),
    dataset=ToyDataset(num_samples=1000, num_features=global_config.num_features),
    optimizer=torch.optim.Adam,
    global_config=global_config,
    training_config=training_config,
)

neural_config = Config(
    model=lazy(
        NeuralNet,
        num_features=global_config.num_features,
        hidden_units=64,
        num_layers=3,
    ),
    dataset=ToyDataset(num_samples=1000, num_features=global_config.num_features),
    optimizer=torch.optim.Adam,
    global_config=global_config,
    training_config=training_config,
)


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--model", choices=["linear", "neural"], default="linear")
    args = parser.parse_args()

    # Select config based on arguments
    config = linear_config if args.model == "linear" else neural_config
    print(f"Using model: {args.model}")
    print(
        f"Serialized config: {json.dumps(serialize_fingy(config), indent=2, sort_keys=True)}"
    )
    print(f"Deserialized config: {deserialize_fingy(serialize_fingy(config))}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        config.dataset, batch_size=config.training_config.batch_size, shuffle=True
    )

    # Create model, optimizer
    model = config.model.instantiate()
    optimizer = config.optimizer(
        model.parameters(), lr=config.training_config.learning_rate
    )

    # Training loop
    for epoch in range(config.training_config.num_epochs):
        print(f"Training epoch {epoch + 1}/{config.training_config.num_epochs}")
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
