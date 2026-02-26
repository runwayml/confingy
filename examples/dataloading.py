import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from confingy import deserialize_fingy, serialize_fingy, track


@track
class Pipeline:
    def __init__(self, processors: list[Callable]):
        self.processors = processors

    def __call__(self, x):
        for processor in self.processors:
            x = processor(x)
        return x


@track
class MeanScaler:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


@track
class PowerScaler:
    def __init__(self, power: float):
        self.power = power

    def __call__(self, x):
        return x**self.power


@track
class MyDataset(Dataset):
    def __init__(self, data: list, processor: Callable):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.processor(self.data[idx])


@dataclass
class Config:
    dataset: Dataset
    batch_size: int


config = Config(
    dataset=MyDataset(
        data=list(range(32)),
        processor=Pipeline([MeanScaler(mean=3, std=1), PowerScaler(power=2)]),
    ),
    batch_size=8,
)


def main():
    batches = []
    dataloader = DataLoader(
        config.dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    for batch in dataloader:
        batches.append(batch)

    print(
        f"Serialized config: {json.dumps(serialize_fingy(config), indent=2, sort_keys=True)}"
    )
    deserialized_config = deserialize_fingy(serialize_fingy(config))

    dataloader = DataLoader(
        deserialized_config.dataset,
        batch_size=deserialized_config.batch_size,
        shuffle=False,
    )
    deserialized_batches = []
    for batch in dataloader:
        deserialized_batches.append(batch)

    for batch, deserialized_batch in zip(batches, deserialized_batches, strict=False):
        assert torch.allclose(batch, deserialized_batch)


if __name__ == "__main__":
    main()
