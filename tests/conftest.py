"""
Shared test fixtures and utilities for confingy tests.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from confingy import Lazy, track


# Test model classes
@track
class MyModel:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        return [_ * self.in_features + self.out_features for _ in x]


@track
class MyDataset:
    def __init__(
        self, num_samples: int, num_features: int, processor: Callable[[float], float]
    ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.processor = processor
        self.data = [random.random() for _ in range(self.num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.processor(self.data[idx])


@track
class MyDataloader:
    def __init__(self, dataset: MyDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for batch_start in range(0, len(self.dataset), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.dataset))
            yield [self.dataset[i] for i in range(batch_start, batch_end)]


@track
class Adder:
    def __init__(self, amount: float):
        self.amount = amount

    def __call__(self, data: float) -> float:
        return self.amount + data


@track
class Multiplier:
    def __init__(self, amount: float):
        self.amount = amount

    def __call__(self, data: float) -> float:
        return self.amount * data


@track
class ProcessorPipeline:
    def __init__(self, processors: Sequence[Callable[[float], float]]):
        self.processors = processors

    def __call__(self, data: float) -> float:
        for processor in self.processors:
            data = processor(data)
        return data


# Configuration classes
@dataclass
class TrainingFingy:
    model: Lazy[MyModel]
    dataloader: Lazy[MyDataloader]


@dataclass
class CollectionFingy:
    numbers_list: list[int]
    numbers_tuple: tuple[int, ...]
    mapping: dict[str, float]
    nested: list[dict[str, list[int]]]


@dataclass
class RequiredFields:
    required: int
    optional: int = 0


@dataclass
class NestedFingy:
    name: str
    values: list[int]


@dataclass
class ComplexFingy:
    simple_field: int
    optional_field: Optional[str]
    nested: NestedFingy
    multiple_nested: list[NestedFingy]


@dataclass
class WithNones:
    maybe_value: Optional[int]
    maybe_list: Optional[list[int]]
    list_with_nones: list[Optional[int]]


class WithMethods:
    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    def method(self, x: int) -> int:
        return x * self.multiplier


# Test utility functions
def standalone_function(x: int) -> int:
    return x * 2


# Lazy instance test classes for nested testing
@track
class Inner:
    def __init__(self, value: int):
        self.value = value


@track
class Middle:
    def __init__(self, inner: Lazy[Inner]):
        self.inner = inner


@track
class Outer:
    def __init__(self, middle: Lazy[Middle]):
        self.middle = middle


class Untracked:
    """A plain (untracked) class for testing inline track() pickling."""

    def __init__(self, value: int):
        self.value = value


class UntrackedWithReduce:
    """A class with a custom __reduce__ for testing tracked pickle doesn't clobber it."""

    def __init__(self, value: int):
        self.value = value
        self.extra = "set_by_init"

    def __reduce__(self):
        return (_rebuild_untracked_with_reduce, (self.value, self.extra))


def _rebuild_untracked_with_reduce(value: int, extra: str) -> "UntrackedWithReduce":
    obj = UntrackedWithReduce.__new__(UntrackedWithReduce)
    obj.value = value
    obj.extra = extra
    return obj
