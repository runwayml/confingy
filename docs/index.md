# Introduction

`confingy` is an _implicit configuration system_ for Python. It was built as an attempt to break out of configuration hell by bringing your configuration and code together, at last.

`confingy` is primarily geared towards iterative or experimental works, such as training machine learning (ML) / AI models, via 3 core features:

1. Track the constructor arguments for arbitrary Python classes. No need to define configuration objects or YAML files.
2. Lazily-instantiate any tracked class. Wait until your model is on the cluster before allocating all that memory.
3. Serialize tracked classes to JSON and deserialize back into Python. Reproducibility and lineage are byproduct of tracking.

The above features also apply to standard python objects like dictionaries and dataclasses, as well as dependency-injected tracked classes, so you can package up an entire deep learning job into a single class rather than in a mess of YAML files that tell your job runner what to do.

## Installation

Install from PyPI with either `pip` or `uv`
```bash
uv add confingy
# OR
pip install confingy
```

If you're using mypy, then add the mypy plugin [confingy.mypy_plugin][confingy.mypy_plugin] to whichever type of config file you're using:

pyproject.toml

```toml
[tool.mypy]
plugins = ["confingy.mypy_plugin"]
```

mypy.ini

```ini
[mypy]
plugins = confingy.mypy_plugin
```

## Quick Start

### Tracking

The arguments to any class' constructor may be tracked with [confingy.track][confingy.track], as long as the arguments themselves are trackable (tracked classes and most stdlib python objects are supported). The arguments will be stored in a private `_tracked_info` attribute.

```python
import random

from confingy import track


@track
class MyDataset:
    def __init__(self, size: int):
        self.size = size
        self.data = [random.random() for _ in range(size)]

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

size_10 = MyDataset(10)
print(size_10._tracked_info)
# {'class': 'MyDataset', 'module': '__main__', 'init_args': {'size': 10}, 'class_hash': 'f8fa2463f8c366f0292f538903a24ed57968a05c9e36bfccbf7147daa77a65ae'}
size_20 = MyDataset(20)
print(size_20._tracked_info)
# {'class': 'MyDataset', 'module': '__main__', 'init_args': {'size': 20}, 'class_hash': 'f8fa2463f8c366f0292f538903a24ed57968a05c9e36bfccbf7147daa77a65ae'}
```

Any object instantiated from a tracked class can be serialized to a standard confingy JSON "fingy":

```python
from confingy import serialize_fingy

print(serialize_fingy(size_10))
# {'_confingy_class': 'MyDataset', '_confingy_module': '__main__', '_confingy_init': {'size': 10}, '_confingy_class_hash': 'f8fa2463f8c366f0292f538903a24ed57968a05c9e36bfccbf7147daa77a65ae'}
```

You can then deserialize your fingy back to python

```python
from confingy import deserialize_fingy

print(deserialize_fingy(serialize_fingy(size_10)))
# <__main__.MyDataset object at 0x7f2830a5b8b0>
```

To save and load fingys directly to/from JSON files, use [save_fingy][confingy.save_fingy] and [load_fingy][confingy.load_fingy]:

```python
from confingy import save_fingy, load_fingy

save_fingy(size_10, "my_dataset.json")
loaded = load_fingy("my_dataset.json")
```

### Composability

You can chain, nest, and dependency inject tracked classes and python types in order to gain lineage and reproducibility.

By packaging up your entire job into a dataclass that consists of classes that take other classes as arguments (a la dependency injection), you end up with a graph of sorts that can be fully serialized and deserialized.

```python
import json
from dataclasses import dataclass

from confingy import track, serialize_fingy


@track
class DataFetcher:
    def __init__(self, start: str, end: str):
        pass 

@track
class DataLoader:
    def __init__(self, fetcher: DataFetcher, batch_size: int):
        pass 

@track
class Model:
    def __init__(self, hyperparameter: float):
        pass

@track
class Ensemble:
    def __init__(self, models: list[Model]):
        pass

@dataclass
class Job:
    dataloader: DataLoader
    model: Ensemble | Model

job = Job(
    dataloader=DataLoader(DataFetcher("2026-01-01", "2026-01-31"), 32), 
    model=Ensemble([Model(1.0), Model(2.0)])
)

serialized = serialize_fingy(job)
print(json.dumps(serialized, indent=2))
```

??? example "Serialized JSON output"

    ```json
    {
      "_confingy_class": "Job",
      "_confingy_module": "__main__",
      "_confingy_dataclass": true,
      "_confingy_fields": {
        "dataloader": {
          "_confingy_class": "DataLoader",
          "_confingy_module": "__main__",
          "_confingy_init": {
            "fetcher": {
              "_confingy_class": "DataFetcher",
              "_confingy_module": "__main__",
              "_confingy_init": {
                "start": "2026-01-01",
                "end": "2026-01-31"
              },
              "_confingy_class_hash": "cab..."
            },
            "batch_size": 32
          },
          "_confingy_class_hash": "3b3..."
        },
        "model": {
          "_confingy_class": "Ensemble",
          "_confingy_module": "__main__",
          "_confingy_init": {
            "models": [
              {
                "_confingy_class": "Model",
                "_confingy_module": "__main__",
                "_confingy_init": { "hyperparameter": 1.0 },
                "_confingy_class_hash": "24a..."
              },
              {
                "_confingy_class": "Model",
                "_confingy_module": "__main__",
                "_confingy_init": { "hyperparameter": 2.0 },
                "_confingy_class_hash": "24a..."
              }
            ]
          },
          "_confingy_class_hash": "7e8..."
        }
      }
    }
    ```

### Lazy Instantiation

Sometimes you may have large classes that you want to use, but you don't want to instantiate them when the class constructor is called. For example, maybe you want to create a config for you disributed training job, but you don't want to instantiate the model until you're on the distributed node. For these classes, you decorate them with `@track` and then call a `.lazy()` classmethod to construct a lazy object of type [Lazy[T]][confingy.tracking.Lazy] for class `T`.

```python
from confingy import Lazy

@track
class MyModel:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.expensive_initialization()

    def expensive_initialization(self):
        print("Creating a bunch of tensors...")


lazy_model = MyModel.lazy(1_000)
print(lazy_model)
# Lazy<MyModel>(lazy, config={'num_layers': 1000})

# Alternatively, you can do the following, even if the class is not wrapped with @track
# lazy_model = confingy.lazy(MyModel)(1_000)

# Inspect the configuration without instantiating
print(lazy_model.get_config())
# {'num_layers': 1000}


@dataclass
class LazyTrainingConfig:
    dataset: MyDataset
    model: Lazy[MyModel]


lazy_training_config = LazyTrainingConfig(
    dataset=MyDataset(100),
    model=lazy_model
)

# Instantiate the model
model = lazy_training_config.model.instantiate()
# Creating a bunch of tensors...

print(model)
# <__main__.MyModel object at 0x105c59b10>
```

### Validation

A benefit of using [@track][confingy.tracking.track] is that you get [Pydantic](https://docs.pydantic.dev/latest/) validation for free. This means that you can catch runtime type errors like

```python
from confingy import track


@track
class Foo:
    def __init__(self, a_string: str):
        self.a_string = a_string

# This raises an error
Foo(1.0)
# ValidationError: Validation failed for Foo:
#   • Field 'a_string': Input should be a valid string (got 1.0)
```

This also works when using [confingy.lazy()][confingy.tracking.lazy]

```python
from confingy import lazy

# This raises an error
lazy(Foo)(1.0)
```

For validating configuration dataclasses, you can use [Pydantic dataclasses](https://docs.pydantic.dev/latest/concepts/dataclasses/) in order to get validation. Note the usage of `arbitrary_types_allowed=True` to support custom classes in the dataclass.

```python
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from confingy import track


@track
class Foo:
    def __init__(self, a_string: str):
        self.a_string = a_string


@track
class Bar:
    def __init__(self, an_int: int):
        self.an_int = an_int


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MyConfig:
    foo: Foo
    string_list: list[str]


# This works
MyConfig(Foo("a string"), ["a", "b"])
# This raises a pydantic validation error
MyConfig(Foo("a string"), [1.0, 2.0])
# We can also raise errors based on custom classes
MyConfig(Bar(1), ["a", "b"])

```

## Why?

We always end up building giant configuration objects/files for machine learning projects. We then have to create interfaces for converting the configuration into python. 

For example, you have probably seen this pattern before:

```python
from dataclasses import dataclass 
# Function to dynamically load a class based on its string path.
from my_lib import load_class

class Foo:
    def __init__(self, bar: int, baz: str):
        self.bar = bar
        self.baz = baz


@dataclass
class MyConfig:
    config_class: str 
    config_kwargs: dict


# No type-hint validation.
config = MyConfig(config_class="Foo", config_kwargs={"bar": 1, "baz": "hello"})

# Both you and the IDE have no idea what type of object `foo` is.
foo = load_class(config.config_class)(**config.config_kwargs)

```

This pattern is painful since we invariably end up having to create interfaces that take in a config object and then figure out how to dynamically instantiate python classes from that object. Additionally, this pattern often encourages inheritance over composition and pushes us away from dependency injection.

Why do we implement this pattern? Because this pattern allows us to use custom-classes from our library, track everything in a reproducible config object, and lazy-instantiate classes that may be too costly to instantiate when we're defining our `config`.

Ideally, we could just use python, avoid interfaces, and keep our IDE happy:

```python

@dataclass
class MyConfig:
    my_obj: Foo 


config = MyConfig(my_obj=Foo(1, "baz"))
```

`confingy` aims to do just this, without losing any of the benefits of the prior approach.

## API Quick Reference

| Function / Class | Description |
|------------------|-------------|
| [@track][confingy.track] | Decorator to track constructor arguments |
| [lazy()][confingy.lazy] | Create a lazy instance of a class |
| [Lazy\[T\]][confingy.Lazy] | Type hint for lazy objects |
| [lens()][confingy.lens] | Convert tracked/lazy objects for deep modifications |
| [serialize_fingy()][confingy.serialize_fingy] | Serialize a fingy to a dict |
| [deserialize_fingy()][confingy.deserialize_fingy] | Deserialize a dict back to Python |
| [save_fingy()][confingy.save_fingy] | Save a fingy to a JSON file |
| [load_fingy()][confingy.load_fingy] | Load a fingy from a JSON file |
| [transpile_fingy()][confingy.transpile_fingy] | Convert serialized fingy to Python code |
| [prettify_serialized_fingy()][confingy.prettify_serialized_fingy] | Make serialized fingy human-readable |
| [disable_validation()][confingy.disable_validation] | Context manager to skip Pydantic validation |

**Lazy methods:** [.instantiate()][confingy.Lazy.instantiate], [.get_config()][confingy.Lazy.get_config], [.copy()][confingy.Lazy.copy], [.unlens()][confingy.Lazy.unlens]

**Exceptions:** [ValidationError][confingy.ValidationError], [SerializationError][confingy.SerializationError], [DeserializationError][confingy.DeserializationError]

**CLI:** `confingy serialize`, `confingy transpile`, `confingy viz` — see [Diving Deeper](diving_deeper.md#transpiler) for usage.

See the [API Reference](api/tracking.md) for full details.