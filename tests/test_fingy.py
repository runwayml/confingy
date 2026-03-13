"""
Tests for confingy.api module - main serialization/deserialization functions.
"""

import json
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from confingy import (
    deserialize_fingy,
    lazy,
    load_fingy,
    save_fingy,
    serialize_fingy,
)
from confingy.fingy import prettify_serialized_fingy, transpile_fingy
from tests.conftest import (
    Adder,
    CollectionFingy,
    ComplexFingy,
    Multiplier,
    MyDataloader,
    MyDataset,
    MyModel,
    NestedFingy,
    ProcessorPipeline,
    TrainingFingy,
    standalone_function,
)


def test_with_fingy(tmp_path):
    """Test saving and loading fingys with nested confingy objects."""
    processor = ProcessorPipeline([Adder(1.0), Multiplier(2.0)])
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=8, out_features=16),
        # We can pass nested confingy'd objects as well.
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(num_samples=100, num_features=8, processor=processor),
            batch_size=32,
        ),
    )

    save_fingy(fingy, (tmp_path / "fingy.json").as_posix())
    # If the test fails, the below line will print the contents of the fingy file
    # to help debug the issue.
    with (tmp_path / "fingy.json").open("r") as f:
        print(f.read())
    loaded_fingy = load_fingy((tmp_path / "fingy.json").as_posix())

    # Check model config without instantiation
    model_config = loaded_fingy.model.get_config()
    assert model_config["in_features"] == 8
    assert model_config["out_features"] == 16
    # Check dataloader config without instantiation
    dataloader_config = loaded_fingy.dataloader.get_config()
    assert dataloader_config["batch_size"] == 32

    # Dataset is tracked (not lazy), so we need to instantiate the dataloader to access it
    loaded_fingy.dataloader.instantiate()
    assert loaded_fingy.dataloader.dataset.num_samples == 100
    assert loaded_fingy.dataloader.dataset.num_features == 8
    assert loaded_fingy.dataloader.dataset.processor.processors[0].amount == 1.0
    assert loaded_fingy.dataloader.dataset.processor.processors[1].amount == 2.0

    model = loaded_fingy.model.instantiate()
    dataloader = loaded_fingy.dataloader.instantiate()

    # Iterate through the dataloader and print the processed data
    for batch in dataloader:
        model(batch)


def test_basic_serialize_deserialize():
    """Test basic serialization and deserialization without file I/O."""
    processor = ProcessorPipeline([Adder(5.0), Multiplier(3.0)])
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=4, out_features=8),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(num_samples=50, num_features=4, processor=processor),
            batch_size=16,
        ),
    )

    # Serialize to dictionary
    serialized = serialize_fingy(fingy)
    assert isinstance(serialized, dict)
    assert "_confingy_class" in serialized
    assert "_confingy_dataclass" in serialized

    # Deserialize from dictionary
    deserialized = deserialize_fingy(serialized)

    # Verify structure using config access
    model_config = deserialized.model.get_config()
    assert model_config["in_features"] == 4
    assert model_config["out_features"] == 8
    # Check dataloader config without instantiation
    dataloader_config = deserialized.dataloader.get_config()
    assert dataloader_config["batch_size"] == 16

    # Need to instantiate dataloader to access dataset
    deserialized.dataloader.instantiate()
    assert deserialized.dataloader.dataset.num_samples == 50
    assert deserialized.dataloader.dataset.num_features == 4

    # Test that lazy objects are still lazy
    assert hasattr(deserialized.model, "_confingy_lazy_info")
    assert hasattr(deserialized.dataloader, "_confingy_lazy_info")


def test_prettify_fingy():
    """Test prettify_fingy function with a basic fingy."""
    from confingy import prettify_fingy

    # Create a fingy
    processor = ProcessorPipeline([Adder(1.0), Multiplier(2.0)])
    dataset = MyDataset(num_samples=100, num_features=8, processor=processor)
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=8, out_features=16),
        dataloader=lazy(MyDataloader, dataset=dataset, batch_size=32),
    )

    # Prettify the fingy
    pretty = prettify_fingy(fingy)

    # Define expected structure
    expected = {
        "tests.conftest.TrainingFingy": {
            "model": {"tests.conftest.MyModel": {"in_features": 8, "out_features": 16}},
            "dataloader": {
                "tests.conftest.MyDataloader": {
                    "dataset": {
                        "tests.conftest.MyDataset": {
                            "num_samples": 100,
                            "num_features": 8,
                            "processor": {
                                "tests.conftest.ProcessorPipeline": {
                                    "processors": [
                                        {"tests.conftest.Adder": {"amount": 1.0}},
                                        {"tests.conftest.Multiplier": {"amount": 2.0}},
                                    ]
                                }
                            },
                        }
                    },
                    "batch_size": 32,
                }
            },
        }
    }

    assert pretty == expected


def test_prettify_serialized_fingy():
    """Test prettify_serialized_fingy with pre-serialized data."""
    from confingy import prettify_serialized_fingy, serialize_fingy

    # Create and serialize a fingy
    processor = ProcessorPipeline([Adder(5.0)])
    dataset = MyDataset(num_samples=50, num_features=4, processor=processor)
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=4, out_features=8),
        dataloader=lazy(MyDataloader, dataset=dataset, batch_size=16),
    )

    serialized = serialize_fingy(fingy)

    # Prettify the serialized fingy
    pretty = prettify_serialized_fingy(serialized)

    # Define expected structure
    expected = {
        "tests.conftest.TrainingFingy": {
            "model": {"tests.conftest.MyModel": {"in_features": 4, "out_features": 8}},
            "dataloader": {
                "tests.conftest.MyDataloader": {
                    "dataset": {
                        "tests.conftest.MyDataset": {
                            "num_samples": 50,
                            "num_features": 4,
                            "processor": {
                                "tests.conftest.ProcessorPipeline": {
                                    "processors": [
                                        {"tests.conftest.Adder": {"amount": 5.0}}
                                    ]
                                }
                            },
                        }
                    },
                    "batch_size": 16,
                }
            },
        }
    }

    assert pretty == expected


def test_prettify_fingy_with_nested_structures():
    """Test prettification with nested lists and dictionaries."""
    from dataclasses import dataclass
    from typing import Any

    from confingy import prettify_fingy

    @dataclass
    class NestedFingy:
        models: list[Any]
        metadata: dict[str, Any]

    # Create nested structure
    fingy = NestedFingy(
        models=[
            lazy(MyModel, in_features=2, out_features=4),
            lazy(MyModel, in_features=4, out_features=8),
        ],
        metadata={
            "version": "1.0",
            "dataset": MyDataset(
                num_samples=10, num_features=2, processor=ProcessorPipeline([])
            ),
        },
    )

    pretty = prettify_fingy(fingy)

    # Define expected structure - note that the module path will be based on where NestedFingy is defined
    expected = {
        "tests.test_fingy.NestedFingy": {
            "models": [
                {"tests.conftest.MyModel": {"in_features": 2, "out_features": 4}},
                {"tests.conftest.MyModel": {"in_features": 4, "out_features": 8}},
            ],
            "metadata": {
                "version": "1.0",
                "dataset": {
                    "tests.conftest.MyDataset": {
                        "num_samples": 10,
                        "num_features": 2,
                        "processor": {
                            "tests.conftest.ProcessorPipeline": {"processors": []}
                        },
                    }
                },
            },
        }
    }

    assert pretty == expected


def test_prettify_fingy_example_from_docstring():
    """Test the exact example from the user's request."""
    from confingy import prettify_serialized_fingy

    # Create the example dict from the docstring
    example_dict = {
        "_confingy_class": "TrainingFingy",
        "_confingy_module": "__main__",
        "_confingy_dataclass": True,
        "_confingy_fields": {
            "dataset": {
                "_confingy_class": "MyDataset",
                "_confingy_module": "__main__",
                "_confingy_init": {"size": 100},
            }
        },
    }

    pretty = prettify_serialized_fingy(example_dict)

    # Check it matches expected structure
    expected = {
        "__main__.TrainingFingy": {"dataset": {"__main__.MyDataset": {"size": 100}}}
    }

    assert pretty == expected


def test_prettify_fingy_with_lazy_objects():
    """Test prettification specifically with lazy objects."""
    from confingy import prettify_fingy

    # Create a fingy with lazy objects
    lazy_model = lazy(MyModel, in_features=10, out_features=20)
    dataset = MyDataset(
        num_samples=200, num_features=10, processor=ProcessorPipeline([])
    )

    config = TrainingFingy(
        model=lazy_model, dataloader=lazy(MyDataloader, dataset=dataset, batch_size=64)
    )

    # Prettify and check the structure
    pretty = prettify_fingy(config)

    # Define expected structure
    expected = {
        "tests.conftest.TrainingFingy": {
            "model": {
                "tests.conftest.MyModel": {"in_features": 10, "out_features": 20}
            },
            "dataloader": {
                "tests.conftest.MyDataloader": {
                    "dataset": {
                        "tests.conftest.MyDataset": {
                            "num_samples": 200,
                            "num_features": 10,
                            "processor": {
                                "tests.conftest.ProcessorPipeline": {"processors": []}
                            },
                        }
                    },
                    "batch_size": 64,
                }
            },
        }
    }

    assert pretty == expected


def test_basic_transpilation():
    """Test basic transpilation of a simple fingy."""
    # Create a simple fingy
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=8, out_features=16),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=100, num_features=8, processor=ProcessorPipeline([])
            ),
            batch_size=32,
        ),
    )

    # Serialize it
    serialized = serialize_fingy(fingy)

    # Transpile it
    python_code = transpile_fingy(serialized)

    # Check that the code contains expected elements
    assert "from confingy import lazy" in python_code
    assert "from tests.conftest import" in python_code
    assert "TrainingFingy" in python_code
    assert "MyModel" in python_code
    assert "MyDataloader" in python_code
    assert "MyDataset" in python_code
    assert "in_features=8" in python_code
    assert "out_features=16" in python_code
    assert "batch_size=32" in python_code
    assert "num_samples=100" in python_code


def test_transpilation_with_processors():
    """Test transpilation with nested processor pipeline."""
    processor = ProcessorPipeline([Adder(1.0), Multiplier(2.0)])
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=4, out_features=8),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(num_samples=50, num_features=4, processor=processor),
            batch_size=16,
        ),
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Check for processor-related code
    assert "ProcessorPipeline" in python_code
    assert "Adder" in python_code
    assert "Multiplier" in python_code
    assert "amount=1.0" in python_code
    assert "amount=2.0" in python_code


def test_transpilation_with_collections():
    """Test transpilation of fingys with collections."""
    fingy = CollectionFingy(
        numbers_list=[1, 2, 3, 4, 5],
        numbers_tuple=(10, 20, 30),
        mapping={"pi": 3.14, "e": 2.718},
        nested=[{"odds": [1, 3, 5]}, {"evens": [2, 4, 6]}],
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Check for collection representations
    assert "CollectionFingy" in python_code
    assert "[1, 2, 3, 4, 5]" in python_code
    assert '"pi": 3.14' in python_code or "'pi': 3.14" in python_code
    assert '"e": 2.718' in python_code or "'e': 2.718" in python_code

    # Check nested structures are preserved
    assert "odds" in python_code
    assert "evens" in python_code


def test_transpilation_with_none_values():
    """Test transpilation handles None values correctly."""
    fingy = ComplexFingy(
        simple_field=42,
        optional_field=None,
        nested=NestedFingy(name="test", values=[1, 2, 3]),
        multiple_nested=[],
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Check None handling
    assert "optional_field=None" in python_code
    assert "simple_field=42" in python_code
    assert 'name="test"' in python_code or "name='test'" in python_code
    assert "multiple_nested=[]" in python_code


def test_transpilation_from_json_file():
    """Test transpilation from a JSON file."""
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=2, out_features=4),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=10, num_features=2, processor=ProcessorPipeline([])
            ),
            batch_size=5,
        ),
    )

    serialized = serialize_fingy(fingy)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(serialized, f)
        temp_path = f.name

    try:
        # Transpile from file path
        python_code = transpile_fingy(temp_path)

        # Check it worked
        assert "TrainingFingy" in python_code
        assert "in_features=2" in python_code
        assert "out_features=4" in python_code
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_transpilation_from_json_string():
    """Test transpilation from a JSON string."""
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=3, out_features=6),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=20, num_features=3, processor=ProcessorPipeline([])
            ),
            batch_size=10,
        ),
    )

    serialized = serialize_fingy(fingy)
    json_string = json.dumps(serialized)

    # Transpile from JSON string
    python_code = transpile_fingy(json_string)

    # Check it worked
    assert "TrainingFingy" in python_code
    assert "in_features=3" in python_code
    assert "out_features=6" in python_code


def test_transpilation_multiline_formatting():
    """Test that complex fingys get properly formatted with indentation."""
    # Create a complex nested fingy
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=100, out_features=200),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=10000,
                num_features=100,
                processor=ProcessorPipeline(
                    [
                        Adder(1.0),
                        Multiplier(2.0),
                        Adder(3.0),
                        Multiplier(4.0),
                    ]
                ),
            ),
            batch_size=128,
        ),
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Check for proper formatting (should have multiple lines due to complexity)
    lines = python_code.split("\n")
    assert len(lines) > 10  # Should be formatted across multiple lines

    # Check indentation is present
    assert any("    " in line for line in lines)  # Should have indented lines


def test_transpilation_preserves_types():
    """Test that transpilation preserves different Python types correctly."""
    fingy = {
        "int": 42,
        "float": 3.14159,
        "str": "hello world",
        "bool_true": True,
        "bool_false": False,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
    }

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Check type representations
    assert "42" in python_code
    assert "3.14159" in python_code
    assert '"hello world"' in python_code or "'hello world'" in python_code
    assert "True" in python_code
    assert "False" in python_code
    assert "None" in python_code
    assert "[1, 2, 3]" in python_code
    assert '"nested"' in python_code or "'nested'" in python_code


def test_transpilation_executable():
    """Test that transpiled code can be executed."""
    # Create a simple fingy
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=5, out_features=10),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=25,
                num_features=5,
                processor=ProcessorPipeline([Adder(1.0)]),
            ),
            batch_size=8,
        ),
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # The transpiled code should be valid Python
    # We can't easily execute it here due to import dependencies,
    # but we can at least compile it
    try:
        compile(python_code, "<string>", "exec")
    except SyntaxError:
        pytest.fail(f"Transpiled code has syntax errors:\n{python_code}")


def test_transpilation_handles_empty_fingy():
    """Test transpilation of empty or minimal fingys."""
    # Empty dict
    empty_dict = {}
    serialized = serialize_fingy(empty_dict)
    python_code = transpile_fingy(serialized)
    assert "config = {}" in python_code

    # Empty list
    empty_list = []
    serialized = serialize_fingy(empty_list)
    python_code = transpile_fingy(serialized)
    assert "config = []" in python_code


def test_lazy_transpilation_with_no_args():
    """Test transpilation of lazy objects with no args."""

    class Foo:
        def __init__(self):
            pass

    fingy = {"foo": lazy(Foo)()}

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    expected = dedent("""
    from confingy import lazy
    from tests.test_fingy import Foo

    config = {"foo": lazy(Foo)()}
    """).strip("\n")

    assert python_code == expected, f"Unexpected transpiled code:\n{python_code}"


def test_transpilation_tuples_and_sets():
    """Test that tuples and sets are correctly transpiled."""
    fingy = CollectionFingy(
        numbers_list=[1, 2, 3],
        numbers_tuple=(10, 20, 30),
        mapping={"a": 1.0},
        nested=[],
    )

    serialized = serialize_fingy(fingy)
    python_code = transpile_fingy(serialized)

    # Tuples should be transpiled as tuple syntax, not as dict with _confingy_tuple
    assert "_confingy_tuple" not in python_code, (
        f"Tuple was not transpiled correctly:\n{python_code}"
    )
    assert "(10, 20, 30)" in python_code, (
        f"Expected tuple syntax (10, 20, 30) not found:\n{python_code}"
    )

    # Test sets as well
    fingy_with_set = {"my_set": {1, 2, 3}}
    serialized = serialize_fingy(fingy_with_set)
    python_code = transpile_fingy(serialized)

    assert "_confingy_set" not in python_code, (
        f"Set was not transpiled correctly:\n{python_code}"
    )
    # Sets may be in any order, so just check set syntax exists
    assert (
        "{1, 2, 3}" in python_code
        or "{1, 3, 2}" in python_code
        or "{2, 1, 3}" in python_code
        or "{2, 3, 1}" in python_code
        or "{3, 1, 2}" in python_code
        or "{3, 2, 1}" in python_code
    ), f"Expected set syntax not found:\n{python_code}"

    # Test single-element tuple (needs trailing comma)
    fingy_single = {"single": (42,)}
    serialized = serialize_fingy(fingy_single)
    python_code = transpile_fingy(serialized)
    assert "(42,)" in python_code, (
        f"Single-element tuple missing trailing comma:\n{python_code}"
    )

    # Test single-element tuple with complex content (multi-line format)
    # This triggers multi-line formatting due to long nested content
    fingy_complex_single = {
        "complex_single": (
            {"a_very_long_key_name": "a_very_long_value_that_exceeds_the_line_limit"},
        )
    }
    serialized = serialize_fingy(fingy_complex_single)
    python_code = transpile_fingy(serialized)
    # Verify the transpiled code compiles and the tuple has trailing comma
    assert "_confingy_tuple" not in python_code
    try:
        compile(python_code, "<string>", "exec")
    except SyntaxError:
        pytest.fail(f"Multi-line single-element tuple has syntax error:\n{python_code}")


def test_prettify_collapses_tuples():
    """Test that prettify collapses tuple metadata into a plain list."""
    from confingy import prettify_fingy

    fingy = CollectionFingy(
        numbers_list=[1, 2],
        numbers_tuple=(10, 20, 30),
        mapping={"a": 1.0},
        nested=[],
    )
    pretty = prettify_fingy(fingy)

    # Navigate into the prettified structure
    inner = next(iter(pretty.values()))
    # Tuple should become a plain list, not a dict with _confingy_tuple
    assert inner["numbers_tuple"] == [10, 20, 30]
    assert not isinstance(inner["numbers_tuple"], dict)


def test_prettify_collapses_callables():
    """Test that prettify collapses callable metadata into a dotted string."""
    serialized = serialize_fingy({"fn": standalone_function})
    pretty = prettify_serialized_fingy(serialized)

    # Should be "module.func_name", not a raw dict with _confingy_callable
    assert isinstance(pretty["fn"], str)
    assert "standalone_function" in pretty["fn"]
    assert "_confingy_callable" not in str(pretty)


def test_transpile_callable_fields():
    """Test that transpile emits function references for callable fields."""
    serialized = serialize_fingy({"fn": standalone_function})
    code = transpile_fingy(serialized)

    # Should reference the function name, not emit a dict literal
    assert "standalone_function" in code
    assert "_confingy_callable" not in code


def test_transpile_nested_indentation():
    """Test that deeply nested lazy/tracked configs have correct indentation."""
    # Use a structure complex enough to force multi-line at multiple depths
    fingy = TrainingFingy(
        model=lazy(MyModel, in_features=8, out_features=16),
        dataloader=lazy(
            MyDataloader,
            dataset=MyDataset(
                num_samples=100,
                num_features=8,
                processor=ProcessorPipeline([Adder(1.0), Multiplier(2.0)]),
            ),
            batch_size=32,
        ),
    )

    serialized = serialize_fingy(fingy)
    code = transpile_fingy(serialized)
    lines = code.split("\n")

    # Verify increasing indentation for nested constructs
    indented_lines = [line for line in lines if line.startswith("    ")]
    double_indented = [line for line in lines if line.startswith("        ")]

    assert len(indented_lines) > 0, f"Expected indented lines:\n{code}"
    assert len(double_indented) > 0, f"Expected double-indented lines:\n{code}"

    # Closing parens should align with their opening construct, not be over-indented
    # Find lines that are just a closing paren (with optional whitespace)
    close_lines = [(i, line) for i, line in enumerate(lines) if line.strip() == ")"]
    for i, close_line in close_lines:
        close_indent = len(close_line) - len(close_line.lstrip())
        # The line before the closing paren should be more indented
        prev_nonblank = lines[i - 1]
        prev_indent = len(prev_nonblank) - len(prev_nonblank.lstrip())
        assert prev_indent > close_indent, (
            f"Line {i}: closing paren indent ({close_indent}) should be less "
            f"than previous line indent ({prev_indent}):\n{code}"
        )

    # The code should compile
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        pytest.fail(f"Transpiled code has syntax errors:\n{code}")
