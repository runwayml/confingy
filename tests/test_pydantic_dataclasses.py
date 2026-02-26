"""
Tests for specifically using Pydantic dataclasses with confingy.
"""

import json
from dataclasses import dataclass as standard_dataclass

import pytest
from pydantic import ConfigDict, Field, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from confingy import (
    Lazy,
    deserialize_fingy,
    lazy,
    load_fingy,
    save_fingy,
    serialize_fingy,
    track,
)

# Test fixtures - Pydantic dataclasses with validation


@pydantic_dataclass
class ValidatedFingy:
    """A fingy with validation rules."""

    value: int = Field(gt=0, le=100)
    name: str = Field(min_length=1, max_length=50)
    rate: float = Field(ge=0.0, le=1.0, default=0.5)


@pydantic_dataclass
class NestedValidatedFingy:
    """A fingy with nested validated fields."""

    dimensions: list[int] = Field(min_length=1, max_length=5)
    dropout: float = Field(ge=0, lt=1.0, default=0.1)
    enabled: bool = Field(default=True)


@standard_dataclass
class MixedFingy:
    """A regular dataclass that contains validated fingys."""

    name: str
    validated: ValidatedFingy
    nested: NestedValidatedFingy
    count: int = 10


# Test fixtures - tracked classes


@track
class SimpleModel:
    """A model that accepts a validated fingy."""

    def __init__(self, fingy: ValidatedFingy):
        self.fingy = fingy
        # Fingy is always a dataclass now (after our fixes)
        self.value = fingy.value


@track
class ComplexModel:
    """A model with multiple fingy parameters."""

    def __init__(self, main_fingy: ValidatedFingy, nested_fingy: NestedValidatedFingy):
        self.main_fingy = main_fingy
        self.nested_fingy = nested_fingy


@track
class LazyModel:
    """A model that accepts a lazy component."""

    def __init__(self, component: Lazy[SimpleModel], fingy: ValidatedFingy):
        self.component = component
        self.fingy = fingy

    def activate(self):
        """Instantiate the lazy component."""
        return self.component.instantiate()


@track
class FingyTypeTestModel:
    """Model for testing fingy type behavior."""

    def __init__(self, fingy):
        self.fingy = fingy
        self.fingy_type = type(fingy).__name__


# Tests for validation


def test_pydantic_dataclass_validation_on_creation():
    """Test that Pydantic dataclasses validate on creation."""
    # Valid fingy
    fingy = ValidatedFingy(value=50, name="test")
    assert fingy.value == 50
    assert fingy.name == "test"
    assert fingy.rate == 0.5  # default

    # Invalid value (> 100)
    with pytest.raises(ValidationError) as exc_info:
        ValidatedFingy(value=150, name="test")
    assert "less than or equal to 100" in str(exc_info.value)

    # Invalid name (empty)
    with pytest.raises(ValidationError) as exc_info:
        ValidatedFingy(value=50, name="")
    assert "at least 1 character" in str(exc_info.value)

    # Invalid rate (> 1.0)
    with pytest.raises(ValidationError) as exc_info:
        ValidatedFingy(value=50, name="test", rate=1.5)
    assert "less than or equal to 1" in str(exc_info.value)


def test_nested_pydantic_dataclass_validation():
    """Test validation of nested Pydantic dataclasses."""
    # Valid nested fingy
    fingy = NestedValidatedFingy(dimensions=[128, 64, 32], dropout=0.2)
    assert fingy.dimensions == [128, 64, 32]
    assert fingy.dropout == 0.2
    assert fingy.enabled is True  # default

    # Invalid: too many dimensions
    with pytest.raises(ValidationError) as exc_info:
        NestedValidatedFingy(
            dimensions=[512, 256, 128, 64, 32, 16]  # > 5 items
        )
    assert "at most 5 items" in str(exc_info.value)

    # Invalid: dropout >= 1.0
    with pytest.raises(ValidationError) as exc_info:
        NestedValidatedFingy(dimensions=[128], dropout=1.0)
    assert "less than 1" in str(exc_info.value)


def test_custom_types_in_dataclass():
    @track
    class Foo:
        def __init__(self, a_string: str):
            self.a_string = a_string

    @track
    class Bar:
        def __init__(self, an_int: int):
            self.an_int = an_int

    @pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
    class MyFingy:
        foo: Foo
        string_list: list[str]

    # This should work
    MyFingy(Foo("a string"), ["a", "b"])

    with pytest.raises(ValidationError):
        # Not a list[str]
        MyFingy(Foo("a string"), [1.0, 2.0])

    with pytest.raises(ValidationError):
        # Wrong custom class
        MyFingy(Bar(1), ["a", "b"])


# Tests for serialization


def test_pydantic_dataclass_serialization():
    """Test that Pydantic dataclasses can be serialized and deserialized."""
    fingy = ValidatedFingy(value=75, name="serialize_test", rate=0.25)

    # Serialize
    serialized = serialize_fingy(fingy)
    assert isinstance(serialized, dict)
    assert serialized["_confingy_dataclass"] is True
    assert serialized["_confingy_fields"]["value"] == 75
    assert serialized["_confingy_fields"]["name"] == "serialize_test"

    # Deserialize
    deserialized = deserialize_fingy(serialized)
    assert deserialized.value == 75
    assert deserialized.name == "serialize_test"
    assert deserialized.rate == 0.25


def test_pydantic_dataclass_save_load(tmp_path):
    """Test saving and loading Pydantic dataclasses."""
    fingy = ValidatedFingy(value=80, name="save_test", rate=0.8)

    # Save
    fingy_file = tmp_path / "pydantic_fingy.json"
    save_fingy(fingy, str(fingy_file))

    # Verify JSON structure
    with open(fingy_file) as f:
        data = json.load(f)
    assert data["_confingy_dataclass"] is True
    assert data["_confingy_class"] == "ValidatedFingy"

    # Load
    loaded = load_fingy(str(fingy_file))
    assert loaded.value == 80
    assert loaded.name == "save_test"
    assert loaded.rate == 0.8


def test_mixed_dataclass_serialization(tmp_path):
    """Test serialization of regular dataclasses containing Pydantic dataclasses."""
    validated = ValidatedFingy(value=30, name="validated")
    nested = NestedValidatedFingy(dimensions=[256, 128], dropout=0.3)

    mixed = MixedFingy(name="mixed_test", validated=validated, nested=nested, count=42)

    # Save and load
    fingy_file = tmp_path / "mixed_fingy.json"
    save_fingy(mixed, str(fingy_file))
    loaded = load_fingy(str(fingy_file))

    # Check all fields are preserved
    assert loaded.name == "mixed_test"
    assert loaded.validated.value == 30
    assert loaded.validated.name == "validated"
    assert loaded.nested.dimensions == [256, 128]
    assert loaded.nested.dropout == 0.3
    assert loaded.count == 42


# Tests with lazy loading


def test_pydantic_dataclass_with_lazy():
    """Test that Pydantic dataclasses work with lazy instantiation."""
    fingy = ValidatedFingy(value=25, name="lazy_test")

    # Create lazy model with validated fingy
    lazy_model = lazy(SimpleModel, fingy=fingy)

    # Lazy instance should exist
    assert hasattr(lazy_model, "_confingy_lazy_info")
    assert hasattr(lazy_model, "instantiate")

    # Instantiate
    model = lazy_model.instantiate()
    assert isinstance(model, SimpleModel)
    assert model.value == 25


def test_lazy_with_multiple_fingys():
    """Test lazy instantiation with multiple fingy parameters."""
    main_fingy = ValidatedFingy(value=40, name="main")
    nested_fingy = NestedValidatedFingy(dimensions=[512, 256])

    # Create lazy model
    lazy_model = lazy(ComplexModel, main_fingy=main_fingy, nested_fingy=nested_fingy)

    # Instantiate
    model = lazy_model.instantiate()
    assert isinstance(model, ComplexModel)
    # After fix: Fingys remain as dataclasses when going through lazy
    assert isinstance(model.main_fingy, ValidatedFingy)
    assert isinstance(model.nested_fingy, NestedValidatedFingy)
    assert model.main_fingy.value == 40
    assert model.nested_fingy.dimensions == [512, 256]


def test_lazy_model_with_lazy_component():
    """Test a model that accepts a Lazy component as a parameter."""
    fingy1 = ValidatedFingy(value=10, name="component")
    fingy2 = ValidatedFingy(value=20, name="container")

    # Create lazy component
    lazy_component = lazy(SimpleModel, fingy=fingy1)

    # Create model with lazy component
    model = LazyModel(component=lazy_component, fingy=fingy2)

    # The component should still be lazy
    assert hasattr(model.component, "instantiate")

    # Activate (instantiate the component)
    component = model.activate()
    assert isinstance(component, SimpleModel)
    assert component.value == 10


def test_lazy_model_itself_lazy():
    """Test making a lazy version of a model that accepts lazy components."""
    fingy1 = ValidatedFingy(value=15, name="inner")
    fingy2 = ValidatedFingy(value=30, name="outer")

    # Create lazy component
    lazy_component = lazy(SimpleModel, fingy=fingy1)

    # Create lazy version of LazyModel
    lazy_lazy_model = lazy(LazyModel, component=lazy_component, fingy=fingy2)

    # Should be lazy
    assert hasattr(lazy_lazy_model, "instantiate")

    # Instantiate the outer model
    model = lazy_lazy_model.instantiate()
    assert isinstance(model, LazyModel)

    # The inner component should still be lazy
    assert hasattr(model.component, "instantiate")

    # Can still activate the inner component
    component = model.activate()
    assert isinstance(component, SimpleModel)
    assert component.value == 15


def test_pydantic_dataclass_lazy_serialization(tmp_path):
    """Test serialization of lazy instances with Pydantic dataclasses."""
    fingy = ValidatedFingy(value=60, name="serialize_lazy")
    lazy_model = lazy(SimpleModel, fingy=fingy)

    # Serialize
    data = {"fingy": fingy, "lazy_model": lazy_model}

    fingy_file = tmp_path / "lazy_pydantic.json"
    save_fingy(data, str(fingy_file))

    # Load
    loaded = load_fingy(str(fingy_file))

    # Fingy should be loaded as dataclass
    assert loaded["fingy"].value == 60
    assert loaded["fingy"].name == "serialize_lazy"

    # Lazy model should still be lazy
    assert hasattr(loaded["lazy_model"], "instantiate")

    # Instantiate
    model = loaded["lazy_model"].instantiate()
    assert isinstance(model, SimpleModel)
    assert model.value == 60


def test_complex_nested_lazy_serialization(tmp_path):
    """Test serialization of complex nested structures with lazy and Pydantic fingys."""
    # Create fingys
    fingy1 = ValidatedFingy(value=100, name="first")

    # Test that invalid value fails validation
    with pytest.raises(ValidationError):
        ValidatedFingy(value=200, name="second")  # value > 100

    # Use valid value for fingy2
    fingy2 = ValidatedFingy(value=99, name="second")
    nested = NestedValidatedFingy(dimensions=[1024, 512, 256])

    # Create components
    lazy_model1 = lazy(SimpleModel, fingy=fingy1)
    lazy_model2 = lazy(SimpleModel, fingy=fingy2)
    lazy_complex = lazy(ComplexModel, main_fingy=fingy1, nested_fingy=nested)

    # Create mixed structure
    mixed = MixedFingy(name="complex_test", validated=fingy1, nested=nested, count=5)

    # Combine everything
    full_fingy = {
        "mixed": mixed,
        "models": [lazy_model1, lazy_model2],
        "complex": lazy_complex,
        "fingys": {"main": fingy1, "secondary": fingy2, "nested": nested},
    }

    # Save and load
    fingy_file = tmp_path / "complex_fingy.json"
    save_fingy(full_fingy, str(fingy_file))
    loaded = load_fingy(str(fingy_file))

    # Verify structure is preserved
    assert loaded["mixed"].name == "complex_test"
    assert loaded["mixed"].validated.value == 100
    assert len(loaded["models"]) == 2
    assert loaded["fingys"]["main"].value == 100
    assert loaded["fingys"]["secondary"].value == 99
    assert loaded["fingys"]["nested"].dimensions == [1024, 512, 256]

    # Instantiate lazy models
    model1 = loaded["models"][0].instantiate()
    model2 = loaded["models"][1].instantiate()
    complex_model = loaded["complex"].instantiate()

    assert model1.value == 100
    assert model2.value == 99
    # After save/load, fingys are restored as dataclasses (not dicts)
    # This is different from direct lazy() which converts to dicts
    assert isinstance(complex_model.main_fingy, ValidatedFingy)
    assert isinstance(complex_model.nested_fingy, NestedValidatedFingy)
    assert complex_model.main_fingy.value == 100
    assert complex_model.nested_fingy.dimensions == [1024, 512, 256]


def test_validation_happens_before_lazy():
    """Test that validation happens when creating the fingy, before passing to lazy."""
    # This should fail immediately when creating the fingy
    with pytest.raises(ValidationError) as exc_info:
        ValidatedFingy(value=500, name="invalid")
    assert "less than or equal to 100" in str(exc_info.value)

    # We never even get to create the lazy instance
    # because the fingy validation failed first


def test_dataclass_preserved_in_lazy():
    """Test that dataclasses are now preserved when passed through lazy."""
    fingy = ValidatedFingy(value=50, name="preserve_test")

    # Original is a dataclass
    assert hasattr(fingy, "value")
    assert fingy.value == 50

    # Create lazy model
    lazy_model = lazy(SimpleModel, fingy=fingy)

    # When instantiated, the fingy remains a dataclass (after fix)
    model = lazy_model.instantiate()
    # Fingy is always a dataclass now
    assert model.value == 50


def test_fingy_type_consistency(tmp_path):
    """Test that fingys now consistently remain as dataclasses."""
    fingy = ValidatedFingy(value=75, name="type_test")

    # Direct lazy: fingy now remains dataclass (after fix)
    lazy_direct = lazy(FingyTypeTestModel, fingy=fingy)
    model_direct = lazy_direct.instantiate()
    assert model_direct.fingy_type == "ValidatedFingy"
    assert model_direct.fingy.value == 75

    # After save/load: fingy also remains dataclass
    save_data = {"lazy_model": lazy(FingyTypeTestModel, fingy=fingy)}
    fingy_file = tmp_path / "type_test.json"
    save_fingy(save_data, str(fingy_file))
    loaded = load_fingy(str(fingy_file))

    model_loaded = loaded["lazy_model"].instantiate()
    assert model_loaded.fingy_type == "ValidatedFingy"
    assert model_loaded.fingy.value == 75

    # Fingys are now consistent!
