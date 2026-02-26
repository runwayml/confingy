"""
Tests for confingy.serde module - serialization handlers and context.
"""

import enum
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import pytest

from confingy import Lazy, deserialize_fingy, serialize_fingy, track
from confingy.exceptions import SerializationError
from tests.conftest import (
    CollectionFingy,
    ComplexFingy,
    NestedFingy,
    WithMethods,
    WithNones,
    standalone_function,
)


# Test dataclasses defined at module level for proper import/export
@dataclass
class ConfigWithTypes:
    """Config that contains type references."""

    int_type: type[int]
    str_type: type[str]
    list_type: type[list]
    dict_type: type[dict]


@dataclass
class ConfigWithCustomType:
    """Config with custom type and instance."""

    custom_type: type[WithMethods]
    custom_type_instance: WithMethods


@dataclass
class ConfigWithGenericTypes:
    """Config with generic types from typing module."""

    list_type: type[list]
    dict_type: type[dict]
    optional_type: type[Optional[str]]


# Test enums defined at module level for proper import/export during deserialization
class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Status(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def test_primitive_types():
    """Test that primitive types are serialized and deserialized correctly."""
    # Create a simple dictionary with primitives
    data = {
        "integer": 42,
        "float": 3.14,
        "string": "hello",
        "boolean": True,
        "none": None,
    }

    # Serialize and deserialize
    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    # Check types and values
    assert isinstance(deserialized["integer"], int)
    assert isinstance(deserialized["float"], float)
    assert isinstance(deserialized["string"], str)
    assert isinstance(deserialized["boolean"], bool)
    assert deserialized["none"] is None

    # Check values
    assert deserialized["integer"] == 42
    assert deserialized["float"] == 3.14
    assert deserialized["string"] == "hello"
    assert deserialized["boolean"]


def test_collections():
    """Test serialization and deserialization of collections."""

    # Create a config with various collections
    config = CollectionFingy(
        numbers_list=[1, 2, 3, 4, 5],
        numbers_tuple=(10, 20, 30, 40, 50),
        mapping={"pi": 3.14, "e": 2.718, "sqrt2": 1.414},
        nested=[{"odds": [1, 3, 5]}, {"evens": [2, 4, 6]}],
    )

    # Save and load
    serialized = serialize_fingy(config)
    deserialized = deserialize_fingy(serialized)

    # Check that types are preserved (especially tuple)
    assert isinstance(deserialized.numbers_list, list)
    assert isinstance(deserialized.numbers_tuple, tuple)
    assert isinstance(deserialized.mapping, dict)
    assert isinstance(deserialized.nested, list)
    assert isinstance(deserialized.nested[0], dict)
    assert isinstance(deserialized.nested[0]["odds"], list)

    # Check values
    assert deserialized.numbers_list == [1, 2, 3, 4, 5]
    assert deserialized.numbers_tuple == (10, 20, 30, 40, 50)
    assert deserialized.mapping["pi"] == 3.14
    assert deserialized.nested[0]["odds"] == [1, 3, 5]
    assert deserialized.nested[1]["evens"] == [2, 4, 6]


def test_functions():
    """Test serializing and deserializing functions and methods."""

    # Create an object with a tracked instance
    tracked_instance = track(WithMethods(multiplier=3))

    # Create a dict with functions
    funcs = {"standalone": standalone_function, "method": tracked_instance.method}

    # Serialize and deserialize
    serialized = serialize_fingy(funcs)

    # Check that the tracked method's object includes class_hash in serialization
    method_obj = serialized["method"]["_confingy_object"]
    assert "_confingy_class_hash" in method_obj
    assert isinstance(method_obj["_confingy_class_hash"], str)
    assert len(method_obj["_confingy_class_hash"]) == 64  # SHA256 hash

    deserialized = deserialize_fingy(serialized)

    # Test that functions still work
    assert deserialized["standalone"](5) == 10
    assert deserialized["method"](5) == 15


def test_nested_dataclasses():
    """Test with nested dataclasses with various field types."""

    # Create a complex config
    config = ComplexFingy(
        simple_field=42,
        optional_field=None,
        nested=NestedFingy(name="main", values=[1, 2, 3]),
        multiple_nested=[
            NestedFingy(name="first", values=[4, 5, 6]),
            NestedFingy(name="second", values=[7, 8, 9]),
        ],
    )

    # Serialize and deserialize
    serialized = serialize_fingy(config)
    deserialized = deserialize_fingy(serialized)

    # Check all fields
    assert deserialized.simple_field == 42
    assert deserialized.optional_field is None
    assert deserialized.nested.name == "main"
    assert deserialized.nested.values == [1, 2, 3]
    assert len(deserialized.multiple_nested) == 2
    assert deserialized.multiple_nested[0].name == "first"
    assert deserialized.multiple_nested[1].values == [7, 8, 9]


def test_nones():
    """Test handling of None values in various positions."""

    config = WithNones(
        maybe_value=None, maybe_list=None, list_with_nones=[1, None, 3, None, 5]
    )

    serialized = serialize_fingy(config)
    deserialized = deserialize_fingy(serialized)

    assert deserialized.maybe_value is None
    assert deserialized.maybe_list is None
    assert deserialized.list_with_nones == [1, None, 3, None, 5]


def test_error_cases():
    """Test various serialization error conditions."""

    # Test importing non-existent module
    bad_import = {
        "_confingy_class": "NonExistentClass",
        "_confingy_module": "non_existent_module",
        "_confingy_init": {},
    }

    # Should raise DeserializationError
    with pytest.raises(Exception):
        deserialize_fingy(bad_import)


def test_unserializable_objects():
    """Test handling of objects that can't be serialized."""
    from confingy.exceptions import SerializationError

    # Lambda functions should raise SerializationError
    lambda_func = lambda x: x * 2  # noqa: E731

    with pytest.raises(SerializationError, match="lambdas and local functions"):
        serialize_fingy({"func": lambda_func})

    # Local functions should also raise SerializationError
    def local_func(x):
        return x * 2

    with pytest.raises(SerializationError, match="lambdas and local functions"):
        serialize_fingy({"func": local_func})

    # Local types should raise SerializationError
    class LocalClass:
        pass

    with pytest.raises(SerializationError, match="local or lambda types"):
        serialize_fingy({"cls": LocalClass})


def test_circular_reference_protection():
    """Test that circular references are handled gracefully."""

    # Create a circular reference
    data = {"key": "value"}
    data["self"] = data

    # Should not crash due to infinite recursion
    # The depth protection should kick in
    from confingy.exceptions import SerializationError

    with pytest.raises(SerializationError, match="Maximum recursion depth"):
        serialize_fingy(data)


def test_serialization_path_tracking():
    """Test that serialization context tracks paths for better error messages."""

    # Create nested data where serialization might fail
    from dataclasses import dataclass

    @dataclass
    class NestedData:
        value: object  # This will contain something unserializable

    # Use a complex object that will cause serialization issues
    complex_obj = type("ComplexType", (), {"__module__": None})()

    config = NestedData(value=complex_obj)

    # Should include path information in any errors
    try:
        serialize_fingy(config)
    except Exception as e:
        # Error should contain some path information
        assert "value" in str(e) or "NestedData" in str(e)


def test_handler_order_matters():
    """Test that handler registration order affects which handler processes objects."""
    from confingy.serde import HandlerRegistry

    handlers = HandlerRegistry.get_default_handlers()

    handler_names = [h.__class__.__name__ for h in handlers]

    # EnumHandler must come before PrimitiveHandler (StrEnum/IntEnum are str/int)
    enum_index = handler_names.index("EnumHandler")
    primitive_index = handler_names.index("PrimitiveHandler")
    assert enum_index < primitive_index

    # Lazy and TrackedInstance should come before generic handlers
    lazy_index = handler_names.index("LazyHandler")
    tracked_index = handler_names.index("TrackedInstanceHandler")
    collection_index = handler_names.index("CollectionHandler")

    assert lazy_index < collection_index
    assert tracked_index < collection_index


def test_empty_collections():
    """Test serialization of empty collections."""
    data = {
        "empty_list": [],
        "empty_dict": {},
        "empty_tuple": (),
        "empty_set": set(),
    }

    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    assert deserialized["empty_list"] == []
    assert deserialized["empty_dict"] == {}
    assert deserialized["empty_tuple"] == ()
    assert deserialized["empty_set"] == set()


def test_sets():
    """Test serialization and deserialization of sets."""
    data = {
        "numbers": {1, 2, 3, 4, 5},
        "strings": {"a", "b", "c"},
    }

    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    # Check types are preserved
    assert isinstance(deserialized["numbers"], set)
    assert isinstance(deserialized["strings"], set)

    # Check values
    assert deserialized["numbers"] == {1, 2, 3, 4, 5}
    assert deserialized["strings"] == {"a", "b", "c"}


def test_nested_tuples_and_sets():
    """Test nested tuples and sets preserve their types."""
    data = {
        "tuple_of_sets": ({1, 2}, {3, 4}),
        "set_in_list": [{1, 2, 3}],
        "tuple_in_dict": {"key": (1, 2, 3)},
    }

    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    # Check nested types
    assert isinstance(deserialized["tuple_of_sets"], tuple)
    assert isinstance(deserialized["tuple_of_sets"][0], set)
    assert isinstance(deserialized["tuple_of_sets"][1], set)
    assert isinstance(deserialized["set_in_list"], list)
    assert isinstance(deserialized["set_in_list"][0], set)
    assert isinstance(deserialized["tuple_in_dict"]["key"], tuple)

    # Check values
    assert deserialized["tuple_of_sets"] == ({1, 2}, {3, 4})
    assert deserialized["set_in_list"] == [{1, 2, 3}]
    assert deserialized["tuple_in_dict"]["key"] == (1, 2, 3)


def test_dict_with_confingy_keys():
    """Test that user dicts with _confingy_ keys are preserved correctly."""
    # User dict that happens to contain keys matching our wrapper format
    data = {
        "user_dict_with_tuple_key": {"_confingy_tuple": [1, 2], "other": "value"},
        "user_dict_with_set_key": {"_confingy_set": [3, 4], "extra": 42},
        # Edge case: single key matching our format but not a boolean True value
        "single_tuple_key": {"_confingy_tuple": [1, 2]},
        "single_set_key": {"_confingy_set": [3, 4]},
    }

    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    # These should remain as dicts, not be converted to tuple/set
    assert isinstance(deserialized["user_dict_with_tuple_key"], dict)
    assert isinstance(deserialized["user_dict_with_set_key"], dict)
    assert deserialized["user_dict_with_tuple_key"]["_confingy_tuple"] == [1, 2]
    assert deserialized["user_dict_with_tuple_key"]["other"] == "value"
    assert deserialized["user_dict_with_set_key"]["_confingy_set"] == [3, 4]
    assert deserialized["user_dict_with_set_key"]["extra"] == 42

    # Single-key dicts should also be preserved as dicts (value is not True)
    assert isinstance(deserialized["single_tuple_key"], dict)
    assert isinstance(deserialized["single_set_key"], dict)
    assert deserialized["single_tuple_key"]["_confingy_tuple"] == [1, 2]
    assert deserialized["single_set_key"]["_confingy_set"] == [3, 4]


def test_mixed_type_collections():
    """Test collections with mixed types."""
    data = {
        "mixed_list": [1, "string", 3.14, None, True],
        "mixed_dict": {
            "int": 42,
            "str": "hello",
            "float": 2.71,
            "bool": False,
            "none": None,
        },
    }

    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    assert deserialized["mixed_list"] == [1, "string", 3.14, None, True]
    assert deserialized["mixed_dict"]["int"] == 42
    assert deserialized["mixed_dict"]["str"] == "hello"
    assert deserialized["mixed_dict"]["float"] == 2.71
    assert deserialized["mixed_dict"]["bool"] is False
    assert deserialized["mixed_dict"]["none"] is None


def test_serialization_depth_limit():
    """Test that serialization respects maximum depth limits to prevent infinite recursion."""
    from confingy.exceptions import SerializationError
    from confingy.serde import HandlerRegistry, SerializationContext

    # Create a deeply nested structure
    deep_data = {"level": 0}
    current = deep_data
    for i in range(1, 100):  # Create a very deep structure
        current["next"] = {"level": i}
        current = current["next"]

    context = SerializationContext()
    # Register default handlers so it can process the nested dict
    for handler in HandlerRegistry.get_default_handlers():
        context.register_handler(handler)
    context._max_depth = 50  # Set a lower limit

    # Should raise SerializationError due to depth limit
    with pytest.raises(SerializationError, match="Maximum recursion depth"):
        context.serialize(deep_data)


def test_deserialization_depth_limit():
    """Test that deserialization respects maximum depth limits."""
    from confingy.exceptions import DeserializationError
    from confingy.serde import DeserializationContext, HandlerRegistry

    # Create a deeply nested structure
    deep_data = {"level": 0}
    current = deep_data
    for i in range(1, 100):
        current["next"] = {"level": i}
        current = current["next"]

    context = DeserializationContext()
    # Register all handlers
    for handler in HandlerRegistry.get_default_handlers():
        context.register_handler(handler)
    context._max_depth = 50

    # Should raise DeserializationError due to depth limit
    with pytest.raises(DeserializationError, match="Maximum recursion depth"):
        context.deserialize(deep_data)


def test_serialization_error_path_tracking():
    """Test that serialization errors include path information."""
    from confingy.serde import SerializationContext, SerializationError

    # Create a circular reference that will trigger depth limit
    circular_data = {"level": 0}
    circular_data["self_ref"] = circular_data

    nested_data = {"outer": {"inner": {"problematic": circular_data}}}

    context = SerializationContext()
    # Register handlers so it can process the nested dictionaries
    from confingy.serde import HandlerRegistry

    for handler in HandlerRegistry.get_default_handlers():
        context.register_handler(handler)
    context._max_depth = 10  # Set a low depth to trigger the error quickly

    # Should wrap the error and include path information
    with pytest.raises(SerializationError) as exc_info:
        context.serialize(nested_data)

    error_msg = str(exc_info.value)
    # Should include path information showing where the error occurred
    assert "path" in error_msg.lower() or "depth" in error_msg.lower()


def test_custom_serialization_handler():
    """Test creating and using custom serialization handlers."""
    from confingy.serde import (
        DeserializationContext,
        SerializationContext,
        SerializationHandler,
    )

    class CustomObject:
        def __init__(self, data):
            self.data = data

    class CustomHandler(SerializationHandler):
        def can_handle(self, obj):
            return isinstance(obj, CustomObject)

        def serialize(self, obj, context):
            return {"_custom_type": "CustomObject", "data": obj.data}

        def deserialize(self, data, context):
            if isinstance(data, dict) and data.get("_custom_type") == "CustomObject":
                return CustomObject(data["data"])
            return None

    # Test serialization
    custom_obj = CustomObject("test_data")
    context = SerializationContext()
    context.register_handler(CustomHandler())

    serialized = context.serialize(custom_obj)
    assert serialized["_custom_type"] == "CustomObject"
    assert serialized["data"] == "test_data"

    # Test deserialization
    deser_context = DeserializationContext()
    deser_context.register_handler(CustomHandler())

    deserialized = deser_context.deserialize(serialized)
    assert isinstance(deserialized, CustomObject)
    assert deserialized.data == "test_data"


def test_handler_fallback_behavior():
    """Test that handlers properly return None when they can't handle an object."""
    from confingy.serde import PrimitiveHandler

    handler = PrimitiveHandler()

    # Should handle primitives
    assert handler.can_handle(42)
    assert handler.can_handle("string")
    assert handler.can_handle(None)

    # Should not handle complex objects
    assert not handler.can_handle(object())
    assert not handler.can_handle([1, 2, 3])

    # Deserialize should return None for unhandled types
    assert handler.deserialize(object(), None) is None


def test_unserializable_object_raises_error():
    """Test that unserializable objects raise SerializationError."""
    from confingy.exceptions import SerializationError

    # Test with a lambda function (known unserializable)
    lambda_func = lambda x: x * 2  # noqa: E731
    data_with_lambda = {"func": lambda_func, "value": 42}

    # Lambda should raise SerializationError
    with pytest.raises(SerializationError, match="lambdas and local functions"):
        serialize_fingy(data_with_lambda)


def test_complex_error_scenarios():
    """Test various error scenarios in serialization/deserialization."""
    from confingy.exceptions import DeserializationError

    # Test with invalid module reference
    invalid_data = {
        "_confingy_class": "NonExistentClass",
        "_confingy_module": "definitely_not_a_real_module_name_12345",
        "_confingy_init": {},
    }

    with pytest.raises((DeserializationError, ImportError, ModuleNotFoundError)):
        deserialize_fingy(invalid_data)

    # Test with malformed confingy data that will trigger a handler but fail
    malformed_data = {
        "_confingy_class": "NonExistentClass",
        "_confingy_module": "tests.conftest",  # Valid module but class doesn't exist
        "_confingy_init": {},
    }

    with pytest.raises((DeserializationError, AttributeError)):
        deserialize_fingy(malformed_data)


def test_handler_registry():
    """Test the HandlerRegistry functionality."""
    from confingy.serde import HandlerRegistry

    handlers = HandlerRegistry.get_default_handlers()

    # Should return a list of handlers
    assert isinstance(handlers, list)
    assert len(handlers) > 0

    # Should include the expected handler types
    handler_names = [h.__class__.__name__ for h in handlers]
    expected_handlers = [
        "PrimitiveHandler",
        "LazyHandler",
        "TrackedInstanceHandler",
        "CollectionHandler",
        "TypeHandler",  # Added TypeHandler
    ]

    for expected in expected_handlers:
        assert expected in handler_names


def test_type_serialization():
    """Test that type objects are properly serialized and deserialized."""
    # Create a config with type references
    config = ConfigWithTypes(int_type=int, str_type=str, list_type=list, dict_type=dict)

    # Serialize the config
    serialized = serialize_fingy(config)

    # Check that types are serialized correctly
    assert serialized["_confingy_fields"]["int_type"]["_confingy_class"] == "type"
    assert serialized["_confingy_fields"]["int_type"]["_confingy_module"] == "builtins"
    assert serialized["_confingy_fields"]["int_type"]["_confingy_name"] == "int"

    assert serialized["_confingy_fields"]["str_type"]["_confingy_class"] == "type"
    assert serialized["_confingy_fields"]["str_type"]["_confingy_module"] == "builtins"
    assert serialized["_confingy_fields"]["str_type"]["_confingy_name"] == "str"

    # Deserialize the config
    deserialized = deserialize_fingy(serialized)

    # Check that types are deserialized correctly
    assert deserialized.int_type is int
    assert deserialized.str_type is str
    assert deserialized.list_type is list
    assert deserialized.dict_type is dict


def test_custom_type_serialization():
    """Test serialization of custom type objects."""
    # Track the custom class instance
    tracked_instance = track(WithMethods, multiplier=3)

    # Create config with the custom type
    config = ConfigWithCustomType(
        custom_type=WithMethods, custom_type_instance=tracked_instance
    )

    # Serialize
    serialized = serialize_fingy(config)

    # Check type is serialized
    assert serialized["_confingy_fields"]["custom_type"]["_confingy_class"] == "type"
    assert (
        serialized["_confingy_fields"]["custom_type"]["_confingy_module"]
        == "tests.conftest"
    )
    assert (
        serialized["_confingy_fields"]["custom_type"]["_confingy_name"] == "WithMethods"
    )

    # Check instance is also serialized
    assert (
        serialized["_confingy_fields"]["custom_type_instance"]["_confingy_class"]
        == "WithMethods"
    )
    assert (
        serialized["_confingy_fields"]["custom_type_instance"]["_confingy_init"][
            "multiplier"
        ]
        == 3
    )

    # Check that the tracked instance includes class_hash
    instance_data = serialized["_confingy_fields"]["custom_type_instance"]
    assert "_confingy_class_hash" in instance_data
    assert isinstance(instance_data["_confingy_class_hash"], str)
    assert len(instance_data["_confingy_class_hash"]) == 64

    # Deserialize
    deserialized = deserialize_fingy(serialized)

    # Check type is preserved
    assert deserialized.custom_type is WithMethods

    # Check instance works
    assert deserialized.custom_type_instance.multiplier == 3


def test_nested_type_serialization():
    """Test serialization of nested/generic types."""
    config = ConfigWithGenericTypes(
        list_type=list, dict_type=dict, optional_type=Optional
    )

    # Note: Generic types from typing module might behave differently
    # This test ensures we handle them gracefully
    serialized = serialize_fingy(config)
    deserialized = deserialize_fingy(serialized)

    # The exact behavior might vary, but it shouldn't crash
    assert deserialized.list_type is not None
    assert deserialized.dict_type is not None
    assert deserialized.optional_type is not None


def test_unserializable_objects_should_raise_error():
    # This class is not tracked or lazy, so it will not be serializable
    class UnserializableObject:
        def __init__(self):
            self.data = "some internal data"

    unserializable = UnserializableObject()

    with pytest.raises(SerializationError, match="No handler found"):
        serialize_fingy({"obj": unserializable})


# ============================================================================
# Tests for default kwargs tracking and strict deserialization
# ============================================================================


# Module-level test classes for deserialization tests (must be importable)
@track
class WithDefaultsForTest:
    def __init__(
        self, required: str, optional_int: int = 42, optional_str: str = "default"
    ):
        self.required = required
        self.optional_int = optional_int
        self.optional_str = optional_str


@track
class ConfigurableClassForTest:
    def __init__(self, name: str, value: int = 10):
        self.name = name
        self.value = value


@track
class CurrentClassForTest:
    def __init__(self, name: str):
        self.name = name


@track
class CurrentClassWithValueForTest:
    def __init__(self, name: str, value: int = 5):
        self.name = name
        self.value = value


@track
class LazyClassForTest:
    def __init__(self, size: int):
        self.size = size


@track
class LazyClassWithNameForTest:
    def __init__(self, size: int, name: str = "default"):
        self.size = size
        self.name = name


@track
class FlexibleClassForTest:
    def __init__(self, required: str, **kwargs):
        self.required = required
        self.extra = kwargs


@track
class SimpleClassForTest:
    def __init__(self, name: str):
        self.name = name


@track
class NormalClassForTest:
    def __init__(self, a: int, b: str = "default"):
        self.a = a
        self.b = b


def test_default_kwargs_are_tracked():
    """Test that default kwargs are included in serialized config."""
    from confingy import lazy

    # Create instance with only required arg
    instance = WithDefaultsForTest(required="test")

    # Serialize it
    serialized = serialize_fingy(instance)

    # Should include ALL kwargs, including defaults
    init_args = serialized["_confingy_init"]
    assert init_args["required"] == "test"
    assert init_args["optional_int"] == 42
    assert init_args["optional_str"] == "default"

    # Also test with lazy instances
    lazy_instance = lazy(WithDefaultsForTest, required="lazy_test")
    serialized_lazy = serialize_fingy(lazy_instance)

    config = serialized_lazy["_confingy_config"]
    assert config["required"] == "lazy_test"
    assert config["optional_int"] == 42
    assert config["optional_str"] == "default"


def test_default_kwargs_ensure_reproducibility():
    """Test that tracking defaults ensures reproducibility even if defaults change.

    This simulates the scenario where:
    1. A config is serialized with a class that has default=10
    2. The class definition changes to have default=20
    3. Deserializing the old config should still produce default=10
    """
    # Create and serialize with only the required arg (default value=10)
    instance = ConfigurableClassForTest(name="test")
    assert instance.value == 10  # Confirm default is used

    serialized = serialize_fingy(instance)

    # The serialized form should have captured value=10
    assert serialized["_confingy_init"]["value"] == 10

    # When deserializing, it should use the serialized value (10),
    # not whatever the current default might be
    deserialized = deserialize_fingy(serialized)
    assert deserialized.value == 10
    assert deserialized.name == "test"


def test_strict_deserialization_raises_on_extra_kwargs():
    """Test that strict=True raises DeserializationError when extra kwargs are found."""
    from confingy.exceptions import DeserializationError

    # Simulate a serialized config that has an extra kwarg 'old_param' that
    # no longer exists in the class definition
    serialized_with_extra = {
        "_confingy_class": "CurrentClassForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_init": {
            "name": "test",
            "old_param": "this_no_longer_exists",  # Extra kwarg
        },
    }

    # With strict=True (default), should raise error
    with pytest.raises(DeserializationError) as exc_info:
        deserialize_fingy(serialized_with_extra, strict=True)

    assert "old_param" in str(exc_info.value)
    assert "no longer exist" in str(exc_info.value)


def test_non_strict_deserialization_warns_on_extra_kwargs():
    """Test that strict=False warns but continues when extra kwargs are found."""
    import warnings

    # Simulate a serialized config with an extra kwarg
    serialized_with_extra = {
        "_confingy_class": "CurrentClassWithValueForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_init": {
            "name": "test",
            "value": 10,
            "removed_param": "this_was_removed",  # Extra kwarg
        },
    }

    # With strict=False, should warn but succeed
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = deserialize_fingy(serialized_with_extra, strict=False)

    # Should have emitted a warning
    assert len(caught_warnings) == 1
    assert "removed_param" in str(caught_warnings[0].message)
    assert "no longer exist" in str(caught_warnings[0].message)

    # Should still create the object with valid kwargs
    assert result.name == "test"
    assert result.value == 10


def test_strict_deserialization_with_lazy_instances():
    """Test that strict mode also works for lazy instance deserialization."""
    from confingy.exceptions import DeserializationError

    # Simulate a serialized lazy config with extra kwargs
    serialized_lazy_with_extra = {
        "_confingy_class": "LazyClassForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_lazy": True,
        "_confingy_config": {
            "size": 100,
            "deprecated_option": True,  # Extra kwarg
        },
    }

    # With strict=True, should raise error
    with pytest.raises(DeserializationError) as exc_info:
        deserialize_fingy(serialized_lazy_with_extra, strict=True)

    assert "deprecated_option" in str(exc_info.value)


def test_non_strict_lazy_deserialization_warns_and_continues():
    """Test that non-strict mode works for lazy instances."""
    import warnings

    serialized_lazy_with_extra = {
        "_confingy_class": "LazyClassWithNameForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_lazy": True,
        "_confingy_config": {
            "size": 50,
            "name": "custom",
            "old_option": "removed",  # Extra kwarg
        },
    }

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = deserialize_fingy(serialized_lazy_with_extra, strict=False)

    # Should have warned about extra kwarg
    assert len(caught_warnings) == 1
    assert "old_option" in str(caught_warnings[0].message)

    # Should return a lazy instance that can be instantiated
    assert hasattr(result, "_confingy_lazy_info")
    instance = result.instantiate()
    assert instance.size == 50
    assert instance.name == "custom"


def test_classes_with_var_kwargs_bypass_strict_check():
    """Test that classes accepting **kwargs don't trigger strict mode errors."""
    # Serialize with extra kwargs
    serialized = {
        "_confingy_class": "FlexibleClassForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_init": {
            "required": "test",
            "any_extra_param": "should_work",
            "another_extra": 123,
        },
    }

    # Should not raise even with strict=True because class accepts **kwargs
    result = deserialize_fingy(serialized, strict=True)
    assert result.required == "test"
    assert result.extra == {"any_extra_param": "should_work", "another_extra": 123}


def test_multiple_extra_kwargs_listed_in_error():
    """Test that error message lists all extra kwargs found."""
    from confingy.exceptions import DeserializationError

    serialized = {
        "_confingy_class": "SimpleClassForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_init": {
            "name": "test",
            "extra1": "one",
            "extra2": "two",
            "extra3": "three",
        },
    }

    with pytest.raises(DeserializationError) as exc_info:
        deserialize_fingy(serialized, strict=True)

    error_msg = str(exc_info.value)
    assert "extra1" in error_msg
    assert "extra2" in error_msg
    assert "extra3" in error_msg


def test_no_warning_when_no_extra_kwargs():
    """Test that no warning is emitted when all kwargs are valid."""
    import warnings

    serialized = {
        "_confingy_class": "NormalClassForTest",
        "_confingy_module": "tests.test_serde",
        "_confingy_init": {
            "a": 42,
            "b": "custom",
        },
    }

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = deserialize_fingy(serialized, strict=False)

    # No extra kwargs, so no warnings
    assert len(caught_warnings) == 0
    assert result.a == 42
    assert result.b == "custom"


# ============================================================================
# Tests for deserialize-then-reserialize (untracked classes)
# ============================================================================


class Downloader:
    """Test class that is NOT decorated with @track."""

    def __init__(self, uri_columns: dict[str, str]):
        self.uri_columns = uri_columns

    def __call__(self, row: dict) -> dict:
        return row


def _save_tracked_downloader(path: str):
    """Helper function to save a tracked Downloader in a separate process."""
    import json

    from confingy import serialize_fingy, track

    downloader = track(Downloader)(
        uri_columns={
            "video_uri": "video_uri",
            "image_uri": "image_uri",
        },
    )
    serialized = serialize_fingy(downloader)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=4)


def test_deserialize_then_reserialize_untracked_class(tmp_path):
    """Test that deserialized instances can be re-serialized even if the class isn't @track decorated.

    This tests the scenario where:
    1. Process A has a class wrapped with track() and serializes an instance
    2. Process B loads the serialized instance (class is NOT decorated with @track)
    3. Process B should be able to re-serialize the loaded instance
    """
    import multiprocessing as mp

    json_path = tmp_path / "downloader.json"

    # Save in a separate process where Downloader is wrapped with track()
    p = mp.Process(target=_save_tracked_downloader, args=(str(json_path),))
    p.start()
    p.join()

    assert json_path.exists()

    # Load and re-serialize in this process where Downloader is NOT decorated
    from confingy import load_fingy, serialize_fingy

    config = load_fingy(str(json_path))
    # This should work because deserialization now uses track() internally
    reserialized = serialize_fingy(config)

    assert reserialized["_confingy_class"] == "Downloader"
    assert reserialized["_confingy_init"]["uri_columns"] == {
        "video_uri": "video_uri",
        "image_uri": "image_uri",
    }


# Module-level classes for test_deserialization_skips_post_config_hook
_post_config_call_count = 0


@track
class _PostConfigChild:
    def __init__(self, value: int = 0):
        self.value = value


@track
class _PostConfigParent:
    def __init__(self, child: Lazy[_PostConfigChild]):
        self.child = child.instantiate()

    @classmethod
    def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
        global _post_config_call_count
        _post_config_call_count += 1
        # This would modify the config if called
        instance.child.value = instance.child.value + 100
        return instance


def test_deserialization_skips_post_config_hook():
    """Test that __post_config__ is not called during deserialization.

    When we serialize a config, the serialized state already reflects the
    result of __post_config__ having been run. Re-running it during
    deserialization could modify the config in undesirable ways.
    """
    global _post_config_call_count
    from confingy import deserialize_fingy, lazy, serialize_fingy

    _post_config_call_count = 0

    # Create a lazy Parent - this calls __post_config__ once
    lazy_parent = lazy(_PostConfigParent)(child=lazy(_PostConfigChild)(value=10))
    assert _post_config_call_count == 1
    assert lazy_parent.child.value == 110  # 10 + 100 from __post_config__

    # Serialize
    serialized = serialize_fingy(lazy_parent)
    assert serialized["_confingy_config"]["child"]["_confingy_config"]["value"] == 110

    # Reset counter
    _post_config_call_count = 0

    # Deserialize - should NOT call __post_config__
    deserialized = deserialize_fingy(serialized)
    assert _post_config_call_count == 0  # __post_config__ should not be called
    assert deserialized.child.value == 110  # Value should remain 110, not become 210

    # Modifying a nested value doesn't trigger parent's __post_config__
    # (it would trigger child's __post_config__ if one existed)
    deserialized.child.value = 50
    assert (
        _post_config_call_count == 0
    )  # Parent's hook not called for child modifications
    assert deserialized.child.value == 50

    # But modifying the parent's config directly DOES call __post_config__
    new_child = lazy(_PostConfigChild)(value=20)
    deserialized.child = new_child
    assert _post_config_call_count == 1  # __post_config__ called on parent modification
    assert deserialized.child.value == 120  # 20 + 100 from __post_config__

    # Re-serialization should work correctly
    reserialized = serialize_fingy(deserialized)
    assert reserialized["_confingy_config"]["child"]["_confingy_config"]["value"] == 120

    # Deserialize again - still should not call __post_config__
    _post_config_call_count = 0
    deserialized2 = deserialize_fingy(reserialized)
    assert _post_config_call_count == 0
    assert deserialized2.child.value == 120


# ============================================================================
# Tests for enum serialization/deserialization
# ============================================================================


def test_enum_serialize_deserialize():
    """Test round-trip serialization for Enum, IntEnum, and str Enum."""
    for member, cls_name in [
        (Color.RED, "Color"),
        (Priority.HIGH, "Priority"),
        (Status.ACTIVE, "Status"),
    ]:
        serialized = serialize_fingy(member)
        assert serialized["_confingy_enum"] is True
        assert serialized["_confingy_class"] == cls_name
        assert serialized["_confingy_name"] == member.name

        deserialized = deserialize_fingy(serialized)
        assert deserialized is member


@dataclass
class ConfigWithEnum:
    color: Color
    priority: Priority
    status: Status


def test_enum_in_dataclass():
    """Test enum fields inside a dataclass."""
    config = ConfigWithEnum(
        color=Color.BLUE, priority=Priority.MEDIUM, status=Status.INACTIVE
    )
    serialized = serialize_fingy(config)
    deserialized = deserialize_fingy(serialized)

    assert deserialized.color is Color.BLUE
    assert deserialized.priority is Priority.MEDIUM
    assert deserialized.status is Status.INACTIVE


def test_enum_in_collection():
    """Test enums inside lists, dicts, tuples, and sets."""
    data = {
        "colors": [Color.RED, Color.GREEN],
        "priority_tuple": (Priority.LOW, Priority.HIGH),
        "status_set": {Status.ACTIVE, Status.INACTIVE},
        "mapping": {"primary": Color.BLUE},
    }
    serialized = serialize_fingy(data)
    deserialized = deserialize_fingy(serialized)

    assert deserialized["colors"] == [Color.RED, Color.GREEN]
    assert deserialized["priority_tuple"] == (Priority.LOW, Priority.HIGH)
    assert deserialized["status_set"] == {Status.ACTIVE, Status.INACTIVE}
    assert deserialized["mapping"]["primary"] is Color.BLUE


@track
class WithEnumTracked:
    def __init__(self, color: Color, priority: Priority):
        self.color = color
        self.priority = priority


def test_enum_in_tracked_class():
    """Test enum as a @track constructor arg."""
    instance = WithEnumTracked(color=Color.GREEN, priority=Priority.LOW)
    serialized = serialize_fingy(instance)
    deserialized = deserialize_fingy(serialized)

    assert deserialized.color is Color.GREEN
    assert deserialized.priority is Priority.LOW


def test_enum_prettify():
    """Test prettification of enum members."""
    from confingy.fingy import prettify_serialized_fingy

    serialized = serialize_fingy(Color.RED)
    pretty = prettify_serialized_fingy(serialized)
    assert pretty == "tests.test_serde.Color.RED"


def test_enum_transpile():
    """Test transpilation of enum members."""
    from confingy.fingy import transpile_fingy

    serialized = serialize_fingy(Color.GREEN)
    code = transpile_fingy(serialized)
    assert "Color.GREEN" in code
    assert "from tests.test_serde import Color" in code
