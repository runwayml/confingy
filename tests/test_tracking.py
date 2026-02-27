"""
Tests for confingy.decorators module - lazy and track decorators.
"""

from typing import Any

import pytest

from confingy import lazy, serialize_fingy, track, update
from confingy.exceptions import ValidationError
from confingy.tracking import Lazy
from confingy.utils.hashing import hash_class
from confingy.utils.types import is_nonlazy_subclass_of
from tests.conftest import (
    Adder,
    Inner,
    Middle,
    Multiplier,
    MyDataloader,
    MyDataset,
    MyModel,
    Outer,
    ProcessorPipeline,
)


# Global test classes for update() tests
@track
class UpdateTestFoo:
    def __init__(self, bar: str, baz: int = 10):
        self.bar = bar
        self.baz = baz


@track
class UpdateTestComplex:
    def __init__(self, required: str, optional: int = 42, another: str | None = None):
        self.required = required
        self.optional = optional
        self.another = another


def test_no_config():
    """Test basic functionality without configuration files."""
    # Create a processor pipeline
    processor = ProcessorPipeline([Adder(10), Multiplier(2)])

    # Create a dataset
    dataset = MyDataset(num_samples=100, num_features=10, processor=processor)

    # Create a lazy dataloader and instantiate it
    lazy_dataloader = lazy(MyDataloader, dataset, batch_size=5)
    dataloader = lazy_dataloader.instantiate()

    # Create a lazy model and instantiate it
    lazy_model = lazy(MyModel, in_features=10, out_features=5)
    model = lazy_model.instantiate()

    # Iterate through the dataloader and print the processed data
    for batch in dataloader:
        model(batch)


def test_lazy_nested_instances():
    """Test that lazy instances can contain other lazy instances and be fully instantiated."""

    # Create nested lazy instances using the new API
    inner = lazy(Inner, value=42)
    middle = lazy(Middle, inner=inner)
    outer = lazy(Outer, middle=middle)

    # Serialize and deserialize
    from confingy import deserialize_fingy, serialize_fingy

    serialized = serialize_fingy(outer)
    deserialized = deserialize_fingy(serialized)

    # After deserialization, everything is lazy
    assert hasattr(deserialized, "_confingy_lazy_info")  # outer is lazy

    # Instantiate the outer level
    instantiated_outer = deserialized.instantiate()
    assert is_nonlazy_subclass_of(instantiated_outer, Outer)

    # The middle is still lazy at this point
    assert hasattr(instantiated_outer.middle, "_confingy_lazy_info")

    # Instantiate the middle level
    instantiated_middle = instantiated_outer.middle.instantiate()
    assert is_nonlazy_subclass_of(instantiated_middle, Middle)

    # The inner is still lazy at this point
    assert hasattr(instantiated_middle.inner, "_confingy_lazy_info")

    # Instantiate the inner level
    instantiated_inner = instantiated_middle.inner.instantiate()
    assert is_nonlazy_subclass_of(instantiated_inner, Inner)

    # Check the value was preserved
    assert instantiated_inner.value == 42


def test_track_variations():
    """Test track decorator used in different ways."""

    # As a class decorator
    @track
    class Tracked:
        def __init__(self, name: str, value: int):
            self.name = name
            self.value = value

    # As a function call with args
    undecorated_instance = track(Tracked, name="test", value=123)

    # As a function call with an existing instance
    existing = Tracked(name="existing", value=456)
    tracked_existing = track(existing)

    # Check serialization of all variants.
    s1 = serialize_fingy(Tracked(name="decorated", value=789))
    assert s1["_confingy_class"] == "Tracked"
    assert s1["_confingy_module"] == "tests.test_tracking"
    assert s1["_confingy_init"] == {"name": "decorated", "value": 789}
    assert "_confingy_class_hash" in s1  # Should include class_hash

    s2 = serialize_fingy(undecorated_instance)
    assert s2["_confingy_class"] == "Tracked"
    assert s2["_confingy_module"] == "tests.test_tracking"
    assert s2["_confingy_init"] == {"name": "test", "value": 123}
    assert "_confingy_class_hash" in s2  # Should include class_hash

    s3 = serialize_fingy(tracked_existing)
    assert s3["_confingy_class"] == "Tracked"
    assert s3["_confingy_module"] == "tests.test_tracking"
    assert s3["_confingy_init"] == {"name": "existing", "value": 456}
    assert "_confingy_class_hash" in s3  # Should include class_hash


def test_lack_of_type_hints_does_not_raise_validation_errors():
    """Test that classes without type hints can still be tracked."""

    @track
    class Foo:
        def __init__(self, bar):
            self.bar = bar

    # We should be able to pass any type to the constructor
    Foo(bar=42)
    Foo(bar="not a number")


def test_kwargs():
    @track
    class Foo:
        def __init__(self, a: int, **kwargs):
            self.a = a
            self.kwargs = kwargs

    instance = Foo(42, b=2, c=3)
    assert instance.a == 42
    assert instance.kwargs == {"b": 2, "c": 3}

    lazy_instance = lazy(Foo)(a=42, b=2, c=3)
    instance = lazy_instance.instantiate()
    assert instance.a == 42
    assert instance.kwargs == {"b": 2, "c": 3}


def test_any_type_works_same_as_no_type():
    """Test that Any type annotation works the same as no type annotation."""

    @track
    class Foo:
        def __init__(self, bar: Any):
            self.bar = bar

    # We should be able to pass any type to the constructor
    Foo(bar=42)
    Foo(bar="not a number")


def test_lazy_instance_repr():
    """Test Lazy repr format."""
    model = lazy(MyModel, in_features=5, out_features=10)

    # Repr should show class name and config
    repr_str = repr(model)
    assert "Lazy" in repr_str
    assert "MyModel" in repr_str
    assert "in_features" in repr_str


def test_track_successful_validation_errors():
    @track
    class Foo:
        def __init__(self, bar: str, baz: int, qux: list[str] | None = None):
            self.bar = bar
            self.baz = baz
            self.qux = qux

    with pytest.raises(ValidationError):
        Foo("bar", "not an int")

    # Test lazy version.
    with pytest.raises(ValidationError):
        lazy(Foo, "bar", "not an int")

    with pytest.raises(ValidationError):
        # Not a string
        Foo(123, 1)

    with pytest.raises(ValidationError):
        # Not a list[str]
        Foo("bar", 1, qux=[1.0, 2.0])


def test_lazy_instance_config_access():
    """Test getting configuration from lazy instances."""
    model = lazy(MyModel, in_features=3, out_features=6)

    # Should be able to get config
    config = model.get_config()
    assert config == {"in_features": 3, "out_features": 6}

    # Config should be a copy (modifying it shouldn't affect the original)
    config["in_features"] = 999
    original_config = model.get_config()
    assert original_config["in_features"] == 3


def test_explicit_instantiation():
    """Test explicit instantiation."""
    model = lazy(MyModel, in_features=2, out_features=4)

    # Explicit instantiation
    actual_model = model.instantiate()
    assert actual_model.in_features == 2
    assert actual_model.out_features == 4

    # Should be able to call methods on the actual instance
    result = actual_model([1, 2, 3])
    assert result == [6, 8, 10]  # [1*2+4, 2*2+4, 3*2+4]


def test_lazy_decorator_with_parentheses():
    """Test that the new lazy() function API works correctly."""

    @track
    class TestModel:
        def __init__(self, size: int):
            self.size = size

    # Create lazy instance using the new API
    instance = lazy(TestModel, size=42)
    assert hasattr(instance, "_confingy_lazy_info")

    # Explicit instantiation required
    actual_instance = instance.instantiate()
    assert actual_instance.size == 42


def test_track_decorator_with_parentheses():
    """Test that @track() (with parentheses) works correctly."""

    @track()
    class TestTracker:
        def __init__(self, name: str):
            self.name = name

    instance = TestTracker(name="test")
    assert hasattr(instance, "_tracked_info")

    serialized = serialize_fingy(instance)
    assert serialized["_confingy_class"] == "TestTracker"
    assert serialized["_confingy_init"]["name"] == "test"


def test_lazy_instance_no_auto_instantiation():
    """Test that lazy instances expose config as attributes without instantiating."""
    model = lazy(MyModel, in_features=5, out_features=10)

    # Accessing a config attribute should return the config value
    assert model.in_features == 5
    assert model.out_features == 10

    # Accessing a non-existent attribute should raise an error
    with pytest.raises(AttributeError) as exc_info:
        _ = model.nonexistent
    assert "has no parameter 'nonexistent'" in str(exc_info.value)

    # Calling Lazy with args returns a new Lazy with merged config
    updated = model(in_features=20)
    assert updated.in_features == 20
    assert updated.out_features == 10  # Preserved from original


def test_lazy_decorator_edge_cases():
    """Test edge cases and error conditions in the lazy function."""

    # Test applying lazy() to a non-class should raise TypeError
    with pytest.raises(TypeError, match="lazy\\(\\) requires a class"):
        lazy(42)

    with pytest.raises(TypeError, match="lazy\\(\\) requires a class"):
        lazy("not_a_class")


def test_lazy_instance_with_validation_error():
    """Test Lazy behavior when validation fails."""

    @lazy
    class StrictClass:
        def __init__(self, number: int, text: str):
            self.number = number
            self.text = text

    # Should raise ValidationError when validation fails
    with pytest.raises(ValidationError) as exc_info:
        StrictClass(number="not_a_number", text=123)

    error = exc_info.value
    assert "StrictClass" in str(error)
    assert "number" in str(error)


def test_lazy_instance_calling_non_callable():
    """Test Lazy error when trying to call a non-callable instantiated object."""

    @lazy
    class NonCallableClass:
        def __init__(self, value: int):
            self.value = value

        # No __call__ method

    lazy_instance = NonCallableClass(value=42)
    real_instance = (
        lazy_instance.instantiate()
    )  # This creates a regular object, not callable

    # Should raise TypeError when trying to call a non-callable instance
    with pytest.raises(TypeError, match="object is not callable"):
        real_instance(123)


def test_lazy_instance_repr_with_many_config_fields():
    """Test Lazy repr when config has many fields (triggers ... truncation)."""

    @lazy
    class ManyFieldsClass:
        def __init__(
            self, field1: int, field2: str, field3: float, field4: bool, field5: int
        ):
            self.field1 = field1
            self.field2 = field2
            self.field3 = field3
            self.field4 = field4
            self.field5 = field5

    instance = ManyFieldsClass(
        field1=1, field2="test", field3=3.14, field4=True, field5=999
    )

    repr_str = repr(instance)
    # Should truncate and show "... and X more" when more than 3 fields
    assert "..." in repr_str
    assert "more" in repr_str
    assert "Lazy" in repr_str


def test_lazy_instance_with_original_cls():
    """Test Lazy with a class that has _original_cls attribute."""

    # Create a mock lazy factory function with _original_cls
    class OriginalClass:
        def __init__(self, value: int):
            self.value = value

    # Create a factory function that simulates what @lazy creates
    def factory_with_original(value: int):
        return Lazy(OriginalClass, {"value": value})

    # Add the _original_cls attribute like @lazy does
    factory_with_original._original_cls = OriginalClass

    # Now test creating a Lazy with this factory
    lazy_instance = Lazy(factory_with_original, {"value": 42})

    # Should use the _original_cls for validation
    assert lazy_instance._confingy_actual_cls == OriginalClass


def test_validation_model_creation_edge_cases():
    """Test edge cases in create_validation_model function."""
    from confingy.tracking import _create_validation_model

    # Test with a class that has no type hints
    class NoHintsClass:
        def __init__(self, arg):
            self.arg = arg

    # Should still create a validation model
    model = _create_validation_model(NoHintsClass)
    instance = model(arg="any_value")
    assert instance.arg == "any_value"

    # Test with a class that has Any type hints
    from typing import Any

    class AnyHintsClass:
        def __init__(self, arg: Any):
            self.arg = arg

    model = _create_validation_model(AnyHintsClass)
    instance = model(arg=12345)
    assert instance.arg == 12345


def test_track_with_validate_false():
    """Test that @track(_validate=False) skips validation."""

    @track(_validate=False)
    class NoValidation:
        def __init__(self, bar: str, baz: int):
            self.bar = bar
            self.baz = baz

    # Should not raise ValidationError even with wrong types
    instance = NoValidation("bar", "not an int")  # baz should be int but we pass str
    assert instance.bar == "bar"
    assert instance.baz == "not an int"

    # Check that tracking info is still stored
    assert hasattr(instance, "_tracked_info")
    assert instance._tracked_info["init_args"] == {"bar": "bar", "baz": "not an int"}

    # Test serialization still works
    serialized = serialize_fingy(instance)
    assert serialized["_confingy_class"] == "NoValidation"
    assert serialized["_confingy_init"]["bar"] == "bar"
    assert serialized["_confingy_init"]["baz"] == "not an int"


def test_track_with_validate_true():
    """Test that @track(_validate=True) performs validation (default behavior)."""

    @track(_validate=True)
    class WithValidation:
        def __init__(self, bar: str, baz: int):
            self.bar = bar
            self.baz = baz

    # Should raise ValidationError with wrong types
    with pytest.raises(ValidationError) as exc_info:
        WithValidation("bar", "not an int")

    error = exc_info.value
    assert "WithValidation" in str(error)

    # Should work with correct types
    instance = WithValidation("bar", 42)
    assert instance.bar == "bar"
    assert instance.baz == 42


def test_track_function_with_validate_false():
    """Test track(Class, _validate=False, ...) function call."""

    class UndecoredClass:
        def __init__(self, name: str, value: int):
            self.name = name
            self.value = value

    # Should not raise ValidationError even with wrong types
    instance = track(UndecoredClass, _validate=False, name="test", value="not_an_int")
    assert instance.name == "test"
    assert instance.value == "not_an_int"

    # Check that tracking info is still stored
    assert hasattr(instance, "_tracked_info")
    assert instance._tracked_info["init_args"] == {
        "name": "test",
        "value": "not_an_int",
    }


def test_track_decorator_parentheses_with_validate():
    """Test @track() with parentheses and _validate parameter."""

    @track()  # Default should be _validate=True
    class DefaultValidation:
        def __init__(self, x: int):
            self.x = x

    # Should validate by default
    with pytest.raises(ValidationError):
        DefaultValidation(x="not_an_int")

    # With correct type
    instance = DefaultValidation(x=10)
    assert instance.x == 10

    @track(_validate=False)
    class NoValidationWithParens:
        def __init__(self, x: int):
            self.x = x

    # Should not validate
    instance = NoValidationWithParens(x="not_an_int")
    assert instance.x == "not_an_int"


def test_track_mixed_validation_scenarios():
    """Test various mixed scenarios with validation flag."""

    # Test positional arguments with _validate=False
    @track(_validate=False)
    class PositionalArgs:
        def __init__(self, a: int, b: str, c: float):
            self.a = a
            self.b = b
            self.c = c

    # Wrong types should not raise error
    instance = PositionalArgs("not_int", 123, "not_float")
    assert instance.a == "not_int"
    assert instance.b == 123
    assert instance.c == "not_float"

    # Test with default values and _validate=False
    @track(_validate=False)
    class WithDefaults:
        def __init__(self, required: int, optional: str = "default"):
            self.required = required
            self.optional = optional

    # Wrong type for required should not raise error
    instance = WithDefaults(required="not_int")
    assert instance.required == "not_int"
    assert instance.optional == "default"

    # Test with **kwargs and _validate=False
    @track(_validate=False)
    class WithKwargs:
        def __init__(self, a: int, **kwargs):
            self.a = a
            self.kwargs = kwargs

    # Wrong type for 'a' should not raise error
    instance = WithKwargs("not_int", b=2, c=3)
    assert instance.a == "not_int"
    assert instance.kwargs == {"b": 2, "c": 3}


# Tests for update() functionality
def test_update_tracked_instance_basic():
    """Test updating a tracked instance with basic parameters."""
    # Create parent instance
    parent_foo = track(UpdateTestFoo)(bar="hello")
    assert parent_foo.bar == "hello"
    assert parent_foo.baz == 10

    # Update bar parameter
    child_foo = update(parent_foo)(bar="world")
    assert child_foo.bar == "world"
    assert child_foo.baz == 10  # Should preserve default

    # Verify child is tracked
    assert hasattr(child_foo, "_tracked_info")
    assert child_foo._tracked_info["init_args"]["bar"] == "world"
    # baz is not in init_args because it uses the default value

    # Verify parent is unchanged
    assert parent_foo.bar == "hello"


def test_update_tracked_instance_multiple_params():
    """Test updating multiple parameters at once."""
    parent_foo = track(UpdateTestFoo)(bar="hello", baz=20)

    # Update both parameters
    child_foo = update(parent_foo)(bar="world", baz=30)
    assert child_foo.bar == "world"
    assert child_foo.baz == 30

    # Update only one parameter
    child_foo2 = update(parent_foo)(baz=40)
    assert child_foo2.bar == "hello"  # Should preserve original
    assert child_foo2.baz == 40


def test_track_does_not_mutate_original_class():
    """
    Regression test: track(Class)(args) should not globally modify Class.

    Previously, calling track(MyClass)(a=3) would permanently replace
    MyClass.__init__ with a tracking wrapper, causing subsequent untracked
    usage of MyClass to go through pydantic validation unexpectedly.
    """

    class MyModule:
        def __init__(self, a: int):
            self.a = a

    # Untracked usage works (3.2 is not int, but no validation without track)
    x1 = MyModule(a=3.2)
    assert x1.a == 3.2

    # Tracked usage with correct type
    x2 = track(MyModule)(a=3)
    assert x2.a == 3
    assert hasattr(x2, "_tracked_info")

    # Original class should NOT be modified - untracked usage should still work
    x3 = MyModule(a=3.2)
    assert x3.a == 3.2
    assert not hasattr(x3, "_tracked_info")

    # Original class should not have a lazy classmethod
    assert "lazy" not in MyModule.__dict__


def test_track_preserves_generic_parameters():
    """
    Regression test: @track on a Generic class must preserve __parameters__
    so that runtime subscripting (e.g. UDF[Any, Any]) still works.
    """
    from typing import Any, Generic, TypeVar

    InputT = TypeVar("InputT")
    OutputT = TypeVar("OutputT")

    @track
    class GenericCls(Generic[InputT, OutputT]):
        def __init__(self, x: int):
            self.x = x

    # Runtime subscripting must work
    subscripted = GenericCls[Any, Any]
    assert subscripted is not None

    # Normal instantiation still works
    obj = GenericCls(x=42)
    assert obj.x == 42
    assert hasattr(obj, "_tracked_info")

    # Lazy still works
    lazy_obj = GenericCls.lazy(x=42)
    assert lazy_obj.instantiate().x == 42


def test_track_parent_and_child_class():
    """
    This is a regression test. Ensure that tracking both parent and child classes works.

    Previously, there was a bug where the child class would reference the parent class name
    if both were tracked.
    """

    class Parent:
        def __init__(self, x: int):
            self.x = x

    class Child(Parent):
        def __init__(self, x: int, y: str):
            super().__init__(x)
            self.y = y

    _ = track(Parent)(x=5)

    child = track(Child)(x=10, y="test")
    assert child._tracked_info["class"] == "Child"
    assert "y" in child._tracked_info["init_args"]


def test_update_lazy_instance_basic():
    """Test updating a lazy instance."""
    # Create lazy parent
    parent_lazy = lazy(UpdateTestFoo)(bar="hello")
    assert isinstance(parent_lazy, Lazy)

    # Update lazy instance
    child_lazy = update(parent_lazy)(bar="world")
    assert isinstance(child_lazy, Lazy)

    # Verify configuration was updated
    config = child_lazy.get_config()
    assert config["bar"] == "world"
    # baz is not in config because it uses the default value

    # Instantiate and verify
    child_instance = child_lazy.instantiate()
    assert child_instance.bar == "world"
    assert child_instance.baz == 10


def test_update_lazy_instance_multiple_params():
    """Test updating multiple parameters on lazy instance."""
    parent_lazy = lazy(UpdateTestFoo)(bar="hello", baz=20)

    # Update both parameters
    child_lazy = update(parent_lazy)(bar="world", baz=30)
    config = child_lazy.get_config()
    assert config["bar"] == "world"
    assert config["baz"] == 30

    # Update only one parameter
    child_lazy2 = update(parent_lazy)(baz=40)
    config2 = child_lazy2.get_config()
    assert config2["bar"] == "hello"  # Should preserve original
    assert config2["baz"] == 40


def test_update_with_positional_args():
    """Test that update works with positional arguments."""
    parent_foo = track(UpdateTestFoo)("initial")

    # Update with positional argument
    child_foo = update(parent_foo)("updated")
    assert child_foo.bar == "updated"
    assert child_foo.baz == 10

    # Update with mix of positional and keyword
    child_foo2 = update(parent_foo)("updated2", baz=25)
    assert child_foo2.bar == "updated2"
    assert child_foo2.baz == 25


def test_update_complex_class():
    """Test updating a class with multiple optional parameters."""
    parent = track(UpdateTestComplex)(required="test")
    assert parent.required == "test"
    assert parent.optional == 42
    assert parent.another is None

    # Update required field
    child1 = update(parent)(required="updated")
    assert child1.required == "updated"
    assert child1.optional == 42
    assert child1.another is None

    # Update optional fields
    child2 = update(parent)(optional=100, another="now set")
    assert child2.required == "test"
    assert child2.optional == 100
    assert child2.another == "now set"

    # Update all fields
    child3 = update(parent)(required="all new", optional=200, another="all set")
    assert child3.required == "all new"
    assert child3.optional == 200
    assert child3.another == "all set"


def test_update_validation():
    """Test that validation runs on updated instances."""
    parent_foo = track(UpdateTestFoo)(bar="hello")

    # This should work - correct types
    child_foo = update(parent_foo)(bar="world", baz=50)
    assert child_foo.bar == "world"
    assert child_foo.baz == 50

    # This should fail validation - wrong type for baz
    with pytest.raises(ValidationError):
        update(parent_foo)(bar="world", baz="not an int")


def test_update_preserves_type():
    """Test that the type of the updated object is preserved."""
    from confingy.utils.types import is_lazy_version_of

    parent_foo = track(UpdateTestFoo)(bar="hello")
    child_foo = update(parent_foo)(bar="world")

    # Child should be same type as parent
    assert type(child_foo) is type(parent_foo)
    assert isinstance(child_foo, UpdateTestFoo)

    # For lazy instances
    parent_lazy = lazy(UpdateTestFoo)(bar="hello")
    child_lazy = update(parent_lazy)(bar="world")

    assert type(child_lazy) is type(parent_lazy)
    assert is_lazy_version_of(child_lazy, UpdateTestFoo)

    # After instantiation
    child_instance = child_lazy.instantiate()
    assert isinstance(child_instance, UpdateTestFoo)


def test_update_chain():
    """Test chaining multiple updates."""
    # Start with parent
    parent = track(UpdateTestFoo)(bar="first", baz=1)

    # First update
    child1 = update(parent)(bar="second")
    assert child1.bar == "second"
    assert child1.baz == 1

    # Second update from child1
    child2 = update(child1)(baz=2)
    assert child2.bar == "second"
    assert child2.baz == 2

    # Third update from child2
    child3 = update(child2)(bar="third", baz=3)
    assert child3.bar == "third"
    assert child3.baz == 3

    # Verify all are independent
    assert parent.bar == "first"
    assert parent.baz == 1
    assert child1.bar == "second"
    assert child1.baz == 1
    assert child2.bar == "second"
    assert child2.baz == 2


def test_update_error_on_non_tracked():
    """Test that update raises appropriate error for non-tracked objects."""

    class NotTracked:
        def __init__(self, value: int):
            self.value = value

    obj = NotTracked(42)

    with pytest.raises(TypeError) as exc_info:
        update(obj)(value=50)

    assert "requires either a tracked instance" in str(exc_info.value)


def test_update_decorated_class():
    """Test updating instances of @track decorated classes."""
    # Create instance of decorated class
    parent = UpdateTestFoo(bar="hello")
    assert hasattr(parent, "_tracked_info")

    # Update it
    child = update(parent)(bar="world", baz=25)
    assert child.bar == "world"
    assert child.baz == 25
    assert hasattr(child, "_tracked_info")


def test_track_adds_class_hash():
    """Test that @track decorator adds class_hash to _tracked_info."""

    @track
    class TestModel:
        def __init__(self, size: int):
            self.size = size

    model = TestModel(size=50)

    assert hasattr(model, "_tracked_info")
    assert "class_hash" in model._tracked_info
    assert isinstance(model._tracked_info["class_hash"], str)
    assert len(model._tracked_info["class_hash"]) == 64  # SHA256 hash length
    assert model._tracked_info["class_hash"] == hash_class(TestModel)


def test_track_function_form_adds_class_hash():
    """Test track(Class, args) form adds class_hash."""

    class UntrackModel:
        def __init__(self, value: int):
            self.value = value

    model = track(UntrackModel, value=42)

    assert hasattr(model, "_tracked_info")
    assert "class_hash" in model._tracked_info
    assert model._tracked_info["class_hash"] == hash_class(UntrackModel)


def test_track_existing_instance_adds_class_hash():
    """Test adding tracking to existing instance includes class_hash."""

    class PlainModel:
        def __init__(self, data: str):
            self.data = data

    instance = PlainModel("test")
    tracked_instance = track(instance)

    assert hasattr(tracked_instance, "_tracked_info")
    assert "class_hash" in tracked_instance._tracked_info
    assert tracked_instance._tracked_info["class_hash"] == hash_class(PlainModel)


def test_track_skip_validation_includes_class_hash():
    """Test that class_hash works with validation skipped."""

    @track(_validate=False)
    class UnvalidatedModel:
        def __init__(self, data: Any):
            self.data = data

    model = UnvalidatedModel({"complex": "data"})
    assert "class_hash" in model._tracked_info
    assert isinstance(model._tracked_info["class_hash"], str)
    assert model._tracked_info["class_hash"] == hash_class(UnvalidatedModel)


def test_lazy_adds_class_hash():
    """Test that lazy() adds class_hash to _confingy_lazy_info."""
    lazy_model = lazy(MyModel, in_features=10, out_features=5)

    assert hasattr(lazy_model, "_confingy_lazy_info")
    assert "class_hash" in lazy_model._confingy_lazy_info
    assert isinstance(lazy_model._confingy_lazy_info["class_hash"], str)
    assert len(lazy_model._confingy_lazy_info["class_hash"]) == 64
    assert lazy_model._confingy_lazy_info["class_hash"] == hash_class(MyModel)


def test_lazy_factory_pattern_includes_class_hash():
    """Test lazy factory pattern includes class_hash."""
    lazy_factory = lazy(MyModel)  # Use MyModel instead which has simpler args
    lazy_model = lazy_factory(in_features=5, out_features=10)

    assert hasattr(lazy_model, "_confingy_lazy_info")
    assert "class_hash" in lazy_model._confingy_lazy_info
    assert lazy_model._confingy_lazy_info["class_hash"] == hash_class(MyModel)


def test_lazy_instantiation_preserves_class_hash():
    """Test that instantiating a lazy object preserves class_hash."""

    @track
    class TrackedModel:
        def __init__(self, value: int):
            self.value = value

    lazy_model = lazy(TrackedModel, value=10)
    lazy_hash = lazy_model._confingy_lazy_info["class_hash"]

    # Instantiate the model
    actual_model = lazy_model.instantiate()

    # The instantiated model should have tracking info with the same hash
    assert hasattr(actual_model, "_tracked_info")
    assert "class_hash" in actual_model._tracked_info
    assert actual_model._tracked_info["class_hash"] == lazy_hash


def test_track_adds_lazy_classmethod():
    """Test that @track decorator adds a .lazy() classmethod."""

    @track
    class TestModel:
        def __init__(self, size: int):
            self.size = size

    # Check that the lazy classmethod exists
    assert hasattr(TestModel, "lazy")
    assert callable(TestModel.lazy)


def test_lazy_classmethod_requires_args_when_no_defaults():
    """Test that .lazy() raises ValidationError when required args are missing."""

    @track
    class TestModel:
        def __init__(self, size: int, name: str):
            self.size = size
            self.name = name

    # .lazy() without args should raise ValidationError (no defaults)
    with pytest.raises(ValidationError):
        TestModel.lazy()

    # .lazy() with args should work
    lazy_model = TestModel.lazy(size=42, name="test")
    assert isinstance(lazy_model, Lazy)

    # Instantiate and verify
    actual_model = lazy_model.instantiate()
    assert actual_model.size == 42
    assert actual_model.name == "test"


def test_lazy_classmethod_direct_instantiation():
    """Test using the .lazy() classmethod with direct arguments."""

    @track
    class TestModel:
        def __init__(self, value: int):
            self.value = value

    # Direct instantiation: MyClass.lazy(args)
    lazy_model = TestModel.lazy(value=100)
    assert isinstance(lazy_model, Lazy)

    # Instantiate and verify
    actual_model = lazy_model.instantiate()
    assert actual_model.value == 100


def test_lazy_classmethod_equivalent_to_lazy_function():
    """Test that MyClass.lazy() is equivalent to lazy(MyClass)."""

    @track
    class TestModel:
        def __init__(self, x: int, y: str):
            self.x = x
            self.y = y

    # Create lazy instances using both methods
    lazy_via_classmethod = TestModel.lazy(x=10, y="hello")
    lazy_via_function = lazy(TestModel, x=10, y="hello")

    # Both should be Lazy instances
    assert isinstance(lazy_via_classmethod, Lazy)
    assert isinstance(lazy_via_function, Lazy)

    # Both should have the same configuration
    assert lazy_via_classmethod.get_config() == lazy_via_function.get_config()

    # Instantiate both and verify they behave the same
    instance1 = lazy_via_classmethod.instantiate()
    instance2 = lazy_via_function.instantiate()

    assert instance1.x == instance2.x == 10
    assert instance1.y == instance2.y == "hello"


class TestLazyAndClassmethodParity:
    """Test that .lazy() and lazy()() behave identically in all edge cases."""

    def test_parity_with_required_args_provided(self):
        """Both .lazy(args) and lazy(Cls)(args) should return equivalent Lazy instances."""

        @track
        class Model:
            def __init__(self, value: int, name: str):
                self.value = value
                self.name = name

        via_classmethod = Model.lazy(value=42, name="test")
        via_function = lazy(Model)(value=42, name="test")

        assert isinstance(via_classmethod, Lazy)
        assert isinstance(via_function, Lazy)
        assert via_classmethod.get_config() == via_function.get_config()

        # Instantiate and verify both work the same
        inst1 = via_classmethod.instantiate()
        inst2 = via_function.instantiate()
        assert inst1.value == inst2.value == 42
        assert inst1.name == inst2.name == "test"

    def test_parity_with_all_defaults(self):
        """Both .lazy() and lazy(Cls)() should return Lazy when all params have defaults."""

        @track
        class ModelWithDefaults:
            def __init__(self, value: int = 10, name: str = "default"):
                self.value = value
                self.name = name

        via_classmethod = ModelWithDefaults.lazy()
        via_function = lazy(ModelWithDefaults)()

        assert isinstance(via_classmethod, Lazy)
        assert isinstance(via_function, Lazy)
        assert via_classmethod.get_config() == via_function.get_config()

        # Instantiate and verify both work the same
        inst1 = via_classmethod.instantiate()
        inst2 = via_function.instantiate()
        assert inst1.value == inst2.value == 10
        assert inst1.name == inst2.name == "default"

    def test_parity_missing_required_args_both_fail(self):
        """Both .lazy() and lazy(Cls)() should raise ValidationError when required args missing."""

        @track
        class RequiredModel:
            def __init__(self, required_value: int):
                self.required_value = required_value

        # Both should raise ValidationError
        err1 = None
        err2 = None

        with pytest.raises(ValidationError) as exc1:
            RequiredModel.lazy()
        err1 = str(exc1.value)

        with pytest.raises(ValidationError) as exc2:
            lazy(RequiredModel)()
        err2 = str(exc2.value)

        # Error messages should be the same
        assert err1 == err2

    def test_parity_partial_args_both_fail(self):
        """Both should fail when only some required args are provided."""

        @track
        class MultiArgModel:
            def __init__(self, a: int, b: str, c: float):
                self.a = a
                self.b = b
                self.c = c

        # Only providing 'a', missing 'b' and 'c'
        with pytest.raises(ValidationError):
            MultiArgModel.lazy(a=1)

        with pytest.raises(ValidationError):
            lazy(MultiArgModel)(a=1)

    def test_parity_mixed_required_and_defaults(self):
        """Both should work when required args provided and defaults exist for others."""

        @track
        class MixedModel:
            def __init__(self, required: int, optional: str = "default"):
                self.required = required
                self.optional = optional

        via_classmethod = MixedModel.lazy(required=42)
        via_function = lazy(MixedModel)(required=42)

        assert isinstance(via_classmethod, Lazy)
        assert isinstance(via_function, Lazy)
        assert via_classmethod.get_config() == via_function.get_config()

        inst1 = via_classmethod.instantiate()
        inst2 = via_function.instantiate()
        assert inst1.required == inst2.required == 42
        assert inst1.optional == inst2.optional == "default"


def test_lazy_classmethod_with_validation():
    """Test that the .lazy() classmethod validates arguments."""

    @track
    class StrictModel:
        def __init__(self, number: int, text: str):
            self.number = number
            self.text = text

    # Should raise ValidationError with wrong types
    with pytest.raises(ValidationError) as exc_info:
        StrictModel.lazy(number="not_a_number", text=123)

    error = exc_info.value
    assert "StrictModel" in str(error)

    # Should work with correct types
    lazy_model = StrictModel.lazy(number=42, text="valid")
    actual_model = lazy_model.instantiate()
    assert actual_model.number == 42
    assert actual_model.text == "valid"


def test_lazy_classmethod_with_positional_args():
    """Test .lazy() classmethod with positional arguments."""

    @track
    class PositionalModel:
        def __init__(self, a: int, b: str):
            self.a = a
            self.b = b

    # .lazy() without args should raise ValidationError (no defaults)
    with pytest.raises(ValidationError):
        PositionalModel.lazy()

    # Direct instantiation with positional args should work
    lazy_model = PositionalModel.lazy(2, "direct")
    actual_model = lazy_model.instantiate()
    assert actual_model.a == 2
    assert actual_model.b == "direct"


def test_lazy_classmethod_with_nested_lazy():
    """Test that .lazy() classmethod works with nested lazy instances."""

    @track
    class Inner:
        def __init__(self, value: int):
            self.value = value

    @track
    class Outer:
        def __init__(self, inner: Lazy[Inner] | Inner):
            self.inner = inner

    # Create nested lazy using classmethod
    lazy_inner = Inner.lazy(value=42)
    lazy_outer = Outer.lazy(inner=lazy_inner)

    # Instantiate outer
    outer_instance = lazy_outer.instantiate()

    # Inner should still be lazy
    assert isinstance(outer_instance.inner, Lazy)

    # Instantiate inner
    inner_instance = outer_instance.inner.instantiate()
    assert inner_instance.value == 42


def test_track_error_on_existing_lazy_attribute():
    """Test that @track raises an error if the class already has a lazy attribute."""

    # Test with a lazy method
    with pytest.raises(AttributeError, match="already has a 'lazy' attribute"):

        @track
        class HasLazyMethod:
            def __init__(self, value: int):
                self.value = value

            def lazy(self):
                return "existing method"

    # Test with a lazy property
    with pytest.raises(AttributeError, match="already has a 'lazy' attribute"):

        @track
        class HasLazyProperty:
            def __init__(self, value: int):
                self.value = value

            @property
            def lazy(self):
                return "existing property"

    # Test with a lazy class variable
    with pytest.raises(AttributeError, match="already has a 'lazy' attribute"):

        @track
        class HasLazyClassVar:
            lazy = "existing class variable"

            def __init__(self, value: int):
                self.value = value


def test_lazy_classmethod_serialization():
    """Test that lazy instances created via classmethod can be serialized."""

    @track
    class SerializableModel:
        def __init__(self, data: str, count: int):
            self.data = data
            self.count = count

    # Create lazy instance via classmethod
    lazy_model = SerializableModel.lazy(data="test", count=5)

    # Serialize it
    serialized = serialize_fingy(lazy_model)

    # Should have the standard lazy serialization structure
    assert "_confingy_class" in serialized
    assert serialized["_confingy_class"] == "SerializableModel"
    assert "_confingy_lazy" in serialized
    assert serialized["_confingy_lazy"] is True
    assert "_confingy_config" in serialized
    assert serialized["_confingy_config"] == {"data": "test", "count": 5}

    # Note: We can't test deserialization here because the class is defined
    # locally in the test function and can't be reimported. But we can verify
    # that the serialization format is correct and compatible with existing code.


def test_lazy_classmethod_with_inheritance():
    """Test that .lazy() classmethod works correctly with child classes."""

    @track
    class Parent:
        def __init__(self, x: int):
            self.x = x

    class Child(Parent):
        def __init__(self, x: int, y: str):
            super().__init__(x)
            self.y = y

    # Child should inherit lazy from Parent
    assert hasattr(Child, "lazy")

    # When we track the child, it should still work correctly
    tracked_child = track(Child)
    assert hasattr(tracked_child, "lazy")

    # Create lazy instance of child using inherited lazy classmethod
    lazy_child = Child.lazy(x=10, y="hello")
    assert isinstance(lazy_child, Lazy)

    # Instantiate and verify it's the child class, not parent
    child_instance = lazy_child.instantiate()
    assert isinstance(child_instance, Child)
    assert child_instance.x == 10
    assert child_instance.y == "hello"

    # Verify the lazy instance knows it's a Child, not Parent
    config = lazy_child.get_config()
    assert config == {"x": 10, "y": "hello"}

    # Create lazy instance of parent for comparison
    lazy_parent = Parent.lazy(x=20)
    parent_instance = lazy_parent.instantiate()
    assert isinstance(parent_instance, Parent)
    assert not isinstance(parent_instance, Child)
    assert parent_instance.x == 20


def test_lazy_classmethod_inheritance_requires_args():
    """Test that .lazy() works correctly with child classes."""

    @track
    class Base:
        def __init__(self, value: int):
            self.value = value

    @track
    class Derived(Base):  # type: ignore[override]  # .lazy() signature differs from Base
        def __init__(self, value: int, name: str):
            super().__init__(value)
            self.name = name

    # .lazy() without args should raise ValidationError (no defaults)
    with pytest.raises(ValidationError):
        Derived.lazy()

    # .lazy() with args should work
    lazy_derived = Derived.lazy(value=42, name="test")

    # Should create a lazy instance of Derived, not Base
    assert isinstance(lazy_derived, Lazy)
    derived_instance = lazy_derived.instantiate()
    assert isinstance(derived_instance, Derived)
    assert derived_instance.value == 42
    assert derived_instance.name == "test"


# Tests for nested parameter updates and new Lazy features


def test_lazy_attribute_access():
    """Test that Lazy instances expose config as attributes."""
    lazy_inner = Inner.lazy(value=42)

    # Should be able to access config as attribute
    assert lazy_inner.value == 42


def test_lazy_nested_attribute_access():
    """Test nested attribute access through Lazy instances."""
    lazy_inner = Inner.lazy(value=42)
    lazy_middle = Middle.lazy(inner=lazy_inner)
    lazy_outer = Outer.lazy(middle=lazy_middle)

    # Should be able to chain through nested Lazy objects
    assert lazy_outer.middle.inner.value == 42


def test_lazy_attribute_mutation():
    """Test that Lazy config can be mutated via attribute assignment."""
    lazy_inner = Inner.lazy(value=42)

    # Mutate the config
    lazy_inner.value = 100

    # Should reflect the new value
    assert lazy_inner.value == 100

    # Instantiate and verify
    instance = lazy_inner.instantiate()
    assert instance.value == 100


def test_lazy_nested_attribute_mutation():
    """Test nested attribute mutation through Lazy instances."""
    lazy_inner = Inner.lazy(value=42)
    lazy_middle = Middle.lazy(inner=lazy_inner)

    # Mutate nested value
    lazy_middle.inner.value = 100

    # Should reflect the new value
    assert lazy_middle.inner.value == 100

    # Instantiate and verify
    middle_instance = lazy_middle.instantiate()
    inner_instance = middle_instance.inner.instantiate()
    assert inner_instance.value == 100


def test_lazy_attribute_mutation_validation():
    """Test that attribute mutation validates the new value."""
    lazy_inner = Inner.lazy(value=42)

    # Setting to wrong type should raise ValidationError
    with pytest.raises(ValidationError):
        lazy_inner.value = "not an int"

    # Original value should be unchanged (rollback)
    assert lazy_inner.value == 42


def test_lazy_attribute_mutation_invalid_param():
    """Test that setting non-existent parameter raises AttributeError."""
    lazy_inner = Inner.lazy(value=42)

    with pytest.raises(AttributeError) as exc_info:
        lazy_inner.nonexistent = 100

    assert "has no parameter 'nonexistent'" in str(exc_info.value)


def test_lazy_instantiate_is_factory():
    """Test that instantiate() creates new instances each time."""
    lazy_inner = Inner.lazy(value=42)

    # Each call creates a new instance
    instance1 = lazy_inner.instantiate()
    instance2 = lazy_inner.instantiate()

    # Different objects but same value
    assert instance1 is not instance2
    assert instance1.value == 42
    assert instance2.value == 42

    # Mutating config affects future instantiations but not existing ones
    lazy_inner.value = 100
    instance3 = lazy_inner.instantiate()

    assert instance1.value == 42  # Unchanged
    assert instance2.value == 42  # Unchanged
    assert instance3.value == 100  # Uses new value


def test_lazy_copy():
    """Test Lazy.copy() creates a new Lazy with updated config."""
    lazy_inner = Inner.lazy(value=42)

    # Create a copy with updated value
    lazy_copy = lazy_inner.copy(value=100)

    # Original should be unchanged
    assert lazy_inner.value == 42

    # Copy should have new value
    assert lazy_copy.value == 100

    # Both should be independent Lazy instances
    assert lazy_inner is not lazy_copy


def test_lazy_copy_chained():
    """Test chaining multiple copy() calls."""
    lazy_adder = Adder.lazy(amount=10.0)

    # Chain multiple copies
    result = lazy_adder.copy(amount=20.0).copy(amount=30.0)

    # Original unchanged
    assert lazy_adder.amount == 10.0

    # Final result has last value
    assert result.amount == 30.0


def test_lazy_copy_invalid_param():
    """Test that copy() with invalid parameter raises AttributeError."""
    lazy_inner = Inner.lazy(value=42)

    with pytest.raises(AttributeError) as exc_info:
        lazy_inner.copy(nonexistent=100)

    assert "has no parameter 'nonexistent'" in str(exc_info.value)


def test_lazy_copy_validates():
    """Test that copy() validates the new config."""
    lazy_inner = Inner.lazy(value=42)

    # Copy with wrong type should raise ValidationError
    with pytest.raises(ValidationError):
        lazy_inner.copy(value="not an int")


def test_is_all_lazy_simple():
    """Test is_all_lazy() with simple Lazy instances."""
    from confingy.utils.types import is_all_lazy

    # Single Lazy - should be True
    lazy_inner = Inner.lazy(value=42)
    assert is_all_lazy(lazy_inner) is True

    # Single instantiated - should be False
    inner = Inner(value=42)
    assert is_all_lazy(inner) is False


def test_is_all_lazy_nested():
    """Test is_all_lazy() with nested Lazy instances."""
    from confingy import disable_validation
    from confingy.utils.types import is_all_lazy

    # All Lazy - should be True
    lazy_inner = Inner.lazy(value=42)
    lazy_middle = Middle.lazy(inner=lazy_inner)
    lazy_outer = Outer.lazy(middle=lazy_middle)
    assert is_all_lazy(lazy_outer) is True

    # Inner is instantiated - should be False
    # Need to disable validation since Middle expects Lazy[Inner]
    inner = Inner(value=42)
    with disable_validation():
        lazy_middle_with_inst = Middle.lazy(inner=inner)
    assert is_all_lazy(lazy_middle_with_inst) is False


def test_is_all_lazy_with_dataclass():
    """Test is_all_lazy() with dataclass containing Lazy instances."""
    from dataclasses import dataclass

    from confingy.utils.types import is_all_lazy

    @dataclass
    class Config:
        inner: Lazy[Inner]
        count: int

    # All Lazy - should be True
    config = Config(inner=Inner.lazy(value=42), count=10)
    assert is_all_lazy(config) is True

    # Inner is instantiated - should be False
    config_inst = Config(inner=Inner(value=42), count=10)
    assert is_all_lazy(config_inst) is False


def test_is_all_lazy_with_list():
    """Test is_all_lazy() with lists containing Lazy instances."""
    from confingy.utils.types import is_all_lazy

    # List of Lazy - should be True
    lazy_list = [Inner.lazy(value=i) for i in range(3)]
    assert is_all_lazy(lazy_list) is True

    # List with one instantiated - should be False
    mixed_list = [Inner.lazy(value=0), Inner(value=1), Inner.lazy(value=2)]
    assert is_all_lazy(mixed_list) is False


def test_is_all_lazy_with_dict():
    """Test is_all_lazy() with dicts containing Lazy instances."""
    from confingy.utils.types import is_all_lazy

    # Dict of Lazy - should be True
    lazy_dict = {"a": Inner.lazy(value=1), "b": Inner.lazy(value=2)}
    assert is_all_lazy(lazy_dict) is True

    # Dict with one instantiated - should be False
    mixed_dict = {"a": Inner.lazy(value=1), "b": Inner(value=2)}
    assert is_all_lazy(mixed_dict) is False


def test_is_all_lazy_with_primitives():
    """Test is_all_lazy() with primitive values."""
    from confingy.utils.types import is_all_lazy

    # Primitives are fine (not tracked objects)
    assert is_all_lazy(42) is True
    assert is_all_lazy("hello") is True
    assert is_all_lazy([1, 2, 3]) is True
    assert is_all_lazy({"a": 1, "b": 2}) is True


def test_is_all_lazy_complex_nested():
    """Test is_all_lazy() with complex nested structure."""
    from dataclasses import dataclass

    from confingy.utils.types import is_all_lazy

    @dataclass
    class ComplexConfig:
        processors: list[Lazy[Adder]]
        settings: dict[str, Lazy[Inner]]

    # All Lazy - should be True
    config = ComplexConfig(
        processors=[Adder.lazy(amount=1.0), Adder.lazy(amount=2.0)],
        settings={"x": Inner.lazy(value=10), "y": Inner.lazy(value=20)},
    )
    assert is_all_lazy(config) is True

    # One instantiated in list - should be False
    config_mixed = ComplexConfig(
        processors=[Adder.lazy(amount=1.0), Adder(amount=2.0)],  # Adder instantiated
        settings={"x": Inner.lazy(value=10), "y": Inner.lazy(value=20)},
    )
    assert is_all_lazy(config_mixed) is False


# Tests for lens() and unlens()


def test_lens_tracked_instance():
    """Test lens() with a tracked instance."""
    from confingy import lens

    inner = Inner(value=42)
    inner_lens = lens(inner)

    # Should be a Lazy now
    assert hasattr(inner_lens, "_confingy_lazy_info")
    assert inner_lens.value == 42
    assert inner_lens._confingy_was_instantiated is True


def test_lens_lazy_instance():
    """Test lens() with a Lazy instance."""
    from confingy import lens

    lazy_inner = Inner.lazy(value=42)
    lazy_inner_lens = lens(lazy_inner)

    # Should still be Lazy
    assert hasattr(lazy_inner_lens, "_confingy_lazy_info")
    assert lazy_inner_lens.value == 42
    assert lazy_inner_lens._confingy_was_instantiated is False


def test_lens_nested_tracked():
    """Test lens() with nested tracked instances."""
    from confingy import lens

    # Define local classes that accept plain types (not Lazy)
    @track
    class InnerPlain:
        def __init__(self, value: int):
            self.value = value

    @track
    class MiddlePlain:
        def __init__(self, inner: InnerPlain):
            self.inner = inner

    @track
    class OuterPlain:
        def __init__(self, middle: MiddlePlain):
            self.middle = middle

    outer = OuterPlain(middle=MiddlePlain(inner=InnerPlain(value=42)))
    outer_lens = lens(outer)

    # Should be able to access nested values
    assert outer_lens.middle.inner.value == 42

    # All should be Lazy now
    assert hasattr(outer_lens, "_confingy_lazy_info")
    assert hasattr(outer_lens.middle, "_confingy_lazy_info")
    assert hasattr(outer_lens.middle.inner, "_confingy_lazy_info")


def test_lens_modify_and_unlens_tracked():
    """Test modifying via lens and unlensing back to tracked."""
    from confingy import lens

    # Define local classes that accept plain types (not Lazy)
    @track
    class InnerPlain:
        def __init__(self, value: int):
            self.value = value

    @track
    class MiddlePlain:
        def __init__(self, inner: InnerPlain):
            self.inner = inner

    @track
    class OuterPlain:
        def __init__(self, middle: MiddlePlain):
            self.middle = middle

    outer = OuterPlain(middle=MiddlePlain(inner=InnerPlain(value=42)))
    outer_lens = lens(outer)

    # Modify nested value
    outer_lens.middle.inner.value = 100

    # Unlens should give us back instances (not Lazy)
    new_outer = outer_lens.unlens()

    # Should be instances, not Lazy
    assert not hasattr(new_outer, "_confingy_lazy_info")
    assert not hasattr(new_outer.middle, "_confingy_lazy_info")
    assert not hasattr(new_outer.middle.inner, "_confingy_lazy_info")

    # Should have the updated value
    assert new_outer.middle.inner.value == 100

    # Original should be unchanged
    assert outer.middle.inner.value == 42


def test_lens_modify_and_unlens_lazy():
    """Test modifying via lens and unlensing back to Lazy."""
    from confingy import lens

    lazy_outer = Outer.lazy(middle=Middle.lazy(inner=Inner.lazy(value=42)))
    lazy_outer_lens = lens(lazy_outer)

    # Modify nested value
    lazy_outer_lens.middle.inner.value = 100

    # Unlens should give us back Lazy instances
    new_lazy = lazy_outer_lens.unlens()

    # Should still be Lazy
    assert hasattr(new_lazy, "_confingy_lazy_info")
    assert hasattr(new_lazy.middle, "_confingy_lazy_info")
    assert hasattr(new_lazy.middle.inner, "_confingy_lazy_info")

    # Should have the updated value
    assert new_lazy.middle.inner.value == 100


def test_lens_mixed_structure():
    """Test lens with mixed Lazy/tracked structure."""
    from confingy import lens

    # Outer is tracked, but middle is Lazy
    @track
    class OuterMixed:
        def __init__(self, middle: Lazy[Middle]):
            self.middle = middle

    outer = OuterMixed(middle=Middle.lazy(inner=Inner.lazy(value=42)))
    outer_lens = lens(outer)

    # Modify nested value
    outer_lens.middle.inner.value = 100

    # Unlens
    new_outer = outer_lens.unlens()

    # Outer should be instance (was tracked)
    assert not hasattr(new_outer, "_confingy_lazy_info")

    # Middle should be Lazy (was Lazy)
    assert hasattr(new_outer.middle, "_confingy_lazy_info")

    # Inner should be Lazy (was Lazy)
    assert hasattr(new_outer.middle.inner, "_confingy_lazy_info")

    # Should have the updated value
    assert new_outer.middle.inner.value == 100


def test_lens_with_list():
    """Test lens with list of tracked instances."""
    from confingy import lens

    @track
    class Pipeline:
        def __init__(self, steps: list[Inner]):
            self.steps = steps

    pipeline = Pipeline(steps=[Inner(value=1), Inner(value=2), Inner(value=3)])
    pipeline_lens = lens(pipeline)

    # Modify first step
    pipeline_lens.steps[0].value = 100

    # Unlens
    new_pipeline = pipeline_lens.unlens()

    # Should be instances
    assert not hasattr(new_pipeline, "_confingy_lazy_info")
    assert not hasattr(new_pipeline.steps[0], "_confingy_lazy_info")

    # Should have updated value
    assert new_pipeline.steps[0].value == 100
    assert new_pipeline.steps[1].value == 2
    assert new_pipeline.steps[2].value == 3


def test_dataclass_with_init_false_field():
    """Test that map_children handles dataclass with init=False fields."""
    from dataclasses import dataclass, field

    from confingy.utils.types import is_all_lazy

    @dataclass
    class ConfigWithComputed:
        inner: Lazy[Inner]
        count: int
        # This field is computed, not passed to __init__
        computed: str = field(init=False, default="computed_value")

        def __post_init__(self):
            self.computed = f"computed_{self.count}"

    # Create config with Lazy inner
    config = ConfigWithComputed(inner=Inner.lazy(value=42), count=10)

    # is_all_lazy uses map_children internally - should not crash
    assert is_all_lazy(config) is True

    # Verify the computed field exists
    assert config.computed == "computed_10"

    # Test with instantiated inner - should detect non-lazy
    config_inst = ConfigWithComputed(inner=Inner(value=42), count=10)
    assert is_all_lazy(config_inst) is False


def test_namedtuple_preserved_in_map_children():
    """Test that namedtuples are preserved (not converted to plain tuples)."""
    from collections import namedtuple

    from confingy import lens

    Point = namedtuple("Point", ["x", "y"])

    @track
    class ConfigWithNamedtuple:
        def __init__(self, point: Point, inner: Inner):
            self.point = point
            self.inner = inner

    # Create config with namedtuple and tracked instance
    config = ConfigWithNamedtuple(point=Point(x=10, y=20), inner=Inner(value=42))

    # Use lens which triggers map_children
    config_lens = lens(config)
    config_lens.inner.value = 100
    new_config = config_lens.unlens()

    # The point should still be a namedtuple, not a plain tuple
    assert type(new_config.point).__name__ == "Point"
    assert hasattr(new_config.point, "_fields")
    assert new_config.point.x == 10
    assert new_config.point.y == 20

    # Verify the inner value was updated
    assert new_config.inner.value == 100


def test_namedtuple_with_lazy_in_is_all_lazy():
    """Test is_all_lazy works with namedtuples containing Lazy instances."""
    from collections import namedtuple
    from dataclasses import dataclass

    from confingy.utils.types import is_all_lazy

    LazyPair = namedtuple("LazyPair", ["first", "second"])

    @dataclass
    class ConfigWithLazyNamedtuple:
        pair: LazyPair

    # All Lazy in namedtuple
    config = ConfigWithLazyNamedtuple(
        pair=LazyPair(first=Inner.lazy(value=1), second=Inner.lazy(value=2))
    )
    assert is_all_lazy(config) is True

    # One instantiated in namedtuple
    config_mixed = ConfigWithLazyNamedtuple(
        pair=LazyPair(first=Inner.lazy(value=1), second=Inner(value=2))
    )
    assert is_all_lazy(config_mixed) is False


def test_lens_with_dict():
    """Test lens with dict of tracked instances."""
    from confingy import lens

    @track
    class Config:
        def __init__(self, settings: dict[str, Inner]):
            self.settings = settings

    config = Config(settings={"a": Inner(value=1), "b": Inner(value=2)})
    config_lens = lens(config)

    # Modify a setting
    config_lens.settings["a"].value = 100

    # Unlens
    new_config = config_lens.unlens()

    # Should be instances
    assert not hasattr(new_config, "_confingy_lazy_info")
    assert not hasattr(new_config.settings["a"], "_confingy_lazy_info")

    # Should have updated value
    assert new_config.settings["a"].value == 100
    assert new_config.settings["b"].value == 2


def test_lens_error_on_invalid_type():
    """Test that lens raises error on invalid type."""
    from confingy import lens

    with pytest.raises(
        TypeError, match="lens\\(\\) requires a Lazy or tracked instance"
    ):
        lens(42)

    with pytest.raises(
        TypeError, match="lens\\(\\) requires a Lazy or tracked instance"
    ):
        lens("not a tracked instance")


def test_unlens_preserves_was_instantiated_through_copy():
    """Test that copy() preserves _was_instantiated."""
    from confingy import lens

    # Define local classes that accept plain types (not Lazy)
    @track
    class InnerPlain:
        def __init__(self, value: int):
            self.value = value

    @track
    class MiddlePlain:
        def __init__(self, inner: InnerPlain):
            self.inner = inner

    @track
    class OuterPlain:
        def __init__(self, middle: MiddlePlain):
            self.middle = middle

    outer = OuterPlain(middle=MiddlePlain(inner=InnerPlain(value=42)))
    outer_lens = lens(outer)

    # Copy should preserve _was_instantiated
    l_copy = outer_lens.copy(
        middle=outer_lens.middle.copy(inner=outer_lens.middle.inner.copy(value=100))
    )

    # Unlens should still instantiate
    new_outer = l_copy.unlens()
    assert not hasattr(new_outer, "_confingy_lazy_info")
    assert new_outer.middle.inner.value == 100


def test_unlens_preserves_was_instantiated_through_call():
    """Test that __call__ preserves _was_instantiated flag.

    When a Lazy is created via lens() from a tracked instance, calling __call__
    to modify it (e.g., lazy_obj(arg=value)) should preserve _was_instantiated=True.
    This ensures unlens() returns an instantiated object, not a Lazy.
    """
    from confingy import lens

    @track
    class SimpleClass:
        def __init__(self, value: int, name: str = "default"):
            self.value = value
            self.name = name

    # Create a tracked instance and lens it
    obj = SimpleClass(value=42, name="original")
    lensed = lens(obj)

    # Verify lens() sets _was_instantiated=True
    assert lensed._confingy_was_instantiated is True

    # Use __call__ to create a new Lazy with updated config
    modified = lensed(value=100)

    # __call__ should preserve _was_instantiated
    assert modified._confingy_was_instantiated is True

    # unlens() should return an instantiated object, not a Lazy
    result = modified.unlens()
    assert not isinstance(result, Lazy)
    assert hasattr(result, "_tracked_info")
    assert result.value == 100
    assert result.name == "original"


def test_call_on_regular_lazy_stays_lazy():
    """Test that __call__ on a regular Lazy (not from lens) keeps _was_instantiated=False."""

    @track
    class SimpleClass:
        def __init__(self, value: int, name: str = "default"):
            self.value = value
            self.name = name

    # Create a regular Lazy (not via lens)
    lazy_obj = SimpleClass.lazy(value=42)

    # Regular Lazy has _was_instantiated=False
    assert lazy_obj._confingy_was_instantiated is False

    # Use __call__ to modify
    modified = lazy_obj(name="updated")

    # Should still have _was_instantiated=False
    assert modified._confingy_was_instantiated is False

    # unlens() should return a Lazy, not an instantiated object
    result = modified.unlens()
    assert isinstance(result, Lazy)


# Tests for __post_config__ hook


def test_post_config_hook_called_on_init():
    """Test that __post_config__ is called on initial Lazy creation."""
    hook_calls = []

    @track
    class WithHook:
        def __init__(self, value: int, derived: int = 0):
            self.value = value
            self.derived = derived

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("init", instance.value))
            return instance

    # Create a Lazy - hook should be called
    _ = WithHook.lazy(value=42)

    assert len(hook_calls) == 1
    assert hook_calls[0] == ("init", 42)


def test_post_config_hook_called_on_setattr():
    """Test that __post_config__ is called when config is updated via setattr."""
    hook_calls = []

    @track
    class WithHook:
        def __init__(self, value: int, derived: int = 0):
            self.value = value
            self.derived = derived

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("update", instance.value))
            return instance

    lazy_instance = WithHook.lazy(value=42)
    hook_calls.clear()  # Clear init call

    # Update the value - hook should be called
    lazy_instance.value = 100

    assert len(hook_calls) == 1
    assert hook_calls[0] == ("update", 100)


def test_post_config_hook_modifies_config():
    """Test that __post_config__ can modify the config."""

    @track
    class AutoDerived:
        def __init__(self, value: int, derived: int = 0):
            self.value = value
            self.derived = derived

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Automatically set derived = value * 2
            instance.derived = instance.value * 2
            return instance

    # Create a Lazy - derived should be auto-set
    lazy_instance = AutoDerived.lazy(value=10)
    assert lazy_instance.value == 10
    assert lazy_instance.derived == 20

    # Update value - derived should update
    lazy_instance.value = 50
    assert lazy_instance.derived == 100


def test_post_config_hook_returns_new_lazy():
    """Test that __post_config__ can return a different Lazy instance."""

    @track
    class ReturnsNew:
        def __init__(self, value: int, computed: int = 0):
            self.value = value
            self.computed = computed

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Return a copy with computed value
            return instance.copy(computed=instance.value + 100)

    lazy_instance = ReturnsNew.lazy(value=10)
    assert lazy_instance.value == 10
    assert lazy_instance.computed == 110


def test_post_config_hook_prevents_recursion():
    """Test that __post_config__ doesn't cause infinite recursion."""
    hook_calls = []

    @track
    class RecursiveHook:
        def __init__(self, value: int, other: int = 0):
            self.value = value
            self.other = other

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(instance.value)
            # This would cause infinite recursion without the guard
            instance.other = instance.value + 1
            return instance

    lazy_instance = RecursiveHook.lazy(value=42)

    # Should only be called once (on init), not infinitely
    assert len(hook_calls) == 1
    assert hook_calls[0] == 42
    assert lazy_instance.other == 43


def test_post_config_hook_rollback_on_failure():
    """Test that config is rolled back if __post_config__ raises an exception."""

    @track
    class FailingHook:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            if instance.value > 100:
                raise ValueError("Value too large")
            return instance

    # Initial creation with valid value should work
    lazy_instance = FailingHook.lazy(value=50)
    assert lazy_instance.value == 50

    # Update to invalid value should fail and rollback
    with pytest.raises(ValueError, match="Value too large"):
        lazy_instance.value = 200

    # Value should be unchanged (rolled back)
    assert lazy_instance.value == 50


def test_post_config_hook_propagates_to_nested():
    """Test using __post_config__ to propagate values to nested configs."""
    from confingy import lens

    @track
    class Dataset:
        def __init__(self, batch_size: int):
            self.batch_size = batch_size

    @track
    class Config:
        def __init__(self, batch_size: int, dataset: Lazy[Dataset]):
            self.batch_size = batch_size
            self.dataset = dataset

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Propagate batch_size to nested dataset
            lensed = lens(instance)
            lensed.dataset.batch_size = instance.batch_size
            return lensed.unlens()

    # Create config with mismatched batch_sizes
    config = Config.lazy(batch_size=32, dataset=Dataset.lazy(batch_size=16))

    # After __post_config__, dataset should have the same batch_size
    assert config.batch_size == 32
    assert config.dataset.batch_size == 32

    # Update batch_size - should propagate
    config.batch_size = 64
    assert config.dataset.batch_size == 64


def test_post_config_hook_not_called_without_hook():
    """Test that classes without __post_config__ work normally."""

    @track
    class NoHook:
        def __init__(self, value: int):
            self.value = value

    # Should work fine without a hook
    lazy_instance = NoHook.lazy(value=42)
    assert lazy_instance.value == 42

    lazy_instance.value = 100
    assert lazy_instance.value == 100


def test_post_config_hook_with_copy():
    """Test that __post_config__ is called when using copy()."""
    hook_calls = []

    @track
    class WithHookCopy:
        def __init__(self, value: int, derived: int = 0):
            self.value = value
            self.derived = derived

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("hook", instance.value))
            instance.derived = instance.value * 2
            return instance

    lazy_instance = WithHookCopy.lazy(value=10)
    assert lazy_instance.derived == 20
    hook_calls.clear()

    # copy() creates a new Lazy, which should trigger the hook
    lazy_copy = lazy_instance.copy(value=50)

    assert len(hook_calls) == 1
    assert hook_calls[0] == ("hook", 50)
    assert lazy_copy.derived == 100


def test_post_config_hook_validates_after_hook():
    """Test that validation runs after __post_config__ to catch invalid modifications."""

    @track
    class BadHook:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Directly manipulate _config to bypass __setattr__ validation
            instance._confingy_config["value"] = "not an int"
            return instance

    # Should raise ValidationError because re-validation catches the bad value
    with pytest.raises(ValidationError):
        BadHook.lazy(value=42)


def test_post_config_hook_validates_returned_lazy():
    """Test that validation catches invalid values in returned Lazy."""
    from confingy import disable_validation

    @track
    class ReturnsInvalid:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Create an invalid Lazy by disabling validation, then return it
            with disable_validation():
                return Lazy(cls, {"value": "not an int"}, _skip_post_config_hook=True)

    # Should raise ValidationError because re-validation catches the bad value
    with pytest.raises(ValidationError):
        ReturnsInvalid.lazy(value=42)


def test_post_config_hook_rollback_with_none_value():
    """Test that rollback works when the previous value was None."""

    @track
    class WithOptional:
        def __init__(self, value: int, optional: str | None = None):
            self.value = value
            self.optional = optional

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            if instance.optional == "fail":
                raise ValueError("Intentional failure")
            return instance

    # Create with optional=None
    lazy_instance = WithOptional.lazy(value=42, optional=None)
    assert lazy_instance.optional is None

    # Update optional to a valid value
    lazy_instance.optional = "valid"
    assert lazy_instance.optional == "valid"

    # Update to None, then to "fail" - should rollback to None
    lazy_instance.optional = None
    assert lazy_instance.optional is None

    with pytest.raises(ValueError, match="Intentional failure"):
        lazy_instance.optional = "fail"

    # Should be rolled back to None
    assert lazy_instance.optional is None


def test_post_config_hook_rollback_entire_config_when_hook_returns_new_lazy():
    """Test that entire config is rolled back when hook returns a modified Lazy and validation fails."""

    @track
    class ReturnsModified:
        def __init__(self, value: int, other: int = 0, trigger: str = "ok"):
            self.value = value
            self.other = other
            self.trigger = trigger

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Return a copy that modifies multiple fields
            new_instance = instance.copy(other=instance.value * 10)
            # Then cause validation to fail by setting invalid type
            if instance.trigger == "fail":
                new_instance._confingy_config["value"] = "not an int"
            return new_instance

    # Create initial instance
    lazy_instance = ReturnsModified.lazy(value=5, other=0, trigger="ok")
    assert lazy_instance.value == 5
    assert lazy_instance.other == 50  # Modified by hook

    # Update trigger to cause failure - entire config should rollback
    with pytest.raises(ValidationError):
        lazy_instance.trigger = "fail"

    # Entire config should be rolled back to before the update
    assert lazy_instance.value == 5
    assert lazy_instance.other == 50
    assert lazy_instance.trigger == "ok"


def test_unlens_does_not_run_hook():
    """Test that unlens() does not run __post_config__ hook (hooks run on setattr)."""
    from confingy import lens

    hook_calls = []

    @track
    class WithDerived:
        def __init__(self, value: int, derived: int = 0):
            self.value = value
            self.derived = derived

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(instance.value)
            instance.derived = instance.value * 2
            return instance

    # Create a Lazy - hook runs
    lazy_instance = WithDerived.lazy(value=10)
    assert lazy_instance.derived == 20
    assert hook_calls == [10]
    hook_calls.clear()

    # Use lens/unlens outside of hook context
    # For flat objects, lens() returns the same object
    lazy_instance_lens = lens(lazy_instance)
    assert lazy_instance_lens is lazy_instance

    # setattr triggers hook once
    lazy_instance_lens.value = 50
    assert len(hook_calls) == 1

    # unlens() should NOT trigger hook (it's a structural transformation)
    # The hook already ran on setattr and set derived=100
    result = lazy_instance_lens.unlens()
    assert len(hook_calls) == 1  # Hook only ran on setattr, not unlens
    assert result.derived == 100  # Value set by hook during setattr


def test_lens_does_not_trigger_post_config_hook():
    """Test that lens() does not trigger __post_config__ hooks."""
    from confingy import lens

    hook_calls = []

    @track
    class Inner:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("inner", instance.value))
            return instance

    @track
    class Outer:
        def __init__(self, inner: Lazy[Inner]):
            self.inner = inner

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("outer", instance.inner.value))
            return instance

    # Create lazy instances - hooks run on creation
    lazy_inner = Inner.lazy(value=42)
    lazy_outer = Outer.lazy(inner=lazy_inner)
    assert len(hook_calls) == 2  # Both hooks ran on creation
    hook_calls.clear()

    # lens() should NOT trigger hooks
    _ = lens(lazy_outer)
    assert len(hook_calls) == 0  # No hooks should have run

    # Also test with tracked instances
    @track
    class TrackedInner:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            hook_calls.append(("tracked_inner", instance.value))
            return instance

    tracked = TrackedInner(value=100)
    hook_calls.clear()

    # lens() on tracked instance should NOT trigger hooks
    _ = lens(tracked)
    assert len(hook_calls) == 0  # No hooks should have run


def test_post_config_changed_key_prevents_unwanted_propagation():
    """Test that using changed_key prevents hooks from overwriting manually-set nested values.

    This tests the scenario where:
    1. Parent propagates hidden_size to children on creation and when hidden_size changes
    2. User manually sets a child's hidden_size to a different value
    3. User changes an unrelated parent param (name)
    4. The child's manually-set hidden_size should NOT be overwritten
    """

    @track
    class ChildModule:
        def __init__(self, hidden_size: int, name: str | None = None):
            self.hidden_size = hidden_size
            self.name = name

    @track
    class ParentModel:
        def __init__(
            self,
            hidden_size: int,
            name: str,
            encoder: Lazy[ChildModule],
        ):
            self.hidden_size = hidden_size
            self.name = name
            self.encoder = encoder

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            # Only propagate hidden_size when it changes (or on init)
            if changed_key is None or changed_key == "hidden_size":
                instance.encoder.hidden_size = instance.hidden_size
            # Only propagate name when it changes (or on init)
            if changed_key is None or changed_key == "name":
                instance.encoder.name = f"{instance.name}_encoder"
            return instance

    # Create parent - hook runs with changed_key=None, propagates both values
    lazy_parent = ParentModel.lazy(
        hidden_size=512,
        name="my_model",
        encoder=ChildModule.lazy(hidden_size=0),
    )
    assert lazy_parent.hidden_size == 512
    assert lazy_parent.encoder.hidden_size == 512
    assert lazy_parent.encoder.name == "my_model_encoder"

    # Change hidden_size on parent - should propagate to child
    lazy_parent.hidden_size = 1024
    assert lazy_parent.encoder.hidden_size == 1024

    # Manually change child's hidden_size
    lazy_parent.encoder.hidden_size = 256
    assert lazy_parent.encoder.hidden_size == 256

    # Change an UNRELATED parent param (name)
    # This should NOT overwrite the child's manually-set hidden_size
    lazy_parent.name = "new_model"
    assert lazy_parent.encoder.name == "new_model_encoder"  # name updated
    assert lazy_parent.encoder.hidden_size == 256  # hidden_size preserved!

    # Verify parent's hidden_size is still 1024
    assert lazy_parent.hidden_size == 1024


def test_post_config_changed_key_is_none_on_init():
    """Test that changed_key is None when hook is called during initial creation."""
    received_changed_keys = []

    @track
    class Model:
        def __init__(self, value: int):
            self.value = value

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            received_changed_keys.append(changed_key)
            return instance

    # On creation, changed_key should be None
    _ = Model.lazy(value=42)
    assert received_changed_keys == [None]

    received_changed_keys.clear()

    # On setattr, changed_key should be the param name
    lazy_model = Model.lazy(value=10)
    received_changed_keys.clear()

    lazy_model.value = 20
    assert received_changed_keys == ["value"]


def test_post_config_changed_key_with_multiple_updates():
    """Test that changed_key correctly identifies which param changed across multiple updates."""
    change_log = []

    @track
    class Config:
        def __init__(self, a: int, b: str, c: float):
            self.a = a
            self.b = b
            self.c = c

        @classmethod
        def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
            change_log.append(changed_key)
            return instance

    lazy_config = Config.lazy(a=1, b="hello", c=3.14)
    assert change_log == [None]  # Initial creation
    change_log.clear()

    lazy_config.a = 2
    assert change_log == ["a"]

    lazy_config.b = "world"
    assert change_log == ["a", "b"]

    lazy_config.c = 2.71
    assert change_log == ["a", "b", "c"]


class TestDefaultConstructibleBehavior:
    """Tests for default constructible behavior - lazy() returns Lazy directly when all params have defaults."""

    def test_lazy_with_all_defaults_returns_lazy_directly(self):
        """When all params have defaults, .lazy() should return a Lazy, not a factory."""

        @track
        class AllDefaults:
            def __init__(self, a: int = 1, b: str = "hello"):
                self.a = a
                self.b = b

        result = AllDefaults.lazy()
        assert isinstance(result, Lazy)
        assert result.a == 1
        assert result.b == "hello"

    def test_lazy_without_defaults_raises_validation_error(self):
        """When some params lack defaults, .lazy() should raise ValidationError."""
        from confingy import ValidationError

        @track
        class RequiredParams:
            def __init__(self, required: int, optional: str = "default"):
                self.required = required
                self.optional = optional

        # Without required params, should raise ValidationError
        with pytest.raises(ValidationError):
            RequiredParams.lazy()

        # With required params, should work
        result = RequiredParams.lazy(required=42)
        assert isinstance(result, Lazy)
        assert result.required == 42
        assert result.optional == "default"

    def test_lazy_call_returns_self(self):
        """Calling a Lazy with no args should return self (supports lazy(Class)() pattern)."""

        @track
        class AllDefaults:
            def __init__(self, value: int = 10):
                self.value = value

        lazy1 = AllDefaults.lazy()
        lazy2 = lazy1()  # Call with no args

        assert lazy1 is lazy2

    def test_lazy_function_with_all_defaults(self):
        """The lazy() function returns a factory; call it to get a Lazy."""

        @track
        class AllDefaults:
            def __init__(self, x: int = 5):
                self.x = x

        # lazy(Class) returns a factory, call it to get Lazy
        factory = lazy(AllDefaults)
        result = factory()
        assert isinstance(result, Lazy)
        assert result.x == 5


class TestFieldDefaultFactory:
    """Tests for dataclass Field.default_factory evaluation."""

    def test_field_default_factory_is_called(self):
        """Field.default_factory should be called when building the config."""
        from dataclasses import field

        call_count = [0]

        def make_list():
            call_count[0] += 1
            return [1, 2, 3]

        @track
        class WithFactory:
            def __init__(self, items: list = field(default_factory=make_list)):
                self.items = items

        # Each lazy() call should invoke the factory
        lazy1 = WithFactory.lazy()
        assert call_count[0] == 1
        assert lazy1.items == [1, 2, 3]

        lazy2 = WithFactory.lazy()
        assert call_count[0] == 2

        # Verify they're different list instances (not shared)
        lazy1.items.append(4)
        assert lazy1.items == [1, 2, 3, 4]
        assert lazy2.items == [1, 2, 3]  # Unchanged

    def test_field_default_value_is_used(self):
        """Field.default should be used when no factory is provided."""
        from dataclasses import field

        @track
        class WithFieldDefault:
            def __init__(self, value: int = field(default=42)):
                self.value = value

        lazy_instance = WithFieldDefault.lazy()
        assert lazy_instance.value == 42


class TestPickleSupport:
    """Tests for pickle/unpickle support on Lazy instances.

    Note: Uses module-level classes from conftest.py since local classes can't be pickled.
    """

    def test_lazy_can_be_pickled_and_unpickled(self):
        """Lazy instances should survive pickle round-trip."""
        import pickle

        from tests.conftest import Adder

        lazy_original = Adder.lazy(amount=42.0)

        # Pickle and unpickle
        pickled = pickle.dumps(lazy_original)
        lazy_restored = pickle.loads(pickled)

        assert isinstance(lazy_restored, Lazy)
        assert lazy_restored.amount == 42.0

    def test_pickled_lazy_can_be_instantiated(self):
        """Unpickled Lazy should still be able to instantiate the object."""
        import pickle

        from tests.conftest import Inner

        lazy_original = Inner.lazy(value=100)
        pickled = pickle.dumps(lazy_original)
        lazy_restored = pickle.loads(pickled)

        instance = lazy_restored.instantiate()
        assert instance.value == 100

    def test_nested_lazy_can_be_pickled(self):
        """Nested Lazy instances should survive pickle round-trip."""
        import pickle

        from tests.conftest import Inner, Middle, Outer

        lazy_outer = Outer.lazy(middle=Middle.lazy(inner=Inner.lazy(value=10)))

        pickled = pickle.dumps(lazy_outer)
        restored = pickle.loads(pickled)

        assert restored.middle.inner.value == 10

    def test_validation_model_rebuilt_after_unpickle(self):
        """The validation model should be rebuilt after unpickling."""
        import pickle

        from tests.conftest import Inner

        lazy_original = Inner.lazy(value=1)
        assert lazy_original._confingy_validation_model is not None  # Has validation
        old_model = lazy_original._confingy_validation_model

        pickled = pickle.dumps(lazy_original)
        lazy_restored = pickle.loads(pickled)

        # Validation model should be rebuilt after unpickling
        assert lazy_restored._confingy_validation_model is not None
        new_model = lazy_restored._confingy_validation_model

        # Not the same object (rebuilt), but equivalent structure
        assert new_model is not old_model
        assert set(old_model.model_fields.keys()) == set(new_model.model_fields.keys())

        # And it should still work for validation
        lazy_restored.value = 42  # Should not raise

    def test_pickle_preserves_skip_validation(self):
        """Lazy created with skip_validation=True should remain unvalidated after unpickling."""
        import pickle

        from confingy import lens
        from tests.conftest import Inner

        # Create a tracked instance and lens it (which creates Lazy with skip_validation=True)
        inner = Inner(value=42)
        lensed = lens(inner)

        # Verify validation is disabled (lens() sets skip_validation=True)
        assert lensed._confingy_validation_model is None

        # Pickle and unpickle
        pickled = pickle.dumps(lensed)
        restored = pickle.loads(pickled)

        # Validation should still be disabled after unpickling
        assert restored._confingy_validation_model is None

        # Should still work - setting value without validation
        restored.value = 100
        assert restored.value == 100

    def test_tracked_instance_can_be_pickled(self):
        """Tracked instances (not just Lazy) should survive pickle round-trip."""
        import pickle

        from tests.conftest import Inner

        instance = Inner(value=42)
        assert hasattr(instance, "_tracked_info")

        # Pickle and unpickle
        pickled = pickle.dumps(instance)
        restored = pickle.loads(pickled)

        assert restored.value == 42
        assert hasattr(restored, "_tracked_info")
        assert restored._tracked_info["init_args"]["value"] == 42

    def test_tracked_instance_inline_track_can_be_pickled(self):
        """track(SomeClass)(args) instances should survive pickle round-trip.

        This is the key regression: track() creates a dynamic subclass that
        pickle can't find by module path, so we need __reduce__ support.
        """
        import pickle

        from tests.conftest import Inner

        # Inner is already @track-decorated, get the original untracked class
        OrigInner = Inner.__bases__[0]

        # Use track() inline (creates a new dynamic subclass)
        instance = track(OrigInner)(value=99)
        assert hasattr(instance, "_tracked_info")

        # Pickle and unpickle
        pickled = pickle.dumps(instance)
        restored = pickle.loads(pickled)

        assert restored.value == 99
        assert hasattr(restored, "_tracked_info")
        assert restored._tracked_info["init_args"]["value"] == 99

    def test_tracked_instance_with_custom_reduce_can_be_pickled(self):
        """track() on a class with its own __reduce__ should still be pickleable."""
        import pickle

        from tests.conftest import UntrackedWithReduce

        instance = track(UntrackedWithReduce)(value=7)
        assert hasattr(instance, "_tracked_info")
        instance.extra = "modified_after_init"

        # Pickle and unpickle
        pickled = pickle.dumps(instance)
        restored = pickle.loads(pickled)

        assert restored.value == 7
        assert hasattr(restored, "_tracked_info")
        # Extra state set after init should be preserved
        assert restored.extra == "modified_after_init"

    def test_pickle_preserves_was_instantiated_flag(self):
        """Lazy created via lens() should preserve _was_instantiated after unpickling."""
        import pickle

        from confingy import lens
        from tests.conftest import Inner

        # Create a tracked instance and lens it
        inner = Inner(value=42)
        lensed = lens(inner)

        # Verify _was_instantiated is True
        assert lensed._confingy_was_instantiated is True

        # Pickle and unpickle
        pickled = pickle.dumps(lensed)
        restored = pickle.loads(pickled)

        # _was_instantiated should be preserved
        assert restored._confingy_was_instantiated is True

        # unlens() should return an instantiated object, not a Lazy
        result = restored.unlens()
        assert not isinstance(result, Lazy)
        assert result.value == 42
