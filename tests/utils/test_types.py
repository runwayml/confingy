from typing import Optional, Union

from confingy import Lazy, lazy, track
from confingy.utils.types import is_lazy_instance, is_lazy_type, is_nonlazy_subclass_of
from tests.conftest import MyModel


def test_is_lazy_type():
    """Test is_lazy_type function with various type annotations."""
    # Test with Lazy type annotation
    assert is_lazy_type(Lazy[MyModel])
    assert is_lazy_type(Lazy[int])
    assert is_lazy_type(Lazy[str])

    # Test with regular types
    assert not is_lazy_type(int)
    assert not is_lazy_type(str)
    assert not is_lazy_type(MyModel)
    assert not is_lazy_type(list[int])
    assert not is_lazy_type(dict[str, int])
    assert not is_lazy_type(Optional[str])
    assert not is_lazy_type(Union[int, str])

    # Test with instances
    assert not is_lazy_type(42)
    assert not is_lazy_type("hello")
    assert not is_lazy_type([1, 2, 3])

    # Test edge cases
    assert not is_lazy_type(None)
    assert not is_lazy_type(object())


def test_is_lazy_instance():
    """Test is_lazy_instance function with various objects."""
    from confingy import lazy

    # Create lazy and regular instances
    lazy_model = lazy(MyModel, in_features=5, out_features=10)
    regular_instance = object()

    @track
    class TrackedClass:
        def __init__(self, value: int):
            self.value = value

    tracked_instance = TrackedClass(value=42)

    # Test lazy instances
    assert is_lazy_instance(lazy_model)

    # Test regular instances
    assert not is_lazy_instance(regular_instance)
    assert not is_lazy_instance(tracked_instance)
    assert not is_lazy_instance(42)
    assert not is_lazy_instance("hello")
    assert not is_lazy_instance([1, 2, 3])
    assert not is_lazy_instance(None)

    # Lazy stays a Lazy even after calling instantiate()
    lazy_model.instantiate()
    assert is_lazy_instance(lazy_model)


def test_is_nonlazy_subclass_of_with_classes():
    """Test is_nonlazy_subclass_of with class objects."""

    class BaseClass:
        pass

    class DerivedClass(BaseClass):
        pass

    class UnrelatedClass:
        pass

    @lazy
    class LazyDerived(BaseClass):
        def __init__(self):
            pass

    # Test with regular classes
    assert is_nonlazy_subclass_of(DerivedClass, BaseClass)
    assert is_nonlazy_subclass_of(BaseClass, BaseClass)
    assert not is_nonlazy_subclass_of(UnrelatedClass, BaseClass)

    # Test with lazy classes - should return False
    assert not is_nonlazy_subclass_of(LazyDerived, BaseClass)

    # Test error cases
    assert not is_nonlazy_subclass_of(42, BaseClass)
    assert not is_nonlazy_subclass_of("string", BaseClass)


def test_is_nonlazy_subclass_of_with_instances():
    """Test is_nonlazy_subclass_of with instance objects."""

    class BaseClass:
        pass

    class DerivedClass(BaseClass):
        pass

    @lazy
    class LazyDerived(BaseClass):
        def __init__(self):
            pass

    # Test with regular instances
    base_instance = BaseClass()
    derived_instance = DerivedClass()
    lazy_instance = LazyDerived()
    unrelated_instance = "string"

    assert is_nonlazy_subclass_of(base_instance, BaseClass)
    assert is_nonlazy_subclass_of(derived_instance, BaseClass)
    assert not is_nonlazy_subclass_of(lazy_instance, BaseClass)
    assert not is_nonlazy_subclass_of(unrelated_instance, BaseClass)


def test_is_nonlazy_subclass_of_with_lazy_types():
    """Test is_nonlazy_subclass_of with lazy type annotations."""

    class BaseClass:
        pass

    # Test with lazy type annotations - should return False because it's a lazy type
    lazy_type = Lazy[BaseClass]
    assert not is_nonlazy_subclass_of(lazy_type, BaseClass)


def test_is_nonlazy_subclass_of_with_original_cls():
    """Test is_nonlazy_subclass_of with classes that have _original_cls attribute."""

    class BaseClass:
        pass

    class MockExpectedClass:
        _original_cls = BaseClass

    # Test with a regular instance and expected_supertype that has _original_cls
    regular_instance = BaseClass()

    # Should use _original_cls for comparison
    assert is_nonlazy_subclass_of(regular_instance, MockExpectedClass)


def test_is_nonlazy_subclass_of_error_handling():
    """Test is_nonlazy_subclass_of error handling for edge cases."""

    # Test with non-callable expected_supertype
    non_callable = 42
    assert not is_nonlazy_subclass_of("test", non_callable)

    # Test TypeError handling - create a class that raises TypeError in isinstance
    class ProblematicType(type):
        def __instancecheck__(cls, instance):
            raise TypeError("Intentional error")

    class ProblematicClass(metaclass=ProblematicType):
        pass

    # Should handle TypeError gracefully
    regular_instance = object()
    assert not is_nonlazy_subclass_of(regular_instance, ProblematicClass)

    # Test TypeError handling in issubclass with class objects
    class AnotherProblematicType(type):
        def __subclasscheck__(cls, subclass):
            raise TypeError("Intentional error")

    class AnotherProblematicClass(metaclass=AnotherProblematicType):
        pass

    assert not is_nonlazy_subclass_of(ProblematicClass, AnotherProblematicClass)


def test_lazy_type_annotation_behavior():
    """Test the Lazy type annotation class behavior."""

    # Test __class_getitem__ - with new implementation, these should be Lazy types
    lazy_int = Lazy[int]
    lazy_str = Lazy[str]
    lazy_model = Lazy[MyModel]

    # Should be actual Lazy types now, not the wrapped type
    assert is_lazy_type(lazy_int)
    assert is_lazy_type(lazy_str)
    assert is_lazy_type(lazy_model)

    # Test with complex types
    lazy_list = Lazy[list[int]]
    lazy_dict = Lazy[dict[str, int]]

    assert is_lazy_type(lazy_list)
    assert is_lazy_type(lazy_dict)

    # Test that they have the expected attributes from types.GenericAlias
    from typing import get_args, get_origin

    assert get_origin(lazy_int) is Lazy
    assert get_args(lazy_int) == (int,)
    assert get_origin(lazy_model) is Lazy
    assert get_args(lazy_model) == (MyModel,)


def test_edge_cases_and_none_values():
    """Test edge cases with None and invalid values."""

    # Test is_lazy_type with None and invalid objects
    assert not is_lazy_type(None)

    # Test object without __origin__
    class NoOrigin:
        pass

    assert not is_lazy_type(NoOrigin())
    assert not is_lazy_type(NoOrigin)

    # Test is_lazy_instance with None
    assert not is_lazy_instance(None)

    # Test is_nonlazy_subclass_of with None
    # None is not a subclass of object, but isinstance(None, object) is True in Python
    # However, our function should handle this gracefully
    assert is_nonlazy_subclass_of(None, object)  # isinstance(None, object) is True

    # Test with None as expected_supertype - should return False
    assert not is_nonlazy_subclass_of(object(), None)


def test_lazy_type_with_generics():
    """Test Lazy type annotation with generic types."""
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class GenericClass(Generic[T]):
        def __init__(self, value: T):
            self.value = value

    # Test Lazy with generic types - should be Lazy types now
    lazy_generic = Lazy[GenericClass[int]]
    assert is_lazy_type(lazy_generic)

    from typing import get_args, get_origin

    assert get_origin(lazy_generic) is Lazy
    assert get_args(lazy_generic) == (GenericClass[int],)

    # Test is_lazy_type with generic Lazy
    assert is_lazy_type(Lazy[GenericClass[str]])
    assert not is_lazy_type(GenericClass[str])
