from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, TypeVar, get_origin

if TYPE_CHECKING:
    from confingy.tracking import Lazy

T = TypeVar("T")


class TrackedInstance(Protocol):
    """Protocol for objects decorated with @track."""

    _tracked_info: dict[str, Any]


def is_lazy_type(obj: Any) -> bool:
    """Check if an object is a [Lazy][confingy.tracking.Lazy] type annotation."""
    # Import at runtime to avoid circular dependency
    from confingy.tracking import Lazy

    # Check if it's a GenericAlias with Lazy as the origin
    origin = get_origin(obj)
    return origin is Lazy


def is_lazy_instance(obj: Any) -> TypeGuard["Lazy[Any]"]:
    """
    Check if an object is a [Lazy][confingy.tracking.Lazy] instance.
    """
    return hasattr(obj, "_confingy_lazy_info") and hasattr(obj, "instantiate")


def is_tracked_instance(obj: Any) -> TypeGuard[TrackedInstance]:
    """
    Check if an object is a tracked instance (decorated with @track).

    Tracked instances have a `_tracked_info` attribute that stores
    the class name, module, and constructor arguments used to create them.
    """
    return hasattr(obj, "_tracked_info")


def is_lazy_version_of(obj: Any, expected_type: type) -> bool:
    """Check if an object is a [Lazy][confingy.tracking.Lazy] version of a given type."""
    return is_lazy_instance(obj) and obj._confingy_cls == expected_type


def is_nonlazy_subclass_of(obj: Any, expected_supertype: type) -> bool:
    """
    Check if an object is a non-lazy subclass of a given type.

    Args:
        obj: The object to check
        expected_supertype: The expected supertype

    Returns:
        True if obj is a subclass of expected_supertype and not lazy
    """
    # Check if it's a lazy type or instance
    if is_lazy_type(obj) or is_lazy_instance(obj):
        return False

    # Get the actual type to check against
    actual_expected_type = expected_supertype
    if hasattr(expected_supertype, "_original_cls"):
        actual_expected_type = expected_supertype._original_cls
    elif callable(expected_supertype) and not isinstance(expected_supertype, type):
        return False

    # Perform the type check
    if isinstance(obj, type):
        try:
            return issubclass(obj, actual_expected_type)
        except TypeError:
            return False
    else:
        try:
            return isinstance(obj, actual_expected_type)
        except TypeError:
            return False


def is_all_lazy(obj: Any) -> bool:
    """
    Check if an entire nested fingy is fully lazy.

    This function recursively walks through a fingy and verifies
    that all tracked objects are Lazy instances (not already instantiated).
    This is useful to ensure a fingy can be safely mutated before instantiation.

    Args:
        obj: The object to check. Can be a Lazy instance, dataclass, dict, list,
             or any other value.

    Returns:
        True if all tracked objects in the structure are Lazy instances,
        False if any tracked object has been instantiated.

    Examples:
        ```python
        @track
        class Inner:
            def __init__(self, value: int):
                self.value = value

        @track
        class Outer:
            def __init__(self, inner: Inner):
                self.inner = inner

        # All lazy - returns True
        config = Outer.lazy(inner=Inner.lazy(value=1))
        assert is_all_lazy(config)

        # Contains instantiated object - returns False
        config = Outer.lazy(inner=Inner(value=1))  # Inner is instantiated!
        assert not is_all_lazy(config)
        ```
    """
    from confingy.serde import HandlerRegistry

    handlers = HandlerRegistry.get_default_handlers()
    found_non_lazy = [False]  # Mutable for closure

    def check(value: Any) -> Any:
        """Check value and recurse into children. Returns value unchanged."""
        if found_non_lazy[0]:
            return value  # Short-circuit if we already found a non-lazy

        # Tracked instance = not lazy!
        if is_tracked_instance(value):
            found_non_lazy[0] = True
            return value

        # Lazy instance - check its config
        if is_lazy_instance(value):
            for v in value._confingy_config.values():
                check(v)
            return value

        # Use handlers to recurse into containers (dict, list, tuple, set, dataclass)
        for handler in handlers:
            if handler.can_handle(value):
                handler.map_children(value, check)
                return value

        return value

    check(obj)
    return not found_non_lazy[0]
