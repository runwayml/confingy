import functools
import inspect
import logging
from contextlib import contextmanager
from dataclasses import MISSING, Field
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

from pydantic import BaseModel, ConfigDict, create_model
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import TypeAliasType

from confingy.exceptions import (
    ValidationError,
)
from confingy.utils.hashing import hash_class
from confingy.utils.imports import get_module_name
from confingy.utils.types import is_lazy_instance, is_tracked_instance

# Global variable to disable validation of lazy and tracked objects
# this is used by the disable_validation context manager
G_DISABLE_VALIDATION: bool = False


@contextmanager
def disable_validation():
    """
    Context manager to disable validation for tracked and lazy objects.

    Examples:
        ```python
        class NonTrackedObject:
            def __init__(self, value):
                self.value = value

        class TrackedObject:
            def __init__(self, obj: NonTrackedObject):
                self.obj = obj

        # Raises validation error
        track(TrackedObject)(obj=NonTrackedObject(value=10))

        with disable_validation():
            # No validation error
            track(TrackedObject)(obj=NonTrackedObject(value=10))
        ```
    """
    global G_DISABLE_VALIDATION
    previous = G_DISABLE_VALIDATION
    G_DISABLE_VALIDATION = True
    try:
        yield
    finally:
        G_DISABLE_VALIDATION = previous


logger = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)
P = ParamSpec("P")


def is_class(obj: Any) -> TypeGuard[type[Any]]:
    """Check if an object is a class."""
    return isinstance(obj, type)


def _get_default_kwargs(cls: type[Any], init_method: Optional[Any] = None) -> dict:
    """Extract default keyword arguments from a class's __init__ signature."""
    if init_method is None:
        init_method = cls.__init__

    sig = inspect.signature(init_method)
    defaults = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        # Skip VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Capture parameters with defaults
        if param.default != inspect.Parameter.empty:
            default_value = param.default
            # Handle dataclass Field objects - call default_factory if present
            if isinstance(default_value, Field):
                if default_value.default_factory is not MISSING:
                    default_value = default_value.default_factory()
                elif default_value.default is not MISSING:
                    default_value = default_value.default
                else:
                    continue  # No default available
            defaults[param_name] = default_value

    return defaults


def _args_to_kwargs(
    cls: type[Any],
    args: tuple,
    kwargs: dict,
    init_method: Optional[Any] = None,
    include_defaults: bool = True,
) -> dict:
    """Convert positional arguments to keyword arguments.

    Args:
        cls: The class whose __init__ signature to inspect
        args: Positional arguments passed to __init__
        kwargs: Keyword arguments passed to __init__
        init_method: Optional specific __init__ method to inspect
        include_defaults: If True, merge in default values for parameters not provided

    Returns:
        Dictionary of all keyword arguments (explicit + defaults if requested)
    """
    if init_method is None:
        init_method = cls.__init__

    sig = inspect.signature(init_method)
    parameters = list(sig.parameters.keys())[1:]  # Skip 'self'

    init_kwargs = {}
    for i, param in enumerate(parameters[: len(args)]):
        init_kwargs[param] = args[i]
    init_kwargs.update(kwargs)

    if include_defaults:
        # Merge in defaults for params that have them
        defaults = _get_default_kwargs(cls, init_method)
        for key, value in defaults.items():
            if key not in init_kwargs:
                init_kwargs[key] = value

    return init_kwargs


def _create_validation_model(cls: type[Any]) -> type[BaseModel]:
    """Create a Pydantic validation model for a class's __init__ signature."""
    import warnings
    from typing import get_type_hints

    init_signature = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__, include_extras=False)

    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in init_signature.parameters.items():
        if param_name == "self":
            continue

        # Get type hint or use Any
        param_type = hints.get(param_name, Any)

        # Handle default values
        if param.default != inspect.Parameter.empty:
            fields[param_name] = (param_type, param.default)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs
            fields[param_name] = (param_type, {})
        else:
            fields[param_name] = (param_type, ...)  # Required field

    # Create the model, suppressing pydantic warnings about field names that
    # shadow BaseModel attributes (e.g. "schema", "validate", "copy").
    # These validation models are ephemeral and the shadowing is harmless.
    model_config: ConfigDict = {"arbitrary_types_allowed": True}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message='Field name ".*" in ".*" shadows an attribute in parent "BaseModel"',
            category=UserWarning,
        )
        return create_model(
            f"{cls.__name__}ValidationModel",
            __config__=model_config,
            **fields,  # type: ignore
        )


class Lazy(Generic[T]):
    """
    A proxy that delays instantiation until the object is actually used.

    This class wraps a configuration for an object and only creates the
    actual instance when the `instantiate()` method is called.

    This class is returned by the [lazy][confingy.tracking.lazy] function or when using the
    `Lazy` classmethod on a [@track][confingy.tracking.track]-decorated class.

    It can also be used as a type hint:
    ```python
    def process(model: Lazy[Model]):
        # model must be a Lazy[Model]
        actual_model = model.instantiate()

    All internal attributes of the Lazy object are prepended with '_confingy_' to avoid
    name collisions with the wrapped class's constructor arguments (including underscore-prefixed ones).
    ```
    """

    def __init__(
        self,
        cls: type,  # Untyped to allow T to be covariant
        config: dict[str, Any],
        skip_validation: bool = False,
        *,
        _was_instantiated: bool = False,
        _skip_post_config_hook: bool = False,
    ):
        self._confingy_cls = cls
        self._confingy_config = config
        self._confingy_was_instantiated = _was_instantiated

        # Handle lazy factory functions
        if hasattr(cls, "_original_cls"):
            self._confingy_actual_cls = cast(Any, cls)._original_cls
        else:
            self._confingy_actual_cls = cls

        if G_DISABLE_VALIDATION:
            skip_validation = True

        # Create validation model if validation is enabled
        self._confingy_validation_model: type[BaseModel] | None = None
        if not skip_validation:
            self._confingy_validation_model = _create_validation_model(
                self._confingy_actual_cls
            )
            self._validate_config()

        # Store metadata for serialization
        self._confingy_lazy_info = {
            "class": self._confingy_actual_cls.__name__,
            "module": get_module_name(self._confingy_actual_cls),
            "class_hash": hash_class(self._confingy_actual_cls),
        }

        # Initialize hook guard flag
        self._confingy_in_hook = False

        # Run post-config hook on initial creation (unless skipped)
        if not _skip_post_config_hook:
            self._run_post_config_hook()

    def _validate_config(self):
        """Validate the configuration against the class's __init__ signature."""
        if self._confingy_validation_model is None:
            return
        try:
            self._confingy_validation_model(**self._confingy_config)
        except PydanticValidationError as e:
            raise ValidationError(
                e, self._confingy_actual_cls.__name__, self._confingy_config
            ) from None

    def _run_post_config_hook(
        self, saved_config: dict[str, Any] | None = None, changed_key: str | None = None
    ) -> None:
        """Run the __post_config__ hook if defined on the class.

        The hook is called after config creation or update. It receives the Lazy
        instance and can modify it via attribute access. The hook should return
        the (possibly modified) Lazy instance.

        Args:
            saved_config: The config state before modification for rollback (only for updates)
            changed_key: The key that was changed (None if called from __init__)
        """
        if not hasattr(self._confingy_actual_cls, "__post_config__"):
            return

        if self._confingy_in_hook:
            return  # Prevent recursion

        self._confingy_in_hook = True
        try:
            result = self._confingy_actual_cls.__post_config__(self, changed_key)
            # If hook returns a different Lazy, use its config
            if result is not None and result is not self:
                self._confingy_config = result._confingy_config

            # Re-validate after hook completes to catch any invalid modifications
            # (e.g., direct _config manipulation or invalid values from returned Lazy)
            self._validate_config()
        except Exception:
            # Rollback entire config on hook failure (only if this was an update)
            if changed_key is not None and saved_config is not None:
                self._confingy_config = saved_config
            raise
        finally:
            self._confingy_in_hook = False

    def __getattr__(self, name: str) -> Any:
        """Access configuration parameters as attributes.

        This allows direct access to the constructor arguments stored in the Lazy config,
        enabling easy inspection and chained access for nested Lazy objects.

        Examples:
            ```python
            lazy = MyDataset.lazy(data=[1,2,3], processor=Pipeline.lazy(scalers=[...]))
            lazy.data           # Returns [1,2,3]
            lazy.processor      # Returns the Pipeline Lazy
            lazy.processor.scalers  # Chained access works!
            ```
        """
        # Check if the attribute exists in the validated config
        # NOTE: Use __dict__ to avoid recursion with __getattr__
        if (
            "_confingy_config" in self.__dict__
            and name in self.__dict__["_confingy_config"]
        ):
            return self._confingy_config[name]
        # Use __dict__.get() to avoid recursion during unpickling
        # (when __getattr__ is called before __init__ completes)
        actual_cls = self.__dict__.get("_confingy_actual_cls")
        config = self.__dict__.get("_confingy_config", {})
        cls_name = (
            getattr(actual_cls, "__name__", "Unknown") if actual_cls else "Unknown"
        )
        raise AttributeError(
            f"'{cls_name}' has no parameter '{name}'. "
            f"Available parameters: {list(config.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set configuration parameters as attributes with validation.

        Internal attributes (starting with '_confingy_') are set normally.
        Other attributes update the configuration and trigger validation.

        Examples:
            ```python
            lazy = MyDataset.lazy(data=[1,2,3], processor=p)
            lazy.data = [4,5,6]      # Updates config, validates
            lazy.processor = new_p   # Updates config, validates
            ```
        """
        # Internal confingy attributes are set normally on the object
        if name.startswith("_confingy_"):
            object.__setattr__(self, name, value)
            return

        # Ensure we're initialized
        if "_confingy_config" not in self.__dict__:
            raise AttributeError(
                f"Cannot set '{name}' - Lazy object not fully initialized"
            )

        # Check if this is a valid parameter
        if name not in self._confingy_config:
            raise AttributeError(
                f"'{self._confingy_actual_cls.__name__}' has no parameter '{name}'. "
                f"Available parameters: {list(self._confingy_config.keys())}"
            )

        # Save entire config before modification for potential rollback
        saved_config = self._confingy_config.copy()
        old_value = self._confingy_config[name]
        self._confingy_config[name] = value

        # Validate if validation is enabled
        if self._confingy_validation_model is not None:
            try:
                self._confingy_validation_model(**self._confingy_config)
            except PydanticValidationError as e:
                # Rollback on validation failure
                self._confingy_config[name] = old_value
                raise ValidationError(
                    e, self._confingy_actual_cls.__name__, self._confingy_config
                ) from None

        # Run post-config hook after successful update
        self._run_post_config_hook(saved_config=saved_config, changed_key=name)

    def get_config(self) -> dict[str, Any]:
        """
        Get a copy of the configuration used to create this lazy instance.
        The returned dictionary contains the constructor arguments for the lazy instance.
        """
        return self._confingy_config.copy()

    def copy(self, **updates: Any) -> "Lazy[T]":
        """
        Create a new Lazy instance with updated configuration.

        This provides an immutable update pattern - the original Lazy is unchanged,
        and a new Lazy is returned with the specified updates applied.

        Args:
            **updates: Keyword arguments to update in the new Lazy's config.
                      These override the original config values.

        Returns:
            A new Lazy instance with the updated configuration.

        Examples:
            ```python
            lazy = MyDataset.lazy(data=[1,2,3], processor=p)

            # Create a new Lazy with updated data
            new_lazy = lazy.copy(data=[4,5,6])

            # Original is unchanged
            assert lazy.data == [1,2,3]
            assert new_lazy.data == [4,5,6]

            # Chain copies for multiple updates
            another = lazy.copy(data=[7,8,9]).copy(processor=new_p)
            ```
        """
        # Start with current config
        new_config = self._confingy_config.copy()

        # Apply updates
        for key, value in updates.items():
            if key not in new_config:
                raise AttributeError(
                    f"'{self._confingy_actual_cls.__name__}' has no parameter '{key}'. "
                    f"Available parameters: {list(new_config.keys())}"
                )
            new_config[key] = value

        # Create new Lazy with updated config, preserving _was_instantiated
        # Skip validation if this Lazy came from lens() (type hints won't match)
        skip_val = (
            self._confingy_was_instantiated or self._confingy_validation_model is None
        )
        # Skip the post-config hook if we're being called from within a hook
        # (to prevent infinite recursion when hook uses copy() to return modified instance)
        skip_hook = self._confingy_in_hook
        return Lazy(
            self._confingy_cls,
            new_config,
            skip_validation=skip_val,
            _was_instantiated=self._confingy_was_instantiated,
            _skip_post_config_hook=skip_hook,
        )

    def instantiate(self) -> T:
        """Create and return an instance of the wrapped class.

        Each call creates a new instance - this is a factory method.

        Returns:
            A new instance of the wrapped class, constructed with the stored config.
        """
        logger.debug(f"Instantiating {self._confingy_actual_cls.__name__}")
        return self._confingy_actual_cls(**self._confingy_config)

    def unlens(self) -> Any:
        """Reconstruct the object, preserving the original laziness structure.

        When a Lazy is created via `lens()` from a tracked instance, calling
        `unlens()` will instantiate it. When created from an existing Lazy,
        it remains a Lazy.

        This enables a round-trip: `lens(obj) -> modify -> unlens()` preserves
        whether each node was originally Lazy or instantiated.

        Returns:
            Either an instantiated object or a new Lazy, depending on how
            this Lazy was created.

        Examples:
            ```python
            # From tracked instance - unlens() instantiates
            obj = Outer(middle=Middle(inner=Inner(value=42)))
            l = lens(obj)
            l.middle.inner.value = 100
            new_obj = l.unlens()  # Returns Outer instance

            # From Lazy - unlens() returns Lazy
            lazy = Outer.lazy(middle=Middle.lazy(inner=Inner.lazy(value=42)))
            l = lens(lazy)
            l.middle.inner.value = 100
            new_lazy = l.unlens()  # Returns Lazy[Outer]
            ```
        """
        from confingy.serde import HandlerRegistry

        handlers = HandlerRegistry.get_default_handlers()

        def unlens_value(value: Any) -> Any:
            """Recursively unlens a value, using handlers for containers."""
            if is_lazy_instance(value):
                return value.unlens()

            # Use handlers for container types
            for handler in handlers:
                if handler.can_handle(value):
                    return handler.map_children(value, unlens_value)

            return value

        # Process all config values
        realized_config = {k: unlens_value(v) for k, v in self._confingy_config.items()}

        if self._confingy_was_instantiated:
            # This Lazy was created from a tracked instance - instantiate
            return self._confingy_actual_cls(**realized_config)
        else:
            # This was originally a Lazy - return new Lazy
            # Always skip the post-config hook since unlens() is a structural
            # transformation, not a semantic creation. Hooks already ran on
            # setattr when values were modified.
            return Lazy(
                self._confingy_cls, realized_config, _skip_post_config_hook=True
            )

    def __call__(self, *args: Any, **kwargs: Any) -> "Lazy[T]":
        """Make Lazy callable to support the lazy(Class)(...) pattern.

        When called with no arguments, returns self.
        When called with arguments, merges them into the config and returns a new Lazy.
        """
        if not args and not kwargs:
            return self

        # Merge provided args into config
        new_kwargs = _args_to_kwargs(
            self._confingy_cls, args, kwargs, include_defaults=False
        )
        merged_config = {**self._confingy_config, **new_kwargs}

        # Preserve _was_instantiated and skip_validation like copy() does
        skip_val = (
            self._confingy_was_instantiated or self._confingy_validation_model is None
        )
        return Lazy(
            self._confingy_cls,
            merged_config,
            skip_validation=skip_val,
            _was_instantiated=self._confingy_was_instantiated,
        )

    def __repr__(self) -> str:
        cls_name = getattr(
            self._confingy_actual_cls, "__name__", str(self._confingy_actual_cls)
        )
        config_preview = {k: v for k, v in list(self._confingy_config.items())[:3]}
        if len(self._confingy_config) > 3:
            config_preview["..."] = f"and {len(self._confingy_config) - 3} more"
        return f"Lazy<{cls_name}>(config={config_preview})"

    def __getstate__(self) -> dict:
        """Prepare state for pickling, excluding unpicklable validation model."""
        state = self.__dict__.copy()
        # Track whether validation was enabled before pickling
        state["_confingy_had_validation"] = (
            state["_confingy_validation_model"] is not None
        )
        # Remove the dynamically-created validation model - it can't be pickled
        state["_confingy_validation_model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        # Extract and remove the temporary flag
        had_validation = state.pop("_confingy_had_validation", True)
        self.__dict__.update(state)
        # Only rebuild validation model if it was originally enabled
        if had_validation:
            self._confingy_validation_model = _create_validation_model(
                self._confingy_actual_cls
            )
        else:
            self._confingy_validation_model = None


# Type alias for values that may be lazy or already resolved
_MaybeLazyT = TypeVar("_MaybeLazyT")
MaybeLazy = TypeAliasType(
    "MaybeLazy", _MaybeLazyT | Lazy[_MaybeLazyT], type_params=(_MaybeLazyT,)
)


# Overloads for lazy() to provide IDE autocomplete
@overload
def lazy(cls: Callable[P, T]) -> Callable[P, Lazy[T]]: ...


@overload
def lazy(cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Lazy[T]: ...


def lazy(cls: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Create a lazy instance of a [@track][confingy.tracking.track]-decorated class.

    This function provides explicit lazy instantiation. Classes should be
    decorated with [@track][confingy.tracking.track], and then [lazy][confingy.tracking.lazy]
    is used when you want deferred instantiation.

    Examples:
        ```python
        @track
        class ExpensiveModel:
            def __init__(self, size: int):
                self.weights = np.random.randn(size, size)

        # Normal instantiation (immediate)
        model = ExpensiveModel(size=1000)

        # Lazy instantiation (deferred) - two ways:
        lazy_model = lazy(ExpensiveModel)(size=1000)  # Returns Lazy[ExpensiveModel]
        # Or more directly:
        lazy_model = lazy(ExpensiveModel, size=1000)  # Returns Lazy[ExpensiveModel]

        # Access when needed
        result = lazy_model.instantiate().forward(data)
        ```
    """
    if not is_class(cls):
        raise TypeError(f"lazy() requires a class, got {type(cls)}")

    if args or kwargs:
        # Direct instantiation: lazy(Class, arg1, arg2, ...)
        init_kwargs = _args_to_kwargs(cls, args, kwargs)
        return Lazy(cls, init_kwargs)
    else:
        # Factory pattern: lazy(Class) returns a function
        def lazy_factory(*factory_args: Any, **factory_kwargs: Any) -> Lazy[T]:
            init_kwargs = _args_to_kwargs(cls, factory_args, factory_kwargs)
            return Lazy(cls, init_kwargs)

        return lazy_factory


def lens(obj: Any) -> Lazy[Any]:
    """
    Convert a tracked or Lazy instance to a Lazy for nested parameter updates.

    This function provides a unified interface for modifying nested configurations.
    After making changes, call `unlens()` to reconstruct the object with the
    original laziness structure preserved.

    Args:
        obj: Either a tracked instance (has `_tracked_info`) or a Lazy instance.

    Returns:
        A Lazy instance that can be modified via attribute access.

    Examples:
        ```python
        @track
        class Outer:
            def __init__(self, middle: Middle):
                self.middle = middle

        @track
        class Middle:
            def __init__(self, inner: Inner):
                self.inner = inner

        @track
        class Inner:
            def __init__(self, value: int):
                self.value = value

        # Create a tracked instance
        obj = Outer(middle=Middle(inner=Inner(value=42)))

        # Use lens to modify nested values
        l = lens(obj)
        l.middle.inner.value = 100

        # Reconstruct with original structure (all instantiated)
        new_obj = l.unlens()
        assert new_obj.middle.inner.value == 100

        # Works with Lazy instances too
        lazy_obj = Outer.lazy(middle=Middle.lazy(inner=Inner.lazy(value=42)))
        l = lens(lazy_obj)
        l.middle.inner.value = 100
        new_lazy = l.unlens()  # Returns Lazy since original was Lazy
        ```
    """
    from confingy.serde import HandlerRegistry

    handlers = HandlerRegistry.get_default_handlers()

    def lens_value(value: Any) -> Any:
        """Recursively convert tracked instances to Lazy."""
        if is_tracked_instance(value):
            return lens(value)
        if is_lazy_instance(value):
            # Recurse into Lazy config in case it contains tracked instances
            return lens(value)

        # Use handlers for container types
        for handler in handlers:
            if handler.can_handle(value):
                return handler.map_children(value, lens_value)

        return value

    if is_lazy_instance(obj):
        # Recurse into config to convert any nested tracked instances
        new_config = {k: lens_value(v) for k, v in obj._confingy_config.items()}

        # Only create new Lazy if config actually changed
        if any(new_config[k] is not obj._confingy_config[k] for k in new_config):
            return Lazy(
                obj._confingy_cls,
                new_config,
                skip_validation=True,  # Config comes from valid source
                _was_instantiated=getattr(obj, "_confingy_was_instantiated", False),
                _skip_post_config_hook=True,  # lens() is just wrapping, don't run hooks
            )
        return obj

    if is_tracked_instance(obj):
        # Convert tracked instance to Lazy with _was_instantiated=True
        # Skip validation since the tracked instance was already valid
        config = {k: lens_value(v) for k, v in obj._tracked_info["init_args"].items()}
        return Lazy(
            type(obj),
            config,
            skip_validation=True,
            _was_instantiated=True,
            _skip_post_config_hook=True,  # lens() is just wrapping, don't run hooks
        )

    raise TypeError(
        f"lens() requires a Lazy or tracked instance, got {type(obj).__name__}"
    )


C = TypeVar("C", bound=type)


# Overloads for track() to provide IDE autocomplete
# We use `C` (bound to type) to preserve class identity and constructor signature
# through inheritance. This means `lazy` won't be recognized by pyright, but
# constructor autocomplete will work correctly for subclasses.
@overload
def track(
    cls_or_instance: None = None,
    *,
    _validate: bool = True,
) -> Callable[[C], C]: ...


@overload
def track(
    cls_or_instance: C,
    *,
    _validate: bool = True,
) -> C: ...


def track(
    cls_or_instance: Optional[Any] = None,
    *args: Any,
    _validate: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Track constructor arguments for serialization.

    Can be used as a decorator or function to enable argument tracking
    for later serialization.

    Args:
        _validate: Whether to validate constructor arguments with pydantic (default: True)
            can be overridden by the context manager [disable_validation][confingy.tracking.disable_validation]

    Example:
        ```python
        from confingy import track, save_config

        @track
        class Dataset:
            def __init__(self, path: str, size: int):
                self.path = path
                self.size = size

        # Arguments are tracked and can be serialized
        dataset = Dataset(path="/data", size=1000)
        save_config(dataset, "config.json")

        # Skip validation
        @track(_validate=False)
        class FastDataset:
            def __init__(self, path: str):
                self.path = path
        ```

        You can also turn a tracked class into a lazy one:

        ```python
        lazy_dataset = Dataset.lazy(path="/data", size=1000)
        ```
    """
    # Case 1: Called as @track() or @track(_validate=False) with parentheses - return decorator

    if G_DISABLE_VALIDATION:
        _validate = False

    if cls_or_instance is None:
        return functools.partial(track, _validate=_validate)

    # Case 2: Called with arguments - instantiate class with tracking
    if args or kwargs:
        return _track_with_args(cls_or_instance, args, kwargs, _validate=_validate)

    # Case 3: Called as @track decorator or track(instance)
    if is_class(cls_or_instance):
        return _track_class_decorator(cls_or_instance, _validate=_validate)
    else:
        return _track_existing_instance(cls_or_instance)


def _track_with_args(
    cls: Any, args: tuple, kwargs: dict, _validate: bool = True
) -> Any:
    """Handle track(Class, arg1=val1, arg2=val2) - instantiate with tracking."""
    if not is_class(cls):
        raise TypeError("Expected a class when arguments are provided")
    return _create_tracked_instance(cls, args, kwargs, _validate=_validate)


def _track_class_decorator(cls: type[Any], _validate: bool = True) -> type[Any]:
    """Handle @track decorator on a class."""
    return _add_tracking_to_class(cls, _validate=_validate)


def _track_existing_instance(instance: Any) -> Any:
    """Handle track(existing_instance) - add tracking to existing instance."""
    return _add_tracking_to_instance(instance)


def _add_tracking_to_class(cls: type[Any], _validate: bool = True) -> type[Any]:
    """Add tracking to a class's __init__ method.

    Args:
        cls: The class to add tracking to.
        _validate: Whether to validate constructor arguments with pydantic.
    """
    # Check if this specific class (not inherited) already has a lazy attribute
    # Use __dict__ to check only the class's own attributes
    if "lazy" in cls.__dict__:
        # If it's our lazy_classmethod, the class is already tracked
        if hasattr(cls.lazy, "__name__") and cls.lazy.__name__ == "lazy_classmethod":
            return cls
        # Otherwise it's a user-defined attribute we shouldn't clobber
        else:
            raise AttributeError(
                f"Class {cls.__name__} already has a 'lazy' attribute. "
                f"The @track decorator would overwrite it, which is not allowed."
            )

    original_init = cls.__init__
    validation_model = _create_validation_model(cls) if _validate else None

    # Create a new subclass to avoid mutating the original class.
    # This ensures that track(SomeClass)(args) doesn't globally modify SomeClass.
    # Preserve __orig_bases__ so the metaclass correctly derives __parameters__
    # for Generic classes (e.g. UDF(Generic[InputT, OutputT])).
    cls_dict: dict[str, Any] = {}
    if "__orig_bases__" in cls.__dict__:
        cls_dict["__orig_bases__"] = cls.__dict__["__orig_bases__"]
    new_cls = type(cls)(cls.__name__, (cls,), cls_dict)  # type: ignore[misc]
    new_cls.__module__ = cls.__module__
    new_cls.__qualname__ = cls.__qualname__

    @functools.wraps(original_init)
    def init_with_tracking(self: Any, *args: Any, **kwargs: Any) -> None:
        # Only set tracking info if it hasn't been set yet
        # This ensures child classes don't get overwritten by parent classes
        should_track = not hasattr(self, "_tracked_info")

        if should_track:
            # Convert arguments using the actual runtime class
            init_kwargs = _args_to_kwargs(
                self.__class__, args, kwargs, self.__class__.__init__
            )

            if _validate and validation_model is not None:
                try:
                    # Validate but keep original objects instead of converting to dict
                    validation_model(
                        **init_kwargs
                    )  # Just validate, don't use the result
                    # Store the original init_kwargs, not the dumped version
                    stored_kwargs = init_kwargs
                except PydanticValidationError as e:
                    raise ValidationError(e, cls.__name__, init_kwargs) from None
            else:
                # Skip validation
                stored_kwargs = init_kwargs

            # Store tracking info with original objects preserved
            self._tracked_info = {
                "class": self.__class__.__name__,
                "module": get_module_name(self.__class__),
                "init_args": stored_kwargs,
                "class_hash": hash_class(self.__class__),
            }

        # Call original __init__, injecting resolved Field defaults so that
        # default_factory values are passed through (matching lazy+instantiate behavior).
        if should_track:
            original_init(self, **init_kwargs)
        else:
            original_init(self, *args, **kwargs)

    # Add the lazy classmethod
    def lazy_classmethod(cls_arg: type[T], *args: Any, **kwargs: Any) -> Lazy[T]:
        """
        Create a lazy instance of this class.

        Always returns a Lazy instance directly. Validation will fail if
        required parameters are not provided.

        Examples:
            ```python
            @track
            class MyModel:
                def __init__(self, size: int):
                    self.size = size

            # Create lazy config:
            lazy_model = MyModel.lazy(size=100)

            # This matches lazy(MyModel)() behavior:
            # MyModel.lazy() will fail if required args missing
            ```
        """
        # Always create Lazy directly - matches lazy(Cls)() behavior
        init_kwargs = _args_to_kwargs(cls_arg, args, kwargs)
        return Lazy(cls_arg, init_kwargs)

    def __reduce__(self: Any) -> tuple:
        """Enable pickling of tracked instances.

        The dynamically-created tracked subclass can't be found by pickle via
        module path. We reconstruct by creating an empty tracked subclass via
        __new__ (skipping __init__) and restoring __dict__ via __setstate__.
        """
        tracked_info = self._tracked_info
        return (
            _reconstruct_tracked_instance,
            (tracked_info["module"], tracked_info["class"]),
            self.__dict__,
        )

    def __setstate__(self: Any, state: dict) -> None:
        """Restore extra instance state after reconstruction."""
        self.__dict__.update(state)

    new_cls.__init__ = init_with_tracking  # type: ignore
    new_cls.__reduce__ = __reduce__
    # Only add __setstate__ if the original class doesn't define one,
    # to respect any custom unpickling logic.
    if "__setstate__" not in cls.__dict__:
        new_cls.__setstate__ = __setstate__

    # Always add lazy classmethod to each tracked class (even if inherited)
    # so that each class has the correct signature matching its __init__
    new_cls.lazy = classmethod(lazy_classmethod)  # type: ignore

    return new_cls


def _reconstruct_tracked_instance(module: str, class_name: str) -> Any:
    """Reconstruct an empty tracked instance for unpickling.

    Imports the original class from its module, wraps it with track(),
    and creates an empty instance via __new__ (skipping __init__).
    State is restored separately by pickle calling __setstate__.
    """
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, class_name)
    tracked_cls = track(cls)
    return tracked_cls.__new__(tracked_cls)


def _create_tracked_instance(
    cls: type[Any], args: tuple, kwargs: dict, _validate: bool = True
) -> Any:
    """Create an instance with tracking information."""
    init_kwargs = _args_to_kwargs(cls, args, kwargs)

    if _validate:
        validation_model = _create_validation_model(cls)
        try:
            # Validate but keep original objects instead of converting to dict
            validation_model(**init_kwargs)  # Just validate, don't use the result
            # Store the original init_kwargs, not the dumped version
            stored_kwargs = init_kwargs
        except PydanticValidationError as e:
            raise ValidationError(e, cls.__name__, init_kwargs) from None
    else:
        # Skip validation
        stored_kwargs = init_kwargs

    instance = cls(*args, **kwargs)
    instance._tracked_info = {  # type: ignore
        "class": cls.__name__,
        "module": get_module_name(cls),
        "init_args": stored_kwargs,
        "class_hash": hash_class(cls),
    }

    return instance


def _add_tracking_to_instance(instance: Any) -> Any:
    """Add tracking information to an existing instance."""
    cls = instance.__class__

    # Try to extract init args from attributes
    init_kwargs = {
        key: value
        for key, value in instance.__dict__.items()
        if not key.startswith("_")
    }

    instance._tracked_info = {  # type: ignore
        "class": cls.__name__,
        "module": get_module_name(cls),
        "init_args": init_kwargs,
        "class_hash": hash_class(cls),
    }

    return instance


def update(parent_obj: Any) -> Callable[..., Any]:
    """
    Create an updated version of a tracked or lazy object with new constructor arguments.

    This function supports "inheritance" for confingy objects by allowing you to create
    a new instance with updated parameters while preserving the original object's type
    and validation behavior.

    Args:
        parent_obj: Either a tracked instance (has `_tracked_info`) or a lazy instance (`Lazy[T]`)

    Returns:
        A function that accepts new constructor arguments and returns:
        - For tracked instances: A new tracked instance of the same type
        - For lazy instances: A new lazy instance with updated configuration

    Examples:
        ```python
        # With tracked instances
        @track
        class Foo:
            def __init__(self, bar: str, baz: int = 10):
                self.bar = bar
                self.baz = baz

        parent_foo = track(Foo)(bar="hello")
        child_foo = update(parent_foo)(bar="world")  # bar="world", baz=10

        # With lazy instances
        parent_lazy = lazy(Foo)(bar="hello")
        child_lazy = update(parent_lazy)(bar="world")  # Returns Lazy[Foo]
        ```
    """

    def updater(*args: Any, **kwargs: Any) -> Any:
        # Handle Lazy instances
        if is_lazy_instance(parent_obj):
            # Get the original configuration
            original_config = parent_obj.get_config()

            # Merge with new arguments (new args take precedence)
            updated_config = original_config.copy()

            # Handle positional arguments by converting to kwargs
            if args:
                new_kwargs = _args_to_kwargs(
                    parent_obj._confingy_actual_cls, args, kwargs
                )
            else:
                new_kwargs = kwargs

            updated_config.update(new_kwargs)

            # Create a new Lazy instance with updated config
            return Lazy(parent_obj._confingy_cls, updated_config)

        # Handle tracked instances
        elif hasattr(parent_obj, "_tracked_info"):
            # Get the class directly from the object
            cls = parent_obj.__class__

            # Get original init args from tracked info
            original_args = parent_obj._tracked_info["init_args"]

            # Merge with new arguments (new args take precedence)
            updated_args = original_args.copy()

            # Handle positional arguments by converting to kwargs
            if args:
                new_kwargs = _args_to_kwargs(cls, args, kwargs)
            else:
                new_kwargs = kwargs

            updated_args.update(new_kwargs)

            # Create new tracked instance
            return _create_tracked_instance(cls, (), updated_args, _validate=True)

        else:
            raise TypeError(
                f"update() requires either a tracked instance (with _tracked_info) "
                f"or a Lazy instance, got {type(parent_obj)}"
            )

    return updater
