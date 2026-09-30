import functools
import inspect
import logging
import os
import sys
import types
import warnings
import weakref
from collections.abc import MutableMapping
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
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import pydantic
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, create_model
from pydantic import Field as PydanticField
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypeAliasType

from confingy.exceptions import (
    ValidationError,
)
from confingy.utils.containers import same_structure
from confingy.utils.hashing import hash_class
from confingy.utils.imports import get_class_name, get_module_name, import_qualname
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
        if param.default is not inspect.Parameter.empty:
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
    params = list(sig.parameters.values())[1:]  # Skip 'self'
    var_positional = next(
        (p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None
    )

    # Bind positional args and keyword args to the signature, so *args are
    # collected into a tuple. Other keywords (unknown ones, those collected by
    # **kwargs, and positional-only parameters given by name) are stored as-is
    # rather than raising here, so validation can report unknown ones.
    bindable = {
        p.name
        for p in params
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    bound = sig.bind_partial(
        None, *args, **{k: v for k, v in kwargs.items() if k in bindable}
    )

    init_kwargs = dict(list(bound.arguments.items())[1:])  # Skip 'self'
    init_kwargs.update({k: v for k, v in kwargs.items() if k not in bindable})

    if include_defaults:
        # Merge in defaults for params that have them
        defaults = _get_default_kwargs(cls, init_method)
        for key, value in defaults.items():
            if key not in init_kwargs:
                init_kwargs[key] = value
        if var_positional is not None and var_positional.name not in init_kwargs:
            init_kwargs[var_positional.name] = ()

    # Keep arguments in signature order, however the call was written, followed
    # by any extra keywords collected by **kwargs in the order they were given
    position = {p.name: i for i, p in enumerate(params)}
    return dict(
        sorted(init_kwargs.items(), key=lambda item: position.get(item[0], len(params)))
    )


def _kwargs_to_call_args(
    init_method: Any, config: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Convert a stored config back into positional and keyword call arguments.

    This is the inverse of `_args_to_kwargs`: positional-only parameters and the
    contents of `*args` are passed positionally, and everything else by keyword.

    Args:
        init_method: The `__init__` method whose signature the config follows.
        config: The stored constructor arguments.

    Returns:
        An `(args, kwargs)` tuple to call the class with.
    """
    params = list(inspect.signature(init_method).parameters.values())[1:]
    kwargs = dict(config)
    var_positional = next(
        (p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None
    )
    extra: tuple[Any, ...] = ()
    if var_positional is not None:
        extra = tuple(kwargs.pop(var_positional.name, ()))

    args: list[Any] = []
    for p in params:
        # Parameters before *args must be passed positionally when *args is used
        positional = p.kind == inspect.Parameter.POSITIONAL_ONLY or (
            extra and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if not positional or p.name not in kwargs:
            break
        args.append(kwargs.pop(p.name))
    return (*args, *extra), kwargs


def _instantiate(cls: type[Any], config: dict[str, Any]) -> Any:
    """Call `cls` with a stored config."""
    args, kwargs = _kwargs_to_call_args(cls.__init__, config)
    return cls(*args, **kwargs)


# BaseModel attribute names, which pydantic ignores (model_config) or rejects
# (e.g. model_dump) as field names
_PYDANTIC_RESERVED_NAMES = frozenset(dir(BaseModel))


def _build_validation_model(cls: type[Any]) -> type[BaseModel]:
    """Build a Pydantic validation model for a class's __init__ signature."""
    from typing import get_type_hints

    init_signature = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__, include_extras=False)

    fields: dict[str, tuple[Any, Any]] = {}
    accepts_var_keyword = False
    for i, (param_name, param) in enumerate(init_signature.parameters.items()):
        if param_name == "self":
            continue

        # Get type hint or use Any
        param_type = hints.get(param_name, Any)

        # Handle default values
        default: Any
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # *args, stored as a tuple
            param_type = tuple[param_type, ...]  # type: ignore[valid-type]
            default = ()
        elif param.default is not inspect.Parameter.empty:
            default = param.default
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs
            accepts_var_keyword = True
            default = {}
        else:
            default = ...  # Required field

        if param_name.startswith("_") or param_name in _PYDANTIC_RESERVED_NAMES:
            # Pydantic doesn't allow field names with leading underscores, and
            # names of BaseModel attributes (e.g. model_config) are ignored or
            # rejected as fields, so validate these under an internal name,
            # aliased to the parameter
            fields[f"confingy_param_{i}"] = (
                param_type,
                PydanticField(default, alias=param_name),
            )
        else:
            fields[param_name] = (param_type, default)

    # Create the model, suppressing pydantic warnings about field names that
    # shadow BaseModel attributes (e.g. "schema", "validate", "copy").
    # These validation models are ephemeral and the shadowing is harmless.
    # Unknown arguments are rejected unless __init__ accepts **kwargs, so a typo
    # fails when a Lazy is created rather than at instantiate().
    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
        "extra": "allow" if accepts_var_keyword else "forbid",
    }
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


# Validation models are a pure function of the class, so they are cached and
# shared by every Lazy/tracked instance of that class. A WeakKeyDictionary is
# used (rather than functools.lru_cache) because track() creates a new subclass
# per call, so a strong-ref cache would keep short-lived dynamic classes alive
# forever.
_VALIDATION_MODEL_CACHE: MutableMapping[type[Any], type[BaseModel]] = (
    weakref.WeakKeyDictionary()
)


def _create_validation_model(cls: type[Any]) -> type[BaseModel]:
    """Get the Pydantic validation model for a class's __init__ signature.

    The model is built once per class and cached. This assumes a class's
    `__init__` signature and type hints do not change after the model is first
    built, which holds for normal (module-level) class definitions.

    Args:
        cls: The class whose `__init__` arguments should be validated.

    Returns:
        The validation model for `cls`.
    """
    model = _VALIDATION_MODEL_CACHE.get(cls)
    if model is None:
        # Errors (e.g. unresolvable forward refs) propagate and are not cached.
        model = _build_validation_model(cls)
        _VALIDATION_MODEL_CACHE[cls] = model
    return model


def _validation_field_names(model: type[BaseModel]) -> set[str]:
    """Get the argument names a validation model accepts, resolving aliases."""
    return {field.alias or name for name, field in model.model_fields.items()}


_UNION_TYPES = (Union, types.UnionType)


def _lazy_hint_targets(target: Any) -> tuple[type, ...]:
    """Get the classes a `Lazy[target]` hint checks against.

    Unions are expanded into their members, and generic aliases (e.g.
    `list[int]`) are reduced to their origin class. `None` members are dropped,
    since a Lazy is never None, so `Lazy[Optional[X]]` checks against `X`. If any
    other member can't be checked by subclassing (see `_is_checkable_class`),
    the hint matches any Lazy, which is signalled by an empty result.

    Args:
        target: The type argument of the `Lazy[...]` hint.

    Returns:
        The classes to check against, or an empty tuple to match any Lazy.
    """
    members = get_args(target) if get_origin(target) in _UNION_TYPES else (target,)
    targets = []
    for member in members:
        if member is type(None):
            continue
        member = get_origin(member) or member
        if not _is_checkable_class(member):
            return ()
        targets.append(member)
    return tuple(targets)


def _is_checkable_class(target: Any) -> bool:
    """Check whether being a subclass of `target` is meaningful for a Lazy's class.

    `Any` and `object` match everything, and type variables and other special
    forms aren't classes. ABCs and Protocols with structural subclass checks
    (e.g. `Iterable`, which only looks for `__iter__`) don't reliably reflect how
    a class is used, so they aren't checked either.
    """
    if not isinstance(target, type) or target in (Any, object):
        return False
    if getattr(target, "_is_protocol", False):
        return False
    return not any("__subclasshook__" in vars(base) for base in target.__mro__[:-1])


def _is_subclass_of_any(actual: Any, targets: tuple[type, ...]) -> bool:
    """Check whether `actual` is a subclass of any of `targets`."""
    try:
        return isinstance(actual, type) and issubclass(actual, targets)
    except TypeError:
        return True


def _warn_lazy_hint_mismatch(targets: tuple[type, ...], actual: Any) -> None:
    """Warn that a Lazy's class doesn't match its `Lazy[...]` hint.

    Python's default warning filter shows this once per location in user code,
    so e.g. building many mismatched Lazies in a loop warns once.
    """
    expected = " | ".join(t.__name__ for t in targets)
    warnings.warn(
        f"Expected Lazy[{expected}], got Lazy[{getattr(actual, '__name__', actual)}], "
        f"which is not a subclass of {expected}.",
        UserWarning,
        stacklevel=_stacklevel_outside_confingy(),
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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Validate `Lazy[X]` type hints.

        The value must be a Lazy. If its class isn't `X` or a subclass (or a
        member of `X`, for unions), a warning is emitted rather than an error,
        since configs often rely on duck typing. See `_lazy_hint_targets` for the
        targets that match any Lazy.
        """
        args = get_args(source)
        targets = _lazy_hint_targets(args[0]) if args else ()

        def validate(value: Any) -> Any:
            if not isinstance(value, Lazy):
                raise ValueError(f"Expected a Lazy, got {type(value).__name__}")
            if targets and not _is_subclass_of_any(value._confingy_actual_cls, targets):
                _warn_lazy_hint_mismatch(targets, value._confingy_actual_cls)
            return value

        return core_schema.no_info_plain_validator_function(validate)

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

        _warn_if_params_shadow_lazy(self._confingy_actual_cls)

        # Create validation model if validation is enabled
        self._confingy_validation_model: type[BaseModel] | None = None
        if not skip_validation:
            self._confingy_validation_model = _create_validation_model(
                self._confingy_actual_cls
            )
            self._validate_config()

        # Store metadata for serialization
        self._confingy_lazy_info = {
            "class": get_class_name(self._confingy_actual_cls),
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
        return _instantiate(self._confingy_actual_cls, self._confingy_config)

    def unlens(self) -> Any:
        """Reconstruct the object, preserving the original laziness structure.

        When a Lazy is created via `lens()` from a tracked instance, calling
        `unlens()` will instantiate it. When created from an existing Lazy,
        it remains a Lazy.

        This enables a round-trip: `lens(obj) -> modify -> unlens()` preserves
        whether each node was originally Lazy or instantiated.

        Only modified nodes and their ancestors are rebuilt: a node that was not
        changed after `lens()` is returned as the original object. A node shared
        by several parents is rebuilt once and stays shared.

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
        return self._unlens({})

    def _unlens(self, memo: dict[int, Any]) -> Any:
        """Implementation of `unlens()`, memoised by id so shared nodes stay shared."""
        if id(self) in memo:
            return memo[id(self)]

        from confingy.serde import HandlerRegistry

        handlers = HandlerRegistry.get_default_handlers()

        def unlens_value(value: Any) -> Any:
            """Recursively unlens a value, using handlers for containers."""
            if is_lazy_instance(value):
                return value._unlens(memo)

            # Use handlers for container types
            for handler in handlers:
                if handler.can_handle(value):
                    return handler.map_children(value, unlens_value)

            return value

        # Process all config values
        realized_config = {k: unlens_value(v) for k, v in self._confingy_config.items()}

        # Reuse the object this Lazy was lensed from if nothing changed
        source = self.__dict__.get("_confingy_source")
        source_config = None
        if is_lazy_instance(source):
            source_config = source._confingy_config
        elif is_tracked_instance(source):
            source_config = source._tracked_info["init_args"]

        result: Any
        if source_config is not None and _same_config(realized_config, source_config):
            result = source
        elif self._confingy_was_instantiated:
            # This Lazy was created from a tracked instance - instantiate. Use
            # _create_tracked_instance so the result stays tracked even if the
            # class isn't decorated (e.g. created with track(Cls, ...) or
            # deserialized from an undecorated class).
            result = _create_tracked_instance(
                self._confingy_actual_cls, (), realized_config, _validate=False
            )
        else:
            # This was originally a Lazy - return new Lazy
            # Always skip the post-config hook since unlens() is a structural
            # transformation, not a semantic creation. Hooks already ran on
            # setattr when values were modified.
            result = Lazy(
                self._confingy_cls, realized_config, _skip_post_config_hook=True
            )
        memo[id(self)] = result
        return result

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


# Public Lazy attributes, which take precedence over config values of the same name
_LAZY_ATTRIBUTES = frozenset(name for name in dir(Lazy) if not name.startswith("_"))

# Classes already checked by _warn_if_params_shadow_lazy
_CHECKED_FOR_SHADOWING: "weakref.WeakSet[type]" = weakref.WeakSet()


def _warn_if_params_shadow_lazy(cls: type[Any], *also_checked: type[Any]) -> None:
    """Warn once per class if `__init__` parameters clash with `Lazy` attributes.

    A `Lazy` exposes its config as attributes, but its own methods (e.g. `copy`,
    `instantiate`) take precedence, so a parameter with one of those names can't
    be read as `lazy.<name>`. Parameters starting with `_confingy_` clash with
    `Lazy`'s internal attributes.

    Args:
        cls: The class to check.
        *also_checked: Classes with the same signature (e.g. the subclass created
            by `@track`) to mark as checked, so they don't warn again.
    """
    if cls in _CHECKED_FOR_SHADOWING:
        return
    for checked in (cls, *also_checked):
        _CHECKED_FOR_SHADOWING.add(checked)
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return
    shadowed = [name for name in params if name in _LAZY_ATTRIBUTES]
    reserved = [name for name in params if name.startswith("_confingy_")]
    if shadowed:
        warnings.warn(
            f"{cls.__qualname__}.__init__ has parameters named "
            f"{', '.join(map(repr, shadowed))}, which clash with Lazy methods. "
            f"On a Lazy[{cls.__name__}], attribute access returns the method, not "
            "the parameter; use lazy.get_config()[name] to read it.",
            UserWarning,
            stacklevel=_stacklevel_outside_confingy(),
        )
    if reserved:
        warnings.warn(
            f"{cls.__qualname__}.__init__ has parameters named "
            f"{', '.join(map(repr, reserved))}. Names starting with '_confingy_' "
            "are reserved for Lazy's internal attributes and won't work as "
            "parameters of a Lazy.",
            UserWarning,
            stacklevel=_stacklevel_outside_confingy(),
        )


def _stacklevel_outside_confingy() -> int:
    """Get the `warnings.warn` stacklevel of the first caller outside confingy.

    Meant to be called as the `stacklevel` argument of `warnings.warn`, so the
    warning points at user code however deep inside confingy (or pydantic
    validation) it is raised.
    """
    # Warnings raised during validation have pydantic's frames in between
    package_dirs = tuple(
        os.path.dirname(os.path.abspath(module.__file__ or "")) + os.sep
        for module in (sys.modules[__name__], pydantic)
    )
    frame = inspect.currentframe()
    # Level 1 is the function calling warnings.warn
    frame = frame.f_back if frame is not None else None
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        package_dirs
    ):
        frame = frame.f_back
        level += 1
    return level


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

    An object shared by several parents is lensed once, so editing it through one
    parent is visible through the others, and it stays shared after `unlens()`.
    """
    if not (is_lazy_instance(obj) or is_tracked_instance(obj)):
        raise TypeError(
            f"lens() requires a Lazy or tracked instance, got {type(obj).__name__}"
        )
    return _lens(obj, {})


def _lens(obj: Any, memo: dict[int, Any]) -> Any:
    """Implementation of `lens()`, memoised by id so shared nodes stay shared."""
    if id(obj) in memo:
        return memo[id(obj)]

    from confingy.serde import HandlerRegistry

    handlers = HandlerRegistry.get_default_handlers()
    converted_tracked = False

    def lens_value(value: Any) -> Any:
        """Recursively convert tracked instances to Lazy."""
        nonlocal converted_tracked
        if is_tracked_instance(value) or is_lazy_instance(value):
            converted_tracked = converted_tracked or is_tracked_instance(value)
            # Recurse into Lazy config too, in case it contains tracked instances
            return _lens(value, memo)

        # Use handlers for container types
        for handler in handlers:
            if handler.can_handle(value):
                return handler.map_children(value, lens_value)

        return value

    result: Any
    if is_lazy_instance(obj):
        # Recurse into config to convert any nested tracked instances
        new_config = {k: lens_value(v) for k, v in obj._confingy_config.items()}

        # Always copy, so edits made through the lens never mutate the input.
        # Keep validating edits unless tracked children were converted to Lazy,
        # since the class's type hints expect instances there.
        result = Lazy(
            obj._confingy_cls,
            new_config,
            skip_validation=(
                converted_tracked or obj._confingy_validation_model is None
            ),
            _was_instantiated=obj._confingy_was_instantiated,
            _skip_post_config_hook=True,  # lens() is just wrapping, don't run hooks
        )
        result._confingy_source = obj
    else:
        # Convert tracked instance to Lazy with _was_instantiated=True
        # Skip validation since the tracked instance was already valid
        config = {k: lens_value(v) for k, v in obj._tracked_info["init_args"].items()}
        result = Lazy(
            type(obj),
            config,
            skip_validation=True,
            _was_instantiated=True,
            _skip_post_config_hook=True,  # lens() is just wrapping, don't run hooks
        )
        result._confingy_source = obj

    memo[id(obj)] = result
    return result


def _same_config(config: dict[str, Any], source_config: dict[str, Any]) -> bool:
    """Check whether a realized config still matches the config it was lensed from."""
    return config.keys() == source_config.keys() and all(
        same_structure(config[k], source_config[k]) for k in config
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
    _warn_if_params_shadow_lazy(cls, new_cls)

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

            if _validate and not G_DISABLE_VALIDATION:
                # Built on first use rather than at decoration time, so type hints
                # can refer to names defined later in the module (e.g. the class
                # itself, with `from __future__ import annotations`).
                validation_model = _create_validation_model(cls)
                to_validate = init_kwargs
                if type(self).__init__ is not init_with_tracking:
                    # An untracked subclass overrode __init__, so init_kwargs follow
                    # its signature and may include arguments cls doesn't declare.
                    # Only check the ones cls does.
                    to_validate = {
                        k: v
                        for k, v in init_kwargs.items()
                        if k in _validation_field_names(validation_model)
                    }
                try:
                    # Validate but keep original objects instead of converting to dict
                    validation_model(
                        **to_validate
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
                "class": get_class_name(self.__class__),
                "module": get_module_name(self.__class__),
                "init_args": stored_kwargs,
                "class_hash": hash_class(self.__class__),
            }

        # Call original __init__, injecting resolved Field defaults so that
        # default_factory values are passed through (matching lazy+instantiate
        # behavior). init_kwargs were resolved against the *runtime* class's
        # signature, so only forward them when that signature is the one being
        # wrapped here. If an untracked subclass overrides __init__ and calls
        # super().__init__(...), init_kwargs belong to the subclass and must
        # not be forwarded to original_init.
        if should_track and type(self).__init__ is init_with_tracking:
            call_args, call_kwargs = _kwargs_to_call_args(original_init, init_kwargs)
            original_init(self, *call_args, **call_kwargs)
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
    cls = import_qualname(module, class_name)
    tracked_cls = track(cls)
    return tracked_cls.__new__(tracked_cls)


def _create_tracked_instance(
    cls: type[Any], args: tuple, kwargs: dict, _validate: bool = True
) -> Any:
    """Create an instance with tracking information."""
    init_kwargs = _args_to_kwargs(cls, args, kwargs)

    if _validate and not G_DISABLE_VALIDATION:
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

    instance = cls(*args, **kwargs) if args else _instantiate(cls, kwargs)
    instance._tracked_info = {  # type: ignore
        "class": get_class_name(cls),
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
        "class": get_class_name(cls),
        "module": get_module_name(cls),
        "init_args": init_kwargs,
        "class_hash": hash_class(cls),
        # init_args holds attributes, not constructor arguments, so this
        # instance can't be rebuilt from them
        "from_instance": True,
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
                    parent_obj._confingy_actual_cls,
                    args,
                    kwargs,
                    include_defaults=False,
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
                new_kwargs = _args_to_kwargs(cls, args, kwargs, include_defaults=False)
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
