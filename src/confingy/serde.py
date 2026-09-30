import dataclasses
import datetime
import decimal
import enum
import importlib
import inspect
import json
import logging
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from confingy.exceptions import (
    DeserializationError,
    SerializationError,
)
from confingy.tracking import Lazy, _create_tracked_instance
from confingy.utils.imports import get_class_name, get_module_name, import_qualname

logger = logging.getLogger(__name__)


def _get_valid_init_params(cls: type[Any]) -> set[str]:
    """Get the set of valid parameter names for a class's __init__ method.

    Returns all parameter names except 'self', including **kwargs if present.
    """
    sig = inspect.signature(cls.__init__)
    params = set()
    has_var_keyword = False

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            params.add(param_name)

    # If class accepts **kwargs, we can't determine invalid params
    if has_var_keyword:
        return None  # type: ignore

    return params


def _filter_extra_kwargs(
    cls: type[Any],
    init_args: dict[str, Any],
    strict: bool,
    context_name: str,
) -> dict[str, Any]:
    """Filter out extra kwargs that don't exist in the class signature.

    Args:
        cls: The class to check against
        init_args: The kwargs to filter
        strict: If True, raise error on extra kwargs; if False, warn and filter
        context_name: Name to use in error/warning messages (e.g., "MyClass")

    Returns:
        Filtered kwargs with extra keys removed

    Raises:
        DeserializationError: If strict=True and extra kwargs are found
    """
    valid_params = _get_valid_init_params(cls)

    # If class accepts **kwargs, all params are valid
    if valid_params is None:
        return init_args

    extra_keys = set(init_args.keys()) - valid_params
    if not extra_keys:
        return init_args

    extra_keys_str = ", ".join(sorted(extra_keys))
    message = (
        f"Serialized config for '{context_name}' contains kwargs that no longer "
        f"exist in the class signature: {extra_keys_str}. "
        f"These may have been removed from the class definition."
    )

    if strict:
        raise DeserializationError(message)
    else:
        warnings.warn(message, UserWarning)
        return {k: v for k, v in init_args.items() if k in valid_params}


class SerializationKeys:
    """Constants for serialization markers in the serialized data."""

    CLASS = "_confingy_class"
    MODULE = "_confingy_module"
    LAZY = "_confingy_lazy"
    CONFIG = "_confingy_config"
    INIT = "_confingy_init"
    DATACLASS = "_confingy_dataclass"
    FIELDS = "_confingy_fields"
    CALLABLE = "_confingy_callable"
    TRACKED_INFO = "_confingy_tracked_info"
    UNSERIALIZABLE = "_confingy_unserializable"
    OBJECT = "_confingy_object"
    METHOD = "_confingy_method"
    NAME = "_confingy_name"
    CLASS_HASH = "_confingy_class_hash"
    ENUM = "_confingy_enum"
    TUPLE = "_confingy_tuple"
    SET = "_confingy_set"
    DICT = "_confingy_dict"
    ITEMS = "_confingy_items"


class SerializationHandler(ABC):
    """Abstract base class for serialization handlers."""

    @abstractmethod
    def can_handle(self, obj: Any) -> bool:
        """Check if this handler can serialize the given object."""
        pass

    @abstractmethod
    def serialize(self, obj: Any, context: "SerializationContext") -> Any:
        """Serialize the object."""
        pass

    @abstractmethod
    def deserialize(self, data: Any, context: "DeserializationContext") -> Any:
        """Deserialize the data."""
        pass

    def map_children(self, obj: Any, fn: "Callable[[Any], Any]") -> Any:
        """Apply fn to all child values, returning new object with transformed children.

        This enables recursive traversal of nested structures for operations like
        lens/unlens without duplicating container-specific logic.

        Default returns obj unchanged (for leaf types like primitives).
        """
        return obj


class SerializationContext:
    """Context for serialization operations."""

    def __init__(self):
        self.handlers: list[SerializationHandler] = []
        self._depth = 0
        self._max_depth = 100
        self._path: list[str] = []  # Track object path for debugging

    def register_handler(self, handler: SerializationHandler):
        """Register a serialization handler."""
        self.handlers.append(handler)

    def serialize(self, obj: Any, key: str | None = None) -> Any:
        """Serialize an object using registered handlers."""
        if key:
            self._path.append(key)

        self._depth += 1
        try:
            if self._depth > self._max_depth:
                path_str = " -> ".join(self._path)
                raise SerializationError(
                    f"Maximum recursion depth ({self._max_depth}) exceeded at path: {path_str}"
                )

            for handler in self.handlers:
                if handler.can_handle(obj):
                    return handler.serialize(obj, self)

            # Raise an error for unserializable objects
            path_str = " -> ".join(self._path) if self._path else "root"
            raise SerializationError(
                f"No handler found for type {type(obj).__name__} at path {path_str}"
            )
        except Exception as e:
            if not isinstance(e, SerializationError):
                path_str = " -> ".join(self._path) if self._path else "root"
                raise SerializationError(
                    f"Failed to serialize at path {path_str}: {e}"
                ) from e
            raise
        finally:
            self._depth -= 1
            if key:
                self._path.pop()


class DeserializationContext:
    """Context for deserialization operations."""

    def __init__(self, strict: bool = True):
        """Initialize deserialization context.

        Args:
            strict: If True (default), raise an error when serialized configs contain
                kwargs that no longer exist in the class signature. If False, emit
                a warning and ignore the extra kwargs. Similar to PyTorch's
                load_state_dict strict parameter.
        """
        self.handlers: list[SerializationHandler] = []
        self._depth = 0
        self._max_depth = 100
        self.strict = strict

    def register_handler(self, handler: SerializationHandler):
        """Register a serialization handler."""
        self.handlers.append(handler)

    def deserialize(self, data: Any, expected_type: type | None = None) -> Any:
        """Deserialize data using registered handlers."""
        self._depth += 1
        try:
            if self._depth > self._max_depth:
                raise DeserializationError(
                    f"Maximum recursion depth ({self._max_depth}) exceeded"
                )

            # Primitives pass through
            if data is None or isinstance(data, int | float | str | bool):
                return data

            # Try all handlers (including collection handlers)
            for handler in self.handlers:
                result = handler.deserialize(data, self)
                if result is not None:
                    return result

            return data
        finally:
            self._depth -= 1


# Specific handlers for different object types


class PrimitiveHandler(SerializationHandler):
    """Handler for primitive types."""

    def can_handle(self, obj: Any) -> bool:
        return obj is None or isinstance(obj, int | float | str | bool)

    def serialize(self, obj: Any, context: SerializationContext) -> Any:
        return obj

    def deserialize(self, data: Any, context: DeserializationContext) -> Any:
        if self.can_handle(data):
            return data
        return None


class EnumHandler(SerializationHandler):
    """Handler for enum types."""

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, enum.Enum)

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        cls = type(obj)
        return {
            SerializationKeys.CLASS: get_class_name(cls),
            SerializationKeys.MODULE: get_module_name(cls),
            SerializationKeys.ENUM: True,
            SerializationKeys.NAME: obj.name,
        }

    def deserialize(self, data: Any, context: DeserializationContext) -> Any:
        if not isinstance(data, dict):
            return None

        if not data.get(SerializationKeys.ENUM):
            return None

        if SerializationKeys.CLASS not in data or SerializationKeys.MODULE not in data:
            return None

        try:
            cls = import_qualname(
                data[SerializationKeys.MODULE], data[SerializationKeys.CLASS]
            )
            return cls[data[SerializationKeys.NAME]]
        except (ImportError, AttributeError, KeyError) as e:
            raise DeserializationError(
                f"Could not recreate enum {data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}.{data.get(SerializationKeys.NAME)}: {e}"
            ) from None


class PathHandler(SerializationHandler):
    """Handler for pathlib.Path objects."""

    _PATH_CLASSES = (
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    )

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, Path)

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        return {
            SerializationKeys.CLASS: "Path",
            SerializationKeys.MODULE: "pathlib",
            SerializationKeys.NAME: str(obj),
        }

    def deserialize(self, data: Any, context: DeserializationContext) -> Any:
        if not isinstance(data, dict):
            return None
        if data.get(SerializationKeys.CLASS) not in self._PATH_CLASSES:
            return None
        if data.get(SerializationKeys.MODULE) != "pathlib":
            return None
        return Path(data.get(SerializationKeys.NAME, ""))


class DatetimeHandler(SerializationHandler):
    """Handler for ``datetime`` stdlib leaf types.

    Covers ``datetime.date``, ``datetime.datetime``, ``datetime.time``, and
    ``datetime.timedelta``. Serialized as a leaf using the same
    ``{_confingy_class, _confingy_module, _confingy_name}`` envelope as
    :class:`PathHandler`.

    Wire formats::

        date       -> {"_confingy_class": "date",      "_confingy_module": "datetime", "_confingy_name": "2024-01-01"}
        datetime   -> {"_confingy_class": "datetime",  "_confingy_module": "datetime", "_confingy_name": "2024-01-01T12:34:56+00:00"}
        time       -> {"_confingy_class": "time",      "_confingy_module": "datetime", "_confingy_name": "12:34:56"}
        timedelta  -> {"_confingy_class": "timedelta", "_confingy_module": "datetime", "_confingy_name": 93784.0}

    ``date`` / ``datetime`` / ``time`` use ``isoformat()`` / ``fromisoformat()``.
    Exact round-trip is guaranteed for values that ``isoformat()`` itself emits
    on the same Python version (the only values we ever feed back in).

    ``timedelta`` uses ``total_seconds()`` (float) for the wire value because the
    stdlib has no ``timedelta.fromisoformat``. A float can't hold microsecond
    resolution for very long durations (roughly 100,000 days and up), so those
    are written as an exact decimal string of seconds (e.g.
    ``"8640000000.000001"``) instead. Both forms are accepted when reading.

    Dispatch is by exact ``type(obj)`` rather than ``isinstance`` because
    ``datetime.datetime`` subclasses ``datetime.date`` — a naive isinstance
    check would mis-serialize datetimes as dates.
    """

    # Map of exact stdlib type -> (wire class name) used for both directions.
    _TYPES = {
        datetime.datetime: "datetime",
        datetime.date: "date",
        datetime.time: "time",
        datetime.timedelta: "timedelta",
    }

    def can_handle(self, obj: Any) -> bool:
        return type(obj) in self._TYPES

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        cls = type(obj)
        if cls is datetime.timedelta:
            name: Any = obj.total_seconds()
            if datetime.timedelta(seconds=name) != obj:
                name = _timedelta_to_decimal_string(obj)
        else:
            name = obj.isoformat()
        return {
            SerializationKeys.CLASS: self._TYPES[cls],
            SerializationKeys.MODULE: "datetime",
            SerializationKeys.NAME: name,
        }

    def deserialize(self, data: Any, context: DeserializationContext) -> Any:
        if not isinstance(data, dict):
            return None
        if data.get(SerializationKeys.MODULE) != "datetime":
            return None
        class_name = data.get(SerializationKeys.CLASS)
        name = data.get(SerializationKeys.NAME)
        if class_name in ("date", "datetime", "time"):
            if not isinstance(name, str):
                return None
            if class_name == "date":
                return datetime.date.fromisoformat(name)
            if class_name == "datetime":
                return datetime.datetime.fromisoformat(name)
            return datetime.time.fromisoformat(name)
        if class_name == "timedelta":
            if isinstance(name, bool) or not isinstance(name, (int, float, str)):
                return None
            if isinstance(name, str):
                try:
                    seconds = decimal.Decimal(name)
                except decimal.InvalidOperation:
                    return None
                microseconds = int((seconds * 1_000_000).to_integral_value())
                return datetime.timedelta(microseconds=microseconds)
            return datetime.timedelta(seconds=float(name))
        return None


def _timedelta_to_decimal_string(value: datetime.timedelta) -> str:
    """Format a timedelta as an exact decimal number of seconds."""
    microseconds = (
        value.days * 86_400 + value.seconds
    ) * 1_000_000 + value.microseconds
    sign = "-" if microseconds < 0 else ""
    whole, fraction = divmod(abs(microseconds), 1_000_000)
    return f"{sign}{whole}.{fraction:06d}"


class LazyHandler(SerializationHandler):
    """Handler for Lazy objects."""

    def can_handle(self, obj: Any) -> bool:
        return hasattr(obj, "_confingy_lazy_info")

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        result = {
            SerializationKeys.CLASS: obj._confingy_lazy_info["class"],
            SerializationKeys.MODULE: obj._confingy_lazy_info["module"],
            SerializationKeys.LAZY: True,
            SerializationKeys.CONFIG: {
                k: context.serialize(v, k) for k, v in obj._confingy_config.items()
            },
        }
        # Include class_hash if available
        if "class_hash" in obj._confingy_lazy_info:
            result[SerializationKeys.CLASS_HASH] = obj._confingy_lazy_info["class_hash"]
        return result

    def deserialize(self, data: dict[str, Any], context: DeserializationContext) -> Any:
        # Only handle dictionaries
        if not isinstance(data, dict):
            return None

        if not (
            data.get(SerializationKeys.LAZY)
            and SerializationKeys.CLASS in data
            and SerializationKeys.MODULE in data
        ):
            return None

        try:
            cls = import_qualname(
                data[SerializationKeys.MODULE], data[SerializationKeys.CLASS]
            )
            actual_cls = getattr(cls, "_original_cls", cls)

            config = {
                k: context.deserialize(v)
                for k, v in data.get(SerializationKeys.CONFIG, {}).items()
            }

            # Filter out extra kwargs that no longer exist in the class signature
            config = _filter_extra_kwargs(
                actual_cls,
                config,
                context.strict,
                f"{data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}",
            )

            return Lazy(
                actual_cls, config, skip_validation=True, _skip_post_config_hook=True
            )
        except (ImportError, AttributeError) as e:
            raise DeserializationError(
                f"Could not recreate {data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}: {e}"
            ) from None


class TrackedInstanceHandler(SerializationHandler):
    """Handler for tracked objects."""

    def can_handle(self, obj: Any) -> bool:
        return hasattr(obj, "_tracked_info")

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        info = obj._tracked_info
        result = {
            SerializationKeys.CLASS: info["class"],
            SerializationKeys.MODULE: info["module"],
            SerializationKeys.INIT: {
                k: context.serialize(v, k) for k, v in info["init_args"].items()
            },
        }
        # Include class_hash if available
        if "class_hash" in info:
            result[SerializationKeys.CLASS_HASH] = info["class_hash"]
        return result

    def deserialize(self, data: dict[str, Any], context: DeserializationContext) -> Any:
        # Only handle dictionaries
        if not isinstance(data, dict):
            return None

        if (
            SerializationKeys.INIT not in data
            or SerializationKeys.CLASS not in data
            or SerializationKeys.MODULE not in data
        ):
            return None

        try:
            cls = import_qualname(
                data[SerializationKeys.MODULE], data[SerializationKeys.CLASS]
            )
            actual_cls = getattr(cls, "_original_cls", cls)

            init_args = {
                k: context.deserialize(v)
                for k, v in data[SerializationKeys.INIT].items()
            }

            # Filter out extra kwargs that no longer exist in the class signature
            init_args = _filter_extra_kwargs(
                actual_cls,
                init_args,
                context.strict,
                f"{data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}",
            )

            # Use _create_tracked_instance to add tracking info to the instance
            # without globally modifying the class. This allows re-serialization
            # even if the class wasn't decorated with @track.
            return _create_tracked_instance(actual_cls, (), init_args, _validate=False)
        except (ImportError, AttributeError) as e:
            raise DeserializationError(
                f"Could not recreate {data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}: {e}"
            ) from None


class TypeHandler(SerializationHandler):
    """Handler for type objects."""

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, type)

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        # Check if the class can be imported
        if (
            hasattr(obj, "__module__")
            and hasattr(obj, "__qualname__")
            and (
                # Ensure it's not a local or lambda class
                "<locals>" not in obj.__qualname__
                and "<lambda>" not in obj.__qualname__
            )
        ):
            return {
                SerializationKeys.CLASS: "type",
                SerializationKeys.MODULE: get_module_name(obj),
                SerializationKeys.NAME: obj.__qualname__,
            }

        # Raise error for unserializable types
        raise SerializationError(
            f"Cannot serialize type {obj!r}: local or lambda types are not serializable. "
            f"Define the class at module level to make it serializable."
        )

    def deserialize(self, data: dict[str, Any], context: DeserializationContext) -> Any:
        # Only handle dictionaries
        if not isinstance(data, dict):
            return None

        # Check if this is a type serialization
        if data.get(SerializationKeys.CLASS) != "type":
            return None

        # Import and return the type
        if SerializationKeys.MODULE in data and SerializationKeys.NAME in data:
            try:
                module = importlib.import_module(data[SerializationKeys.MODULE])
                parts = data[SerializationKeys.NAME].split(".")
                obj = module
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except (ImportError, AttributeError) as e:
                raise DeserializationError(
                    f"Could not import type {data[SerializationKeys.MODULE]}.{data[SerializationKeys.NAME]}: {e}"
                ) from None

        return None


class CallableHandler(SerializationHandler):
    """Handler for callable objects."""

    def can_handle(self, obj: Any) -> bool:
        return callable(obj) and not isinstance(obj, type)

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        # Handle methods bound to an instance. The instance is serialized along
        # with the method name. Methods bound to a class (classmethods) or a
        # module (builtins) are importable by qualified name, like functions.
        bound_to = getattr(obj, "__self__", None)
        if bound_to is not None and not isinstance(bound_to, type | types.ModuleType):
            try:
                serialized_object = context.serialize(bound_to, "bound_object")
            except SerializationError as e:
                raise SerializationError(
                    f"Cannot serialize bound method {obj!r}: the object it is bound "
                    f"to is not serializable. Use @track on its class. ({e})"
                ) from None
            return {
                SerializationKeys.CALLABLE: "method",
                SerializationKeys.OBJECT: serialized_object,
                SerializationKeys.METHOD: obj.__name__,
            }

        # Handle standalone functions
        if (
            hasattr(obj, "__module__")
            and hasattr(obj, "__qualname__")
            and (
                "<locals>" not in obj.__qualname__
                and "<lambda>" not in obj.__qualname__
            )
        ):
            return {
                SerializationKeys.CALLABLE: "function",
                SerializationKeys.MODULE: get_module_name(obj),
                SerializationKeys.NAME: obj.__qualname__,
            }

        raise SerializationError(
            f"Cannot serialize callable {obj!r}: lambdas and local functions are not serializable. "
            f"Define the function at module level to make it serializable."
        )

    def deserialize(self, data: dict[str, Any], context: DeserializationContext) -> Any:
        # Only handle dictionaries
        if not isinstance(data, dict):
            return None

        if SerializationKeys.CALLABLE not in data:
            return None

        # Handle serializable callables
        if data[SerializationKeys.CALLABLE] == "function":
            try:
                module = importlib.import_module(data[SerializationKeys.MODULE])
                parts = data[SerializationKeys.NAME].split(".")
                obj = module
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except (ImportError, AttributeError) as e:
                raise DeserializationError(
                    f"Could not import function {data[SerializationKeys.MODULE]}.{data[SerializationKeys.NAME]}: {e}"
                ) from None

        elif data[SerializationKeys.CALLABLE] == "method":
            obj = context.deserialize(data[SerializationKeys.OBJECT])
            try:
                return getattr(obj, data[SerializationKeys.METHOD])
            except AttributeError as e:
                raise DeserializationError(
                    f"Could not recreate method {data[SerializationKeys.METHOD]}: {e}"
                ) from None

        else:
            raise DeserializationError(
                f"Unknown callable type: {data[SerializationKeys.CALLABLE]}"
            )


class CollectionHandler(SerializationHandler):
    """Handler for collections (lists, tuples, sets, dicts)."""

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, list | tuple | set | dict)

    def serialize(self, obj: Any, context: SerializationContext) -> Any:
        if isinstance(obj, tuple):
            return {
                SerializationKeys.TUPLE: True,
                SerializationKeys.ITEMS: [
                    context.serialize(item, f"[{i}]") for i, item in enumerate(obj)
                ],
            }
        elif isinstance(obj, set):
            items = [context.serialize(item, f"[{i}]") for i, item in enumerate(obj)]
            # Set iteration order depends on PYTHONHASHSEED, so sort for output
            # that is the same across processes
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {SerializationKeys.SET: True, SerializationKeys.ITEMS: items}
        elif isinstance(obj, list):
            return [context.serialize(item, f"[{i}]") for i, item in enumerate(obj)]
        elif isinstance(obj, dict):
            if all(isinstance(k, str) and not k.startswith("_confingy_") for k in obj):
                return {k: context.serialize(v, k) for k, v in obj.items()}
            # JSON object keys must be strings, and keys starting with
            # "_confingy_" would be read back as markers, so store other dicts as
            # a list of [key, value] pairs.
            return {
                SerializationKeys.DICT: True,
                SerializationKeys.ITEMS: [
                    [context.serialize(k, f"[{k!r}]"), context.serialize(v, f"[{k!r}]")]
                    for k, v in obj.items()
                ],
            }

    def deserialize(self, data: Any, context: DeserializationContext) -> Any:
        if isinstance(data, list):
            return [context.deserialize(item) for item in data]
        elif isinstance(data, dict):
            # Handle tuple wrapper
            if data.get(SerializationKeys.TUPLE) is True:
                return tuple(
                    context.deserialize(item)
                    for item in data.get(SerializationKeys.ITEMS, [])
                )
            # Handle set wrapper
            elif data.get(SerializationKeys.SET) is True:
                return {
                    context.deserialize(item)
                    for item in data.get(SerializationKeys.ITEMS, [])
                }
            # Handle dicts stored as [key, value] pairs
            elif data.get(SerializationKeys.DICT) is True:
                return {
                    context.deserialize(k): context.deserialize(v)
                    for k, v in data.get(SerializationKeys.ITEMS, [])
                }
            # Only handle regular dicts, not confingy special objects
            elif not any(k.startswith("_confingy_") for k in data):
                return {k: context.deserialize(v) for k, v in data.items()}
        return None

    def map_children(self, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, list):
            return [fn(v) for v in obj]
        # Check for namedtuple before tuple (namedtuple is a subclass of tuple)
        if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
            return type(obj)(*[fn(v) for v in obj])
        if isinstance(obj, tuple):
            return tuple(fn(v) for v in obj)
        if isinstance(obj, set):
            return {fn(v) for v in obj}
        if isinstance(obj, dict):
            return {k: fn(v) for k, v in obj.items()}
        return obj


class DataclassHandler(SerializationHandler):
    """Handler for dataclass objects."""

    def can_handle(self, obj: Any) -> bool:
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    def serialize(self, obj: Any, context: SerializationContext) -> dict[str, Any]:
        cls = obj.__class__
        return {
            SerializationKeys.CLASS: get_class_name(cls),
            SerializationKeys.MODULE: get_module_name(cls),
            SerializationKeys.DATACLASS: True,
            # init=False fields are derived (e.g. in __post_init__) and can't be
            # passed to the constructor, so they aren't part of the config
            SerializationKeys.FIELDS: {
                field.name: context.serialize(getattr(obj, field.name), field.name)
                for field in dataclasses.fields(obj)
                if field.init
            },
        }

    def deserialize(self, data: dict[str, Any], context: DeserializationContext) -> Any:
        # Only handle dictionaries
        if not isinstance(data, dict):
            return None

        if not (
            data.get(SerializationKeys.DATACLASS)
            and SerializationKeys.FIELDS in data
            and SerializationKeys.CLASS in data
            and SerializationKeys.MODULE in data
        ):
            return None

        try:
            cls = import_qualname(
                data[SerializationKeys.MODULE], data[SerializationKeys.CLASS]
            )

            # Deserialize field values. init=False fields, which older versions
            # serialized, are skipped since the constructor doesn't accept them.
            non_init = {f.name for f in dataclasses.fields(cls) if not f.init}
            field_values = {
                k: context.deserialize(v)
                for k, v in data[SerializationKeys.FIELDS].items()
                if k not in non_init
            }

            # Create the dataclass instance
            return cls(**field_values)
        except (ImportError, AttributeError, TypeError) as e:
            raise DeserializationError(
                f"Could not recreate dataclass {data[SerializationKeys.MODULE]}.{data[SerializationKeys.CLASS]}: {e}"
            ) from None

    def map_children(self, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
            return obj
        changes = {}
        for field in dataclasses.fields(obj):
            # Skip init=False fields - dataclasses.replace() can't handle them
            if not field.init:
                continue
            old_value = getattr(obj, field.name)
            new_value = fn(old_value)
            if new_value is not old_value:
                changes[field.name] = new_value
        if changes:
            return dataclasses.replace(obj, **changes)
        return obj


class HandlerRegistry:
    """Central registry for serialization and deserialization handlers."""

    @staticmethod
    def get_default_handlers():
        """Get the default set of handlers in the correct order."""
        return [
            PathHandler(),  # Must be before PrimitiveHandler (Path is not a primitive)
            EnumHandler(),  # Must be before PrimitiveHandler (StrEnum/IntEnum are str/int)
            DatetimeHandler(),  # Must be before PrimitiveHandler (datetime types are leaves, not primitives)
            PrimitiveHandler(),
            LazyHandler(),
            TrackedInstanceHandler(),
            DataclassHandler(),
            TypeHandler(),  # Handle type objects before callables
            CallableHandler(),
            CollectionHandler(),
        ]
