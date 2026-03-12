"""
Functions for transforming fingys, which are any python object that can be serialized/deserialized by `confingy`.
This includes any class wrapped with [confingy.track][] or [confingy.lazy][], as well as dataclasses and built-in types.
"""

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from confingy.serde import (
    DeserializationContext,
    HandlerRegistry,
    SerializationContext,
    SerializationKeys,
)

logger = logging.getLogger(__name__)


def serialize_fingy(fingy: Any) -> dict[str, Any]:
    """
    Serialize a fingy to a JSON-compatible dictionary.

    Args:
        fingy: The object to serialize

    Returns:
        A dictionary that can be saved to JSON
    """
    context = SerializationContext()

    # Register handlers using the central registry
    for handler in HandlerRegistry.get_default_handlers():
        context.register_handler(handler)

    result = context.serialize(fingy)
    return result if result is not None else {}


def deserialize_fingy(
    data: dict[str, Any],
    expected_type: Optional[type] = None,
    strict: bool = True,
) -> Any:
    """
    Deserialize a dictionary back to a fingy.

    Args:
        data: The dictionary to deserialize
        expected_type: Optional type hint for the expected result
        strict: If True (default), raise an error when serialized configs contain
            kwargs that no longer exist in the class signature. If False, emit
            a warning and ignore the extra kwargs. Similar to PyTorch's
            load_state_dict strict parameter.

    Returns:
        The reconstructed object
    """
    context = DeserializationContext(strict=strict)

    # Register all handlers using the central registry
    for handler in HandlerRegistry.get_default_handlers():
        context.register_handler(handler)

    return context.deserialize(data, expected_type)


def save_fingy(fingy: Any, filename: str) -> None:
    """
    Save a fingy to a JSON file.

    Args:
        fingy: The fingy object to save
        filename: Path to the output file
    """
    logger.debug(f"Saving fingy to {filename}")
    serialized = serialize_fingy(fingy)

    with open(filename, "w") as f:
        json.dump(serialized, f, indent=2, sort_keys=True)


def load_fingy(filename: str, strict: bool = True) -> Any:
    """
    Load and deserialize a fingy from a JSON file.

    Args:
        filename: Path to the JSON file
        strict: If True (default), raise an error when serialized configs contain
            kwargs that no longer exist in the class signature. If False, emit
            a warning and ignore the extra kwargs.

    Returns:
        The deserialized fingy object (may contain lazy instances)
    """
    logger.debug(f"Loading fingy from {filename}")

    with open(filename) as f:
        data = json.load(f)

    return deserialize_fingy(data, strict=strict)


def prettify_fingy(fingy: Any) -> dict[str, Any]:
    """
    Prettify a fingy object.

    This function serializes a fingy object and then transforms it into a more
    readable format by collapsing `_confingy_*` metadata fields into a simple string
    representation of the form `{module}.{class}` and preserving the actual field values.

    Args:
        fingy: The fingy object to prettify

    Returns:
        A dictionary with collapsed metadata that's easier to read

    Examples:
        Input fingy object that serializes to:
        ```json
        {
            "_confingy_class": "TrainingConfig",
            "_confingy_module": "__main__",
            "_confingy_dataclass": true,
            "_confingy_fields": {
                "dataset": {
                    "_confingy_class": "MyDataset",
                    "_confingy_module": "__main__",
                    "_confingy_init": {
                        "size": 100
                    }
                }
            }
        }
        ```

        Output:
        ```json
        {
            "__main__.TrainingConfig": {
                "dataset": {
                    "__main__.MyDataset": {
                        "size": 100
                    }
                }
            }
        }
        ```
    """
    serialized = serialize_fingy(fingy)
    return prettify_serialized_fingy(serialized)


def prettify_serialized_fingy(data: Any) -> Any:
    """
    Transform confingy serialized data into a prettier format.

    This function recursively processes serialized confingy data, collapsing metadata
    fields into readable string representations while preserving the actual content.
    See [prettify_fingy][confingy.fingy.prettify_fingy] for examples.

    Args:
        data: The serialized data to transform

    Returns:
        Transformed data with collapsed metadata
    """
    # Base cases
    if data is None or isinstance(data, int | float | str | bool):
        return data

    # Handle lists and tuples
    if isinstance(data, list | tuple):
        result = [prettify_serialized_fingy(item) for item in data]
        return tuple(result) if isinstance(data, tuple) else result

    # Handle dictionaries
    if isinstance(data, dict):
        # Handle pathlib.Path objects
        if data.get(SerializationKeys.MODULE) == "pathlib" and str(
            data.get(SerializationKeys.CLASS, "")
        ).endswith("Path"):
            return data.get(SerializationKeys.NAME, "")

        # Check if this dictionary represents a confingy object
        if SerializationKeys.CLASS in data and SerializationKeys.MODULE in data:
            # This is a confingy object - collapse it
            class_name = data.get(SerializationKeys.CLASS)
            module_name = data.get(SerializationKeys.MODULE)

            # Special handling for type objects
            if class_name == "type" and SerializationKeys.NAME in data:
                return f"{module_name}.{data[SerializationKeys.NAME]}"

            # Special handling for enum members
            if data.get(SerializationKeys.ENUM):
                return f"{module_name}.{class_name}.{data[SerializationKeys.NAME]}"

            key = f"{module_name}.{class_name}"

            # Determine what the nested content should be
            if SerializationKeys.FIELDS in data:
                # Dataclass - use fields
                nested_content = prettify_serialized_fingy(
                    data[SerializationKeys.FIELDS]
                )
            elif SerializationKeys.INIT in data:
                # Tracked class - use init args
                nested_content = prettify_serialized_fingy(data[SerializationKeys.INIT])
            elif SerializationKeys.CONFIG in data:
                # Lazy object - use config
                nested_content = prettify_serialized_fingy(
                    data[SerializationKeys.CONFIG]
                )
            elif SerializationKeys.NAME in data:
                # Type or callable with qualified name
                return f"{module_name}.{data[SerializationKeys.NAME]}"
            elif SerializationKeys.UNSERIALIZABLE in data:
                # Unserializable object
                return data[SerializationKeys.UNSERIALIZABLE]
            elif data.get(SerializationKeys.CALLABLE) == "method":
                # Bound method
                obj = prettify_serialized_fingy(data.get(SerializationKeys.OBJECT, {}))
                method = data.get(SerializationKeys.METHOD, "unknown")
                if isinstance(obj, dict) and len(obj) == 1:
                    # If obj is a collapsed dict with single key, extract it
                    obj_key = next(iter(obj.keys()))
                    return f"{obj_key}.{method}"
                return f"{obj}.{method}"
            elif data.get(SerializationKeys.CALLABLE) == "function":
                # Function
                return f"{module_name}.{data.get(SerializationKeys.NAME, 'unknown')}"
            else:
                # Unknown structure, try to extract any meaningful content
                nested_content = {
                    k: prettify_serialized_fingy(v)
                    for k, v in data.items()
                    if not k.startswith("_confingy_")
                }
                if not nested_content:
                    # No non-metadata content
                    return key

            # Return the collapsed representation
            if isinstance(nested_content, dict) and nested_content or nested_content:
                return {key: nested_content}
            else:
                return key
        else:
            # Regular dictionary - recursively process values
            return {k: prettify_serialized_fingy(v) for k, v in data.items()}

    # For any other type, return as-is
    return data


class _ConfingyTranspiler:
    """Transpiles serialized fingys back into Python code."""

    def __init__(self):
        self.imports: set[tuple[str, str]] = set()
        self.lazy_imports_needed = False
        self.dataclass_needed = False
        self.track_needed = False
        self.indent_level = 0
        self.indent_str = "    "

    def transpile(self, fingy_data: dict[str, Any] | str | Path) -> str:
        """
        Transpile a serialized confingy fingy to Python code.

        Args:
            fingy_data: Either a dictionary of serialized fingy data,
                        a JSON string, or a path to a JSON file

        Returns:
            Python code as a string
        """
        # Load fingy data if needed
        if isinstance(fingy_data, (str, Path)):
            if isinstance(fingy_data, str) and (
                fingy_data.startswith("{") or fingy_data.startswith("[")
            ):
                # It's a JSON string
                data = json.loads(fingy_data)
            else:
                # It's a file path
                with open(fingy_data) as f:
                    data = json.load(f)
        else:
            data = fingy_data

        # Reset state
        self.imports.clear()
        self.lazy_imports_needed = False
        self.dataclass_needed = False
        self.track_needed = False

        # Generate the main fingy
        fingy_code = self._transpile_value(data, "config")

        # Build the final code
        code_parts = []

        # Add imports
        import_lines = self._generate_imports()
        if import_lines:
            code_parts.append(import_lines)
            code_parts.append("")

        # Add the config instantiation
        code_parts.append(f"config = {fingy_code}")

        return "\n".join(code_parts)

    def _generate_imports(self) -> str:
        """Generate import statements."""
        import_lines = []

        # Add confingy imports if needed
        confingy_imports = []
        if self.lazy_imports_needed:
            confingy_imports.append("lazy")
        if self.track_needed:
            confingy_imports.append("track")

        if confingy_imports:
            import_lines.append(f"from confingy import {', '.join(confingy_imports)}")

        # Add dataclass import if needed
        if self.dataclass_needed:
            import_lines.append("from dataclasses import dataclass")

        # Group imports by module
        imports_by_module: dict[str, list[str]] = {}
        for module, name in sorted(self.imports):
            if module not in imports_by_module:
                imports_by_module[module] = []
            imports_by_module[module].append(name)

        # Generate import statements
        for module, names in imports_by_module.items():
            if len(names) == 1:
                import_lines.append(f"from {module} import {names[0]}")
            else:
                import_lines.append(f"from {module} import {', '.join(sorted(names))}")

        return "\n".join(import_lines)

    def _clean_expr(self, expr: str) -> str:
        """Clean an expression."""
        bgn = expr.find("fingy.var(")
        while bgn != -1:
            end = expr.find(")", bgn)
            assert end != -1, f"Invalid expression: {expr}"
            expr = expr[0:bgn] + expr[bgn + len("fingy.var(") : end] + expr[end + 1 :]
            bgn = expr.find("fingy.var(")
        return expr

    def _transpile_value(self, value: Any, var_name: str | None = None) -> str:
        """Transpile a single value."""
        if value is None:
            return "None"

        if isinstance(value, bool):
            return str(value)

        if isinstance(value, (int, float)):
            return repr(value)

        if isinstance(value, str):
            if value.startswith("fingy.var(") and value.endswith(")"):
                return value[len("fingy.var(") : -1]
            if value.startswith("fingy.eval(") and value.endswith(")"):
                expr = value[len("fingy.eval(") : -1]
                return self._clean_expr(expr)
            return repr(value)

        if isinstance(value, list):
            return self._transpile_list(value)

        if isinstance(value, dict):
            # Check if it's a tuple
            if value.get(SerializationKeys.TUPLE) is True:
                return self._transpile_tuple(value.get(SerializationKeys.ITEMS, []))
            # Check if it's a set
            if value.get(SerializationKeys.SET) is True:
                return self._transpile_set(value.get(SerializationKeys.ITEMS, []))
            # Check if it's a confingy object
            if SerializationKeys.CLASS in value:
                return self._transpile_confingy_object(value)
            else:
                return self._transpile_dict(value)

        # Fallback
        return repr(value)

    def _transpile_list(self, lst: list) -> str:
        """Transpile a list."""
        if not lst:
            return "[]"

        # Check if items are simple enough for single line
        items = [self._transpile_value(item) for item in lst]
        if all(len(item) < 40 and "\n" not in item for item in items):
            return f"[{', '.join(items)}]"

        # Multi-line format
        self.indent_level += 1
        indent = self.indent_str * self.indent_level
        formatted_items = [f"{indent}{item}" for item in items]
        self.indent_level -= 1
        return (
            "[\n"
            + ",\n".join(formatted_items)
            + "\n"
            + self.indent_str * self.indent_level
            + "]"
        )

    def _transpile_tuple(self, items: list) -> str:
        """Transpile a tuple."""
        if not items:
            return "()"

        # Check if items are simple enough for single line
        transpiled = [self._transpile_value(item) for item in items]
        if all(len(item) < 40 and "\n" not in item for item in transpiled):
            # Single element tuples need trailing comma
            if len(transpiled) == 1:
                return f"({transpiled[0]},)"
            return f"({', '.join(transpiled)})"

        # Multi-line format
        self.indent_level += 1
        indent = self.indent_str * self.indent_level
        formatted_items = [f"{indent}{item}" for item in transpiled]
        self.indent_level -= 1
        # Single element tuples need trailing comma
        trailing = "," if len(transpiled) == 1 else ""
        return (
            "(\n"
            + ",\n".join(formatted_items)
            + trailing
            + "\n"
            + self.indent_str * self.indent_level
            + ")"
        )

    def _transpile_set(self, items: list) -> str:
        """Transpile a set."""
        if not items:
            return "set()"

        # Check if items are simple enough for single line
        transpiled = [self._transpile_value(item) for item in items]
        if all(len(item) < 40 and "\n" not in item for item in transpiled):
            return "{" + ", ".join(transpiled) + "}"

        # Multi-line format
        self.indent_level += 1
        indent = self.indent_str * self.indent_level
        formatted_items = [f"{indent}{item}" for item in transpiled]
        self.indent_level -= 1
        return (
            "{\n"
            + ",\n".join(formatted_items)
            + "\n"
            + self.indent_str * self.indent_level
            + "}"
        )

    def _transpile_dict(self, dct: dict) -> str:
        """Transpile a regular dictionary."""
        if not dct:
            return "{}"

        items = []
        for key, value in dct.items():
            key_repr = repr(key) if not key.isidentifier() else f'"{key}"'
            value_repr = self._transpile_value(value)
            items.append(f"{key_repr}: {value_repr}")

        # Check if simple enough for single line
        if all(len(item) < 40 and "\n" not in item for item in items):
            return "{" + ", ".join(items) + "}"

        # Multi-line format
        self.indent_level += 1
        indent = self.indent_str * self.indent_level
        formatted_items = [f"{indent}{item}" for item in items]
        self.indent_level -= 1
        return (
            "{\n"
            + ",\n".join(formatted_items)
            + "\n"
            + self.indent_str * self.indent_level
            + "}"
        )

    def _transpile_confingy_object(self, obj: dict[str, Any]) -> str:
        """Transpile a confingy serialized object."""
        class_name = obj.get(SerializationKeys.CLASS, "Unknown")
        module_name = obj.get(SerializationKeys.MODULE, "unknown")

        # Handle enum members early (no constructor call needed)
        if obj.get(SerializationKeys.ENUM):
            member_name = obj.get(SerializationKeys.NAME, "unknown")
            self.imports.add((module_name, class_name))
            return f"{class_name}.{member_name}"

        # Add to imports
        self.imports.add((module_name, class_name))

        # Handle different confingy object types
        if obj.get(SerializationKeys.DATACLASS):
            # Dataclass
            return self._transpile_dataclass(obj, class_name)

        elif SerializationKeys.LAZY in obj:
            # Lazy object
            self.lazy_imports_needed = True
            config = obj.get(SerializationKeys.CONFIG, {})
            return self._transpile_lazy(class_name, config)

        elif SerializationKeys.INIT in obj:
            # Tracked class
            init_args = obj.get(SerializationKeys.INIT, {})
            return self._transpile_tracked_class(class_name, init_args)

        elif obj.get(SerializationKeys.CALLABLE) == "function":
            # Function
            func_name = obj.get(SerializationKeys.NAME, "unknown")
            self.imports.add((module_name, func_name))
            return func_name

        elif obj.get(SerializationKeys.CALLABLE) == "method":
            # Bound method
            bound_obj = self._transpile_value(obj.get(SerializationKeys.OBJECT, {}))
            method_name = obj.get(SerializationKeys.METHOD, "unknown")
            return f"{bound_obj}.{method_name}"

        elif obj.get(SerializationKeys.CLASS) == "type":
            # Type object
            type_name = obj.get(SerializationKeys.NAME, "Unknown")
            self.imports.add((module_name, type_name))
            return type_name

        elif module_name == "pathlib":
            # pathlib.Path object
            path_str = obj.get(SerializationKeys.NAME, "")
            self.imports.add(("pathlib", "Path"))
            return f"Path({path_str!r})"

        elif SerializationKeys.UNSERIALIZABLE in obj:
            # Unserializable object - add as comment
            return f"# Unserializable: {obj[SerializationKeys.UNSERIALIZABLE]}"

        else:
            # Unknown structure
            return f"# Unknown confingy object: {class_name}"

    def _transpile_dataclass(self, obj: dict[str, Any], class_name: str) -> str:
        """Transpile a dataclass."""
        fields_dict = obj.get(SerializationKeys.FIELDS, {})

        if not fields_dict:
            return f"{class_name}()"

        # Process fields
        field_args = []
        for field_name, field_value in fields_dict.items():
            value_repr = self._transpile_value(field_value)

            # Format field assignment
            if "\n" in value_repr or len(value_repr) > 50:
                # Multi-line value
                self.indent_level += 1
                indent = self.indent_str * self.indent_level
                field_args.append(f"{field_name}={value_repr}")
                self.indent_level -= 1
            else:
                field_args.append(f"{field_name}={value_repr}")

        # Format constructor call
        if len(field_args) == 1 and len(field_args[0]) < 60:
            # Single short argument
            return f"{class_name}({field_args[0]})"
        else:
            # Multi-line format
            self.indent_level += 1
            indent = self.indent_str * self.indent_level
            formatted_args = [f"{indent}{arg}" for arg in field_args]
            self.indent_level -= 1
            return (
                f"{class_name}(\n"
                + ",\n".join(formatted_args)
                + "\n"
                + self.indent_str * self.indent_level
                + ")"
            )

    def _transpile_lazy(self, class_name: str, config: dict[str, Any]) -> str:
        """Transpile a lazy object."""
        if not config:
            return f"lazy({class_name})()"

        # Process config arguments
        arg_parts = []
        for key, value in config.items():
            value_repr = self._transpile_value(value)
            arg_parts.append(f"{key}={value_repr}")

        # Format lazy call
        if len(arg_parts) == 1 and len(arg_parts[0]) < 50:
            return f"lazy({class_name})({arg_parts[0]})"
        else:
            # Multi-line format
            self.indent_level += 1
            indent = self.indent_str * self.indent_level
            formatted_args = [f"{indent}{arg}" for arg in arg_parts]
            self.indent_level -= 1
            args_str = ",\n".join(formatted_args)
            return (
                f"lazy({class_name})(\n{args_str}\n"
                + self.indent_str * self.indent_level
                + ")"
            )

    def _transpile_tracked_class(
        self, class_name: str, init_args: dict[str, Any]
    ) -> str:
        """Transpile a tracked class."""
        if not init_args:
            return f"{class_name}()"

        # Process init arguments
        arg_parts = []
        for key, value in init_args.items():
            value_repr = self._transpile_value(value)
            arg_parts.append(f"{key}={value_repr}")

        # Format constructor call
        if len(arg_parts) == 1 and len(arg_parts[0]) < 60:
            return f"{class_name}({arg_parts[0]})"
        else:
            # Multi-line format
            self.indent_level += 1
            indent = self.indent_str * self.indent_level
            formatted_args = [f"{indent}{arg}" for arg in arg_parts]
            self.indent_level -= 1
            return (
                f"{class_name}(\n"
                + ",\n".join(formatted_args)
                + "\n"
                + self.indent_str * self.indent_level
                + ")"
            )


def transpile_fingy(fingy_data: dict[str, Any] | str | Path) -> str:
    """
    Transpile a serialized fingy back into Python code.

    Args:
        fingy_data: Either a dictionary of serialized confingy fingy data,
                    a JSON string, or a path to a JSON file

    Returns:
        Python code as a string

    Examples:
        ```python
        from confingy import serialize_fingy, transpile_fingy
        config = MyConfig(...)
        serialized = serialize_fingy(config)
        python_code = transpile_fingy(serialized)
        print(python_code)
        ```
    """
    transpiler = _ConfingyTranspiler()
    return transpiler.transpile(fingy_data)
