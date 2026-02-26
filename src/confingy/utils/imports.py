"""Import utility functions for confingy."""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any


def get_module_name(obj: Any) -> str:
    """Get the actual module name for an object, resolving __main__ to the real module name.

    When a Python script is run directly (e.g., `python script.py`), classes and functions
    defined in that script have __module__ == '__main__'. This function attempts to resolve
    the actual module name by inspecting the file path.
    """
    module_name = getattr(obj, "__module__", None)

    if module_name is not None and module_name != "__main__":
        return module_name

    # Try to get the file where the object was defined
    try:
        # For classes and functions, use them directly; for instances, use their class
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
            source_file = inspect.getfile(obj)
        else:
            source_file = inspect.getfile(obj.__class__)
    except (TypeError, OSError):
        # If we can't get the file, fall back to __main__
        return "__main__"

    # Try to convert the file path to a module name
    source_path = Path(source_file).resolve()

    # Look for the file in sys.path to determine the module name
    for path_str in sys.path:
        if not path_str:
            continue
        path = Path(path_str).resolve()
        try:
            relative = source_path.relative_to(path)
            # Convert path to module name (remove .py and replace / with .)
            if relative.suffix == ".py":
                module_parts = relative.with_suffix("").parts
                # Skip if it would result in an invalid module name
                if all(part.isidentifier() for part in module_parts):
                    return ".".join(module_parts)
        except ValueError:
            # source_path is not relative to this path
            continue

    # If we couldn't determine the module, try using just the filename without extension
    stem = source_path.stem
    if stem.isidentifier() and stem != "__main__":
        return stem

    # Fall back to __main__ if we can't determine the actual module
    return "__main__"


def derive_module_name(file_path: Path) -> str:
    """
    Derive the module name from a file path relative to the current working directory.

    The current working directory is assumed to be the root of the module structure.
    For example, if cwd is /root/project and file is /root/project/foo/bar/baz.py,
    the module name will be foo.bar.baz.

    Args:
        file_path: Path to the Python file

    Returns:
        The derived module name (e.g., "training.fingys.apollo.my_module")
    """
    file_path = file_path.resolve()
    cwd = Path.cwd().resolve()

    try:
        # Try to make the file path relative to the current working directory
        relative_path = file_path.relative_to(cwd)
    except ValueError:
        # File is outside the current working directory, just use the filename
        return file_path.stem

    # Convert path to module name: remove .py extension and replace / with .
    parts = relative_path.with_suffix("").parts
    if len(parts) > 1 and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_file_path(path: str) -> bool:
    """Return True if path looks like a file path rather than a dotted module path."""
    return "/" in path or path.endswith(".py")


def _resolve_module_path(module_path: str) -> Path:
    """Convert a dotted module path to a file path.

    Tries:
      1. path/to/module.py
      2. path/to/module/__init__.py

    Raises:
        FileNotFoundError: If neither file exists.
    """
    parts = module_path.split(".")
    base = Path(*parts)

    py_file = base.with_suffix(".py")
    if py_file.exists():
        return py_file

    init_file = base / "__init__.py"
    if init_file.exists():
        return init_file

    raise FileNotFoundError(
        f"Cannot resolve module path '{module_path}': tried {py_file} and {init_file}"
    )


def load_variable_from_file(file_spec: str, default_name: str = "config") -> Any:
    """
    Load a variable from a Python file or dotted module path.

    Args:
        file_spec: File path or dotted module path with optional variable name.
                  Supported formats:
                    - "path/to/file.py" (file path)
                    - "path/to/file.py::variable_name" (file path with variable)
                    - "path.to.module" (dotted module path)
                    - "path.to.module::variable_name" (dotted module path with variable)
                  If variable name is not specified, defaults to default_name.
        default_name: Default variable name to use if not specified in file_spec

    Returns:
        The variable loaded from the file

    Raises:
        ValueError: If the file_spec format is invalid or variable is not found
        FileNotFoundError: If the file doesn't exist
    """
    if "::" in file_spec:
        path_part, variable_name = file_spec.split("::", 1)
    else:
        path_part = file_spec
        variable_name = default_name

    if _is_file_path(path_part):
        file_path_obj = Path(path_part)
    else:
        file_path_obj = _resolve_module_path(path_part)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path_obj}")

    # Derive the proper module name from the file path
    module_name = derive_module_name(file_path_obj)
    spec = importlib.util.spec_from_file_location(module_name, file_path_obj)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {file_path_obj}")

    module = importlib.util.module_from_spec(spec)

    # Add the file's directory to sys.path temporarily to allow relative imports
    file_dir = str(file_path_obj.parent.resolve())
    sys.path.insert(0, file_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(file_dir)

    if not hasattr(module, variable_name):
        raise ValueError(
            f"Variable '{variable_name}' not found in {file_path_obj}. "
            f"Available names: {', '.join(name for name in dir(module) if not name.startswith('_'))}"
        )

    return getattr(module, variable_name)
