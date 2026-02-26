import hashlib
import inspect
from typing import Any


def hash_class(cls: type, algorithm: str = "sha256") -> str:
    """
    Create a hash of a class based on its bytecode.
    Only changes that affect code execution will change the hash.

    Args:
        cls: The class to hash
        algorithm: The hashing algorithm to use (default: 'sha256')

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)

    # Get all components that affect execution
    components = _get_class_bytecode_components(cls)

    # Sort components by name for deterministic hashing
    components.sort(key=lambda x: x[0])

    # Hash each component
    for name, data in components:
        hasher.update(name.encode("utf-8"))
        hasher.update(b":::")  # Separator
        hasher.update(data)
        hasher.update(b"|||")  # Component separator

    return hasher.hexdigest()


def _get_class_bytecode_components(cls: type) -> list[tuple[str, bytes]]:
    """Extract all bytecode components from a class."""
    components = []

    # Add class name and bases
    components.append(("__classname__", cls.__name__.encode("utf-8")))
    bases_str = ",".join(base.__name__ for base in cls.__bases__ if base is not object)
    if bases_str:
        components.append(("__bases__", bases_str.encode("utf-8")))

    # Process all class members
    for name, member in inspect.getmembers(cls):
        # Skip special attributes and non-code objects
        if (
            name.startswith("__")
            and name.endswith("__")
            and name not in ["__init__", "__call__", "__new__"]
        ):
            continue

        # Handle methods and functions
        if inspect.ismethod(member) or inspect.isfunction(member):
            bytecode_data = _extract_code_bytes(member)
            if bytecode_data:
                components.append((f"method:{name}", bytecode_data))

        # Handle properties
        elif isinstance(member, property):
            for prop_name, prop_func in [
                ("getter", member.fget),
                ("setter", member.fset),
                ("deleter", member.fdel),
            ]:
                if prop_func:
                    bytecode_data = _extract_code_bytes(prop_func)
                    if bytecode_data:
                        components.append(
                            (f"property:{name}:{prop_name}", bytecode_data)
                        )

        # Handle static/class methods
        elif isinstance(member, (staticmethod, classmethod)):
            bytecode_data = _extract_code_bytes(member.__func__)
            if bytecode_data:
                method_type = (
                    "staticmethod"
                    if isinstance(member, staticmethod)
                    else "classmethod"
                )
                components.append((f"{method_type}:{name}", bytecode_data))

        # Handle class variables (that aren't methods or special attributes)
        elif not callable(member) and not name.startswith("_"):
            # For class variables, we include their value if it's hashable
            try:
                value_bytes = repr(member).encode("utf-8")
                components.append((f"classvar:{name}", value_bytes))
            except Exception:
                pass

    return components


def _extract_code_bytes(func: Any) -> bytes:
    """Extract bytecode and relevant metadata from a function/method."""
    try:
        if hasattr(func, "__code__"):
            code = func.__code__
        elif hasattr(func, "func_code"):  # Python 2 compatibility
            code = func.func_code
        else:
            return b""

        # Combine all code attributes that affect execution
        parts = []

        # The actual bytecode
        parts.append(code.co_code)

        # Constants - properly detect and exclude docstrings
        # Docstrings are stored as the first constant only if:
        # 1. The function has a docstring
        # 2. The first constant is a string
        # 3. The string is not referenced in the bytecode (it's just metadata)
        #
        # Additionally, functions without docstrings often have None as co_consts[0]
        consts = list(code.co_consts)
        if consts:
            # Check if first constant is the docstring
            if (
                hasattr(func, "__doc__")
                and func.__doc__ is not None
                and consts[0] == func.__doc__
            ):
                # This is the actual docstring, exclude it
                consts = consts[1:]
            # Check if first constant is None and function has no docstring
            elif consts[0] is None and (
                not hasattr(func, "__doc__") or func.__doc__ is None
            ):
                # This None is just a placeholder for missing docstring, exclude it
                consts = consts[1:]

        parts.append(repr(tuple(consts)).encode("utf-8"))

        # Names used in the code
        parts.append(repr(code.co_names).encode("utf-8"))

        # Argument specification
        parts.append(str(code.co_argcount).encode("utf-8"))
        if hasattr(code, "co_kwonlyargcount"):  # Python 3+
            parts.append(str(code.co_kwonlyargcount).encode("utf-8"))
        if hasattr(code, "co_posonlyargcount"):  # Python 3.8+
            parts.append(str(code.co_posonlyargcount).encode("utf-8"))

        # Variable names (parameters and local variables)
        parts.append(repr(code.co_varnames).encode("utf-8"))

        # Flags that affect execution
        parts.append(str(code.co_flags).encode("utf-8"))

        # Combine all parts
        return b"|".join(parts)
    except Exception:
        return b""
