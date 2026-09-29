"""Helpers for traversing the containers that can hold tracked or lazy objects."""

import copy
import dataclasses
from typing import Any


def container_children(value: Any) -> list[tuple[str, Any]] | None:
    """Return `(path_suffix, child)` pairs for a supported container.

    Supported containers are lists, tuples (including namedtuples), dicts, sets,
    frozensets, and dataclass instances. Path suffixes look like `[0]` for
    sequences, `['key']` for dicts, `.field` for dataclasses, and `{}` for set
    members, which have no stable position.

    Args:
        value: Any value.

    Returns:
        The container's children in iteration order, or None if `value` is not a
        supported container.
    """
    if isinstance(value, (list, tuple)):
        return [(f"[{i}]", v) for i, v in enumerate(value)]
    if isinstance(value, dict):
        return [(f"[{k!r}]", v) for k, v in value.items()]
    if isinstance(value, (set, frozenset)):
        return [("{}", v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # init=False fields are derived, and dataclasses.replace() can't set them
        return [
            (f".{f.name}", getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.init
        ]
    return None


def rebuild_container(value: Any, new_children: list[Any]) -> Any:
    """Return a copy of a container with its children replaced.

    The container type is preserved where possible, including list and dict
    subclasses such as `OrderedDict` and `defaultdict`.

    Args:
        value: A container supported by
            [container_children][confingy.utils.containers.container_children].
        new_children: The new children, in the order returned by
            `container_children(value)`.

    Returns:
        A new container of the same type as `value`.
    """
    if isinstance(value, list):
        new_list = copy.copy(value)
        new_list[:] = new_children
        return new_list
    if isinstance(value, tuple):
        if hasattr(type(value), "_fields"):  # namedtuple
            return type(value)(*new_children)
        return type(value)(new_children)
    if isinstance(value, dict):
        new_dict = copy.copy(value)
        for key, child in zip(value, new_children):
            new_dict[key] = child
        return new_dict
    if isinstance(value, (set, frozenset)):
        return type(value)(new_children)
    names = [f.name for f in dataclasses.fields(value) if f.init]
    return dataclasses.replace(value, **dict(zip(names, new_children)))


def same_structure(a: Any, b: Any) -> bool:
    """Check whether two values are identical, or containers of identical values.

    Containers are compared recursively by type, keys, and child identity. Leaf
    values are compared by identity only.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        True if `a` and `b` are the same object, or containers of the same type
        whose children are recursively the same.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, (set, frozenset)):
        # Members have no order to pair them up by. Unchanged objects compare
        # equal by identity or value, which is enough here.
        return bool(a == b)
    children_a = container_children(a)
    children_b = container_children(b)
    if children_a is None or children_b is None or len(children_a) != len(children_b):
        return False
    return all(
        suffix_a == suffix_b and same_structure(child_a, child_b)
        for (suffix_a, child_a), (suffix_b, child_b) in zip(children_a, children_b)
    )
