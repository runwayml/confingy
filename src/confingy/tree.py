"""
Read, match, and transform utilities for trees of tracked and lazy objects.

A fingy is usually a tree: a root object whose constructor arguments contain
other tracked or [Lazy][confingy.tracking.Lazy] objects, possibly nested inside
lists, tuples, dicts, sets, or dataclasses. The functions in this module let you
inspect and rewrite such trees without touching confingy internals.

Examples:
    ```python
    from confingy import track, replace_args, walk_fingy

    @track
    class Scaler:
        def __init__(self, factor: float = 1.0):
            self.factor = factor

    @track
    class Pipeline:
        def __init__(self, steps: list):
            self.steps = steps

    pipeline = Pipeline(steps=[Scaler(), Scaler(factor=2.0)])

    for path, node in walk_fingy(pipeline):
        print(repr(path), type(node).__name__)
    # '' Pipeline
    # 'steps[0]' Scaler
    # 'steps[1]' Scaler

    new_pipeline, n_rebuilt = replace_args(pipeline, Scaler, factor=3.0)
    assert n_rebuilt == 2
    assert [s.factor for s in new_pipeline.steps] == [3.0, 3.0]
    assert pipeline.steps[0].factor == 1.0  # The input is never mutated
    ```
"""

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from confingy.tracking import Lazy, update
from confingy.utils.containers import container_children, rebuild_container
from confingy.utils.types import (
    TrackedInstance,
    is_lazy_instance,
    is_tracked_instance,
)

T = TypeVar("T")

_MISSING = object()


def get_init_args(obj: TrackedInstance | Lazy[Any]) -> dict[str, Any]:
    """Return the constructor arguments recorded for a tracked or lazy object.

    The returned dict is a shallow copy, so mutating it does not affect `obj`.
    To change arguments, use [update][confingy.tracking.update],
    [map_fingy][confingy.tree.map_fingy], or [replace_args][confingy.tree.replace_args].

    Args:
        obj: A tracked instance or a [Lazy][confingy.tracking.Lazy] instance.

    Returns:
        A new dict mapping constructor argument names to their values.

    Raises:
        TypeError: If `obj` is neither tracked nor lazy.

    Examples:
        ```python
        @track
        class Foo:
            def __init__(self, bar: int, baz: str = "x"):
                self.bar = bar
                self.baz = baz

        get_init_args(Foo(bar=1))       # {'bar': 1, 'baz': 'x'}
        get_init_args(Foo.lazy(bar=1))  # {'bar': 1, 'baz': 'x'}
        ```
    """
    if is_lazy_instance(obj):
        return obj.get_config()
    if is_tracked_instance(obj):
        return dict(obj._tracked_info["init_args"])
    raise TypeError(
        f"get_init_args() requires a tracked or Lazy instance, got {type(obj).__name__}"
    )


def is_fingy_of(obj: Any, expected: type) -> bool:
    """Check whether `obj` is a tracked or lazy instance of `expected` or a subclass.

    Unlike [is_lazy_version_of][confingy.utils.types.is_lazy_version_of], this
    matches the subclasses created by `track(expected)` and works for both tracked
    and lazy objects, so a single check finds a class however it was constructed.

    Always pass the *original* class. Each call to `track(SomeClass)` creates a new
    subclass, so `is_fingy_of(obj, track(SomeClass))` never matches anything.

    Args:
        obj: Any value. Values that are neither tracked nor lazy return `False`.
        expected: The class to match against.

    Returns:
        True if `obj` is a tracked instance of `expected` (or a subclass), or a
        `Lazy` whose target class is `expected` (or a subclass).

    Examples:
        ```python
        class Foo:
            def __init__(self, bar: int):
                self.bar = bar

        is_fingy_of(track(Foo)(bar=1), Foo)        # True
        is_fingy_of(lazy(track(Foo))(bar=1), Foo)  # True
        is_fingy_of(lens(track(Foo)(bar=1)), Foo)  # True
        is_fingy_of(Foo(bar=1), Foo)               # False: not tracked
        ```
    """
    if is_lazy_instance(obj):
        actual_cls = obj._confingy_actual_cls
        return isinstance(actual_cls, type) and issubclass(actual_cls, expected)
    if is_tracked_instance(obj):
        return isinstance(obj, expected)
    return False


def walk_fingy(obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield `(path, node)` for every tracked or lazy node in a tree.

    Nodes are yielded depth-first, parents before children. A node reachable
    from several parents is yielded once, at the first path it is found on.
    Children are found through recorded constructor arguments, and through
    lists, tuples, dicts, sets, and dataclasses nested inside them.

    Paths are built from argument names, sequence indices (`[0]`), dict keys
    (`['key']`), and dataclass fields (`.field`). Set members have no stable
    position and are shown as `{}`. The root node has the path `""`.

    Args:
        obj: The root of the tree. May be a tracked or lazy node, or a container
            holding them.

    Yields:
        `(path, node)` tuples.

    Examples:
        ```python
        pipeline = Pipeline(steps=[Scaler(), Scaler(factor=2.0)])
        [path for path, _ in walk_fingy(pipeline)]
        # ['', 'steps[0]', 'steps[1]']
        ```
    """
    seen: set[int] = set()

    def visit(value: Any, path: str) -> Iterator[tuple[str, Any]]:
        if _is_node(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            yield path, value
            for name, child in get_init_args(value).items():
                yield from visit(child, _join(path, name))
            return

        children = container_children(value)
        if children is None:
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        for suffix, child in children:
            yield from visit(child, path + suffix)

    yield from visit(obj, "")


def map_fingy(obj: T, fn: Callable[[Any], Any]) -> T:
    """Return a copy of a tree with `fn` applied to every tracked or lazy node.

    Nodes are visited bottom-up. For each node, its children are mapped first; if
    any child changed, the node is rebuilt with
    [update][confingy.tracking.update] using the new children. Then `fn` is
    called on the (possibly rebuilt) node, and its return value takes the node's
    place in the parent. A replacement returned by `fn` is not traversed again.

    Guarantees:

    - The input is never mutated.
    - Structural sharing: if `fn` returns every node unchanged, the result is `obj`
      itself. Only the ancestors of changed nodes are rebuilt, and untouched
      subtrees and containers keep their identity.
    - Laziness is preserved: lazy nodes are rebuilt as `Lazy`, tracked nodes as
      tracked instances.
    - A node shared by several parents is transformed once, and the result is
      shared in the output.

    Notes:

    - Children are found through recorded constructor arguments, not live
      attributes. Attributes set after construction are not part of the config
      and are not carried over when a node is rebuilt (the same contract as
      [update][confingy.tracking.update]).
    - Rebuilding re-runs the constructor of each rebuilt tracked node, and the
      validation and `__post_config__` hook of each rebuilt lazy node, so any side
      effects happen again. Only the changed path is rebuilt.
    - Nodes created with `track(existing_instance)` record attributes rather than
      constructor arguments, so they cannot be rebuilt. A `ValueError` is raised
      if one lies on a changed path.

    Args:
        obj: The root of the tree. May be a tracked or lazy node, or a container
            holding them.
        fn: Called with each node. Return the node itself to leave it unchanged,
            or a replacement.

    Returns:
        The transformed tree, or `obj` itself if nothing changed.

    Raises:
        ValueError: If the tree contains a cycle, or a node created with
            `track(existing_instance)` needs to be rebuilt.

    Examples:
        ```python
        pipeline = Pipeline(steps=[Scaler(factor=0.5), Scaler(factor=2.0)])

        def clamp(node):
            if is_fingy_of(node, Scaler) and get_init_args(node)["factor"] > 1.0:
                return update(node)(factor=1.0)
            return node

        clamped = map_fingy(pipeline, clamp)
        assert [s.factor for s in clamped.steps] == [0.5, 1.0]
        assert clamped.steps[0] is pipeline.steps[0]  # Untouched node is reused
        ```
    """
    return _map_fingy(obj, lambda node, path: fn(node))


def replace_args(obj: T, target: type, **init_args: Any) -> tuple[T, int]:
    """Set constructor arguments on every node in a tree that matches `target`.

    This is [map_fingy][confingy.tree.map_fingy] with a function that rebuilds each
    node matching [is_fingy_of(node, target)][confingy.tree.is_fingy_of] using
    `update(node)(**init_args)`. Nodes whose arguments already have the requested
    values are left untouched and not counted, so repeating a call is a no-op.

    Args:
        obj: The root of the tree.
        target: The class to match. Subclasses match too.
        **init_args: Constructor arguments to set on each matching node.

    Returns:
        A `(new_obj, n_rebuilt)` tuple, where `n_rebuilt` is the number of matching
        nodes that were rebuilt.

    Raises:
        ValidationError: If an argument is unknown or has an invalid value for a
            matching node's class.
        ValueError: See [map_fingy][confingy.tree.map_fingy].

    Examples:
        ```python
        pipeline = Pipeline(steps=[Scaler(), Scaler(factor=2.0)])
        new_pipeline, n = replace_args(pipeline, Scaler, factor=2.0)
        assert n == 1  # The second Scaler already had factor=2.0
        ```
    """
    n_rebuilt = 0

    def apply(node: Any, path: str) -> Any:
        nonlocal n_rebuilt
        if not is_fingy_of(node, target):
            return node
        current = get_init_args(node)
        if all(_same_value(current.get(k, _MISSING), v) for k, v in init_args.items()):
            return node
        n_rebuilt += 1
        return _rebuild(node, init_args, path)

    return _map_fingy(obj, apply), n_rebuilt


def _map_fingy(obj: T, fn: Callable[[Any, str], Any]) -> T:
    """Implementation of [map_fingy][confingy.tree.map_fingy] with node paths."""
    # Keys are ids of objects in the input tree, which stays alive for the whole
    # call, so ids cannot be reused.
    memo: dict[int, Any] = {}
    in_progress: set[int] = set()

    def visit(value: Any, path: str) -> Any:
        is_node = _is_node(value)
        children = None if is_node else container_children(value)
        if not is_node and children is None:
            return value

        key = id(value)
        if key in memo:
            return memo[key]
        if key in in_progress:
            raise ValueError(f"Cycle detected at {_describe(path)}")
        in_progress.add(key)
        try:
            if is_node:
                args = get_init_args(value)
                changes = {}
                for name, child in args.items():
                    new_child = visit(child, _join(path, name))
                    if new_child is not child:
                        changes[name] = new_child
                node = _rebuild(value, changes, path) if changes else value
                result = fn(node, path)
            else:
                assert children is not None
                new_children = [
                    visit(child, path + suffix) for suffix, child in children
                ]
                if all(new is old for new, (_, old) in zip(new_children, children)):
                    result = value
                else:
                    result = rebuild_container(value, new_children)
        finally:
            in_progress.discard(key)
        memo[key] = result
        return result

    return visit(obj, "")


def _is_node(value: Any) -> bool:
    return is_lazy_instance(value) or is_tracked_instance(value)


def _rebuild(node: Any, changes: dict[str, Any], path: str) -> Any:
    """Rebuild a tracked or lazy node with some constructor arguments replaced."""
    if is_lazy_instance(node):
        if node._confingy_was_instantiated:
            # Lazies created by lens() hold Lazy children where the class expects
            # instances, so update()'s validation would reject them.
            return node.copy(**changes)
        return update(node)(**changes)
    if node._tracked_info.get("from_instance"):
        raise ValueError(
            f"Cannot rebuild {type(node).__name__} at {_describe(path)}: it was "
            "tracked with track(existing_instance), so its recorded arguments are "
            "attributes, not constructor arguments."
        )
    return update(node)(**changes)


def _same_value(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # e.g. arrays, where == is elementwise
        return False


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _describe(path: str) -> str:
    return f"path '{path}'" if path else "the root"
