"""
Mypy type-checking tests for confingy.tracking.

This file has no pytest tests. Instead, it is analyzed statically by mypy as
part of `make mypy` (and so in CI):

    uv run mypy tests/test_tracking_mypy.py

All lines should type-check cleanly (exit 0, no errors). Expected errors are
marked with `# type: ignore[code]`, and unused ignores are errors for this file.
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from typing_extensions import assert_type

from confingy import Lazy, MaybeLazy, disable_validation, lazy, lens, track, update


# ---------------------------------------------------------------------------
# 1. @track decorator preserves class identity
# ---------------------------------------------------------------------------


@track
class Foo:
    def __init__(self, x: int, y: str = "hello") -> None:
        self.x = x
        self.y = y


@track()
class Bar:
    def __init__(self, val: float) -> None:
        self.val = val


@track(_validate=False)
class Baz:
    def __init__(self, name: str) -> None:
        self.name = name


# Instances are correctly typed
foo = Foo(x=1, y="world")
assert_type(foo, Foo)

bar = Bar(val=3.14)
assert_type(bar, Bar)

baz = Baz(name="test")
assert_type(baz, Baz)


# ---------------------------------------------------------------------------
# 2. .lazy() classmethod (mypy plugin)
# ---------------------------------------------------------------------------

lazy_foo = Foo.lazy(x=1, y="world")
assert_type(lazy_foo, Lazy[Foo])

lazy_bar = Bar.lazy(val=2.71)
assert_type(lazy_bar, Lazy[Bar])


# ---------------------------------------------------------------------------
# 3. lazy() function overloads
# ---------------------------------------------------------------------------

# lazy(Cls) returns a callable that produces Lazy[Cls]
foo_factory = lazy(Foo)
lazy_foo_from_factory = foo_factory(x=10, y="hi")
assert_type(lazy_foo_from_factory, Lazy[Foo])

# lazy(Cls, args) returns Lazy[Cls] directly
lazy_foo_direct = lazy(Foo, x=5, y="bye")
assert_type(lazy_foo_direct, Lazy[Foo])


# ---------------------------------------------------------------------------
# 4. Lazy[T] as a type hint
# ---------------------------------------------------------------------------


def accept_lazy_foo(lf: Lazy[Foo]) -> Foo:
    assert_type(lf.get_config(), dict[str, Any])
    copied = lf.copy(x=99)
    assert_type(copied, Lazy[Foo])
    instance = lf.instantiate()
    assert_type(instance, Foo)
    return instance


# Attribute access on Lazy returns Any (dynamic config access)
_attr: Any = lazy_foo.x


# ---------------------------------------------------------------------------
# 5. MaybeLazy[T] type alias
# ---------------------------------------------------------------------------


def accept_maybe_lazy(val: MaybeLazy[Foo]) -> None:
    pass


# Both T and Lazy[T] are accepted
accept_maybe_lazy(Foo(x=1))
accept_maybe_lazy(Foo.lazy(x=1))


# ---------------------------------------------------------------------------
# 6. lens() and update()
# ---------------------------------------------------------------------------

tracked_foo = Foo(x=1)
lensed = lens(tracked_foo)
assert_type(lensed, Lazy[Any])

updater = update(tracked_foo)
assert_type(updater, Callable[..., Any])


# ---------------------------------------------------------------------------
# 7. track() does not mutate original class (subclass fix)
# ---------------------------------------------------------------------------


class Original:
    def __init__(self, val: int) -> None:
        self.val = val


TrackedOriginal = track(Original)
assert_type(TrackedOriginal, type[Original])

tracked_instance = TrackedOriginal(val=42)
assert_type(tracked_instance, Original)


# ---------------------------------------------------------------------------
# 8. Inheritance with @track
# ---------------------------------------------------------------------------


@track
class Parent:
    def __init__(self, a: int) -> None:
        self.a = a


@track
class Child(Parent):
    def __init__(self, a: int, b: str = "default") -> None:
        super().__init__(a)
        self.b = b


child = Child(a=1, b="yes")
assert_type(child, Child)

lazy_child = Child.lazy(a=1, b="yes")
assert_type(lazy_child, Lazy[Child])


# ---------------------------------------------------------------------------
# 9. disable_validation() context manager
# ---------------------------------------------------------------------------

with disable_validation():
    Foo(x=1)


# ---------------------------------------------------------------------------
# 10. Generic classes keep their type arguments through .lazy()
# ---------------------------------------------------------------------------

T = TypeVar("T")
S = TypeVar("S")


@track
class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item


@track
class Pair(Generic[T, S]):
    def __init__(self, first: T, second: S) -> None:
        self.first = first
        self.second = second


assert_type(Box.lazy(item=1), Lazy[Box[int]])
assert_type(Box.lazy(item="x").instantiate(), Box[str])
assert_type(Box[int].lazy(item=1), Lazy[Box[int]])
assert_type(Pair.lazy(first=1, second="a"), Lazy[Pair[int, str]])


def accept_lazy_int_box(lb: Lazy[Box[int]]) -> None:
    pass


accept_lazy_int_box(Box.lazy(item=1))


# ---------------------------------------------------------------------------
# 11. Unannotated and inherited __init__ still get .lazy()
# ---------------------------------------------------------------------------


@track
class Untyped:
    def __init__(self, a, b=1):
        self.a = a
        self.b = b


class UntrackedBase:
    def __init__(self, v: int) -> None:
        self.v = v


@track
class InheritsInit(UntrackedBase):
    pass


assert_type(Untyped.lazy(a=1, b=2), Lazy[Untyped])
assert_type(InheritsInit.lazy(v=1), Lazy[InheritsInit])


# ---------------------------------------------------------------------------
# 12. Expected type errors
#
# Each line must produce the error code in its "type: ignore". This module has
# warn_unused_ignores enabled (see pyproject.toml), so if mypy stops reporting
# one of these errors, the ignore becomes unused and type checking fails.
# ---------------------------------------------------------------------------


def _expected_errors() -> None:
    """Never called; only type-checked."""
    # .lazy() from the mypy plugin checks argument types, names, and presence
    Foo.lazy(x="not an int")  # type: ignore[arg-type]
    Foo.lazy(x=1, z=1)  # type: ignore[call-arg]
    Foo.lazy()  # type: ignore[call-arg]
    Untyped.lazy(c=1)  # type: ignore[call-arg]
    Untyped.lazy()  # type: ignore[call-arg]
    InheritsInit.lazy(v="not an int")  # type: ignore[arg-type]

    # lazy() checks arguments through its ParamSpec overloads
    lazy(Foo)(x="not an int")  # type: ignore[arg-type]
    lazy(Foo, x=1, z=1)  # type: ignore[call-overload]

    # Lazy[T] hints reject a Lazy of an unrelated class
    accept_lazy_foo(Bar.lazy(val=1.0))  # type: ignore[arg-type]
    accept_lazy_int_box(Box.lazy(item="not an int"))  # type: ignore[arg-type]

    # @track keeps the constructor signature
    Foo(x="not an int")  # type: ignore[arg-type]
