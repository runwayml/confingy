"""Mypy plugin for confingy.

This plugin teaches mypy that classes decorated with @track have a .lazy() classmethod
that returns Lazy[T] where T is the class type.

To use this plugin, add to your mypy.ini or pyproject.toml:

    [mypy]
    plugins = confingy.mypy_plugin

Or in pyproject.toml:

    [tool.mypy]
    plugins = ["confingy.mypy_plugin"]
"""

from typing import Callable

from mypy.nodes import Argument, TypeInfo, Var
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.common import add_method_to_class
from mypy.types import CallableType, Instance


def _track_class_decorator_callback(ctx: ClassDefContext) -> None:
    """Add the lazy classmethod to classes decorated with @track."""
    # Look up confingy.tracking.Lazy
    lazy_sym = ctx.api.lookup_fully_qualified_or_none("confingy.tracking.Lazy")
    if lazy_sym is None or lazy_sym.node is None:
        return

    lazy_info = lazy_sym.node
    if not isinstance(lazy_info, TypeInfo):
        return

    # Create Lazy[ThisClass] type
    class_type = Instance(ctx.cls.info, [])
    lazy_type = Instance(lazy_info, [class_type])

    # Get the __init__ method to copy its signature
    init_method = ctx.cls.info.get_method("__init__")
    if init_method is None:
        return

    init_type = init_method.type
    if not isinstance(init_type, CallableType):
        return

    # Create the lazy classmethod signature: same args as __init__, returns Lazy[T]
    # Skip 'self' argument (first arg of __init__)
    lazy_args = init_type.arg_types[1:]  # Skip self
    lazy_arg_names = init_type.arg_names[1:]  # Skip self
    lazy_arg_kinds = init_type.arg_kinds[1:]  # Skip self

    # Add the lazy classmethod to the class
    add_method_to_class(
        ctx.api,
        ctx.cls,
        "lazy",
        args=[
            Argument(Var(name or f"arg{i}", typ), typ, None, kind)
            for i, (name, typ, kind) in enumerate(
                zip(lazy_arg_names, lazy_args, lazy_arg_kinds)
            )
        ],
        return_type=lazy_type,
        is_classmethod=True,
    )


class ConfingyPlugin(Plugin):
    """Mypy plugin for confingy."""

    def get_class_decorator_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        """Hook for class decorators."""
        if fullname in ("confingy.track", "confingy.tracking.track"):
            return _track_class_decorator_callback
        return None


def plugin(version: str) -> type[Plugin]:
    """Entry point for mypy plugin."""
    return ConfingyPlugin
