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

from mypy.nodes import Argument, FuncDef, TypeInfo, Var
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType, CallableType, Instance, Type, TypeOfAny, TypeVarType
from mypy.typevars import fill_typevars


def _track_class_decorator_callback(ctx: ClassDefContext) -> None:
    """Add the lazy classmethod to classes decorated with @track."""
    # Look up confingy.tracking.Lazy
    lazy_sym = ctx.api.lookup_fully_qualified_or_none("confingy.tracking.Lazy")
    if lazy_sym is None or lazy_sym.node is None:
        return

    lazy_info = lazy_sym.node
    if not isinstance(lazy_info, TypeInfo):
        return

    # Create Lazy[ThisClass] type. For generic classes this is Lazy[C[T, ...]],
    # so the class's type variables are inferred from the arguments to .lazy().
    class_type = fill_typevars(ctx.cls.info)
    if not isinstance(class_type, Instance):
        return
    lazy_type = Instance(lazy_info, [class_type])

    # Get the __init__ method to copy its signature
    init_method = ctx.cls.info.get_method("__init__")
    if init_method is None:
        return

    # Create the lazy classmethod signature: same args as __init__ (skipping
    # 'self'), returns Lazy[T]
    init_type = init_method.type
    tvar_defs: list[TypeVarType] = []
    if isinstance(init_type, CallableType):
        lazy_arg_names = init_type.arg_names[1:]
        lazy_arg_types: list[Type] = list(init_type.arg_types[1:])
        lazy_arg_kinds = init_type.arg_kinds[1:]
        # Type variables scoped to __init__ itself (not to the class)
        tvar_defs = [v for v in init_type.variables if isinstance(v, TypeVarType)]
    elif isinstance(init_method, FuncDef):
        # Unannotated __init__: mypy hasn't inferred a type for it, so keep the
        # parameter names and kinds and type every parameter as Any
        params = init_method.arguments[1:]
        lazy_arg_names = [p.variable.name for p in params]
        lazy_arg_types = [AnyType(TypeOfAny.unannotated) for _ in params]
        lazy_arg_kinds = [p.kind for p in params]
    else:
        return

    # Add the lazy classmethod to the class
    add_method_to_class(
        ctx.api,
        ctx.cls,
        "lazy",
        args=[
            Argument(Var(name or f"arg{i}", typ), typ, None, kind)
            for i, (name, typ, kind) in enumerate(
                zip(lazy_arg_names, lazy_arg_types, lazy_arg_kinds)
            )
        ],
        return_type=lazy_type,
        tvar_def=tvar_defs or None,
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
