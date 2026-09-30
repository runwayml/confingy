"""
Tests for validating constructor arguments against type hints.

These cover `Lazy[X]` hints (including `Any`, unions, `Optional`, ABCs, and
Protocols), parameters whose names collide with pydantic `BaseModel`
attributes, and argument order. They run against a real module written to a
temporary directory, both with and without `from __future__ import annotations`,
so classes can also be imported by transpiled code.
"""

import importlib
import sys
import warnings
from textwrap import dedent

import pytest

from confingy import (
    ValidationError,
    deserialize_fingy,
    serialize_fingy,
    transpile_fingy,
)

MODULE_SOURCE = dedent(
    """
    from abc import ABC, abstractmethod
    from collections.abc import Iterable
    from typing import Any, Optional, Protocol, Union

    from confingy import Lazy, track


    @track
    class Encoder:
        def __init__(self, dim: int = 8):
            self.dim = dim


    @track
    class Decoder:
        def __init__(self, dim: int = 8):
            self.dim = dim


    @track
    class OtherBlock:  # same interface as Encoder, but not a subclass
        def __init__(self, dim: int = 8):
            self.dim = dim


    @track
    class IndexedDataset:  # iterable via __getitem__, no __iter__
        def __init__(self, size: int = 3):
            self.size = size

        def __getitem__(self, i: int) -> int:
            if i >= self.size:
                raise IndexError
            return i


    class HasDim(Protocol):
        dim: int


    class Block(ABC):  # a user's own ABC: checked nominally
        @abstractmethod
        def forward(self): ...


    @track
    class ConcreteBlock(Block):
        def __init__(self, dim: int = 8):
            self.dim = dim

        def forward(self):
            return self.dim


    @track
    class BlockStack:
        def __init__(self, blocks: list[Lazy[Block]]):
            self.blocks = blocks


    @track
    class AnyHolder:
        def __init__(self, child: Lazy[Any]):
            self.child = child


    @track
    class PipeUnionHolder:
        def __init__(self, child: Lazy[Encoder | Decoder]):
            self.child = child


    @track
    class TypingUnionHolder:
        def __init__(self, child: Lazy[Union[Encoder, Decoder]]):
            self.child = child


    @track
    class OptionalHolder:
        def __init__(self, child: Lazy[Optional[Encoder]]):
            self.child = child


    @track
    class GenericAliasUnionHolder:
        def __init__(self, child: Lazy[list[int] | Encoder]):
            self.child = child


    @track
    class ProtocolHolder:
        def __init__(self, child: Lazy[HasDim]):
            self.child = child


    @track
    class Estimator:
        def __init__(self, model_config: int, scale: int = 1):
            self.model_config = model_config
            self.scale = scale


    @track
    class Dumper:
        def __init__(self, model_dump: str = "json", model_validate: bool = True):
            self.model_dump = model_dump
            self.model_validate = model_validate


    @track
    class EncoderStack:
        def __init__(self, blocks: list[Lazy[Encoder]]):
            self.blocks = blocks


    @track
    class Loader:
        def __init__(self, dataset: Lazy[Iterable]):
            self.dataset = dataset


    @track
    class Model:
        def __init__(
            self,
            encoder: Lazy[Any],
            head: Lazy[Encoder | Decoder],
            estimator: Estimator,
            stack: EncoderStack,
        ):
            self.encoder = encoder
            self.head = head
            self.estimator = estimator
            self.stack = stack
    """
)


@pytest.fixture(params=[False, True], ids=["eager-annotations", "future-annotations"])
def mod(request, tmp_path, monkeypatch):
    """The test classes, in an importable module."""
    future = request.param
    name = f"hint_validation_{'future' if future else 'eager'}"
    header = "from __future__ import annotations\n" if future else ""
    (tmp_path / f"{name}.py").write_text(header + MODULE_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def _no_warnings(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


class TestLazyHints:
    def test_any_accepts_any_lazy(self, mod):
        assert _no_warnings(lambda: mod.AnyHolder.lazy(child=mod.Encoder.lazy()))
        assert _no_warnings(
            lambda: mod.AnyHolder(
                child=mod.Loader.lazy(dataset=mod.IndexedDataset.lazy())
            )
        )

    @pytest.mark.parametrize("holder", ["PipeUnionHolder", "TypingUnionHolder"])
    def test_union_accepts_members(self, mod, holder):
        cls = getattr(mod, holder)
        assert _no_warnings(lambda: cls.lazy(child=mod.Encoder.lazy()))
        assert _no_warnings(lambda: cls.lazy(child=mod.Decoder.lazy()))

    @pytest.mark.parametrize("holder", ["PipeUnionHolder", "TypingUnionHolder"])
    def test_union_warns_on_non_member(self, mod, holder):
        cls = getattr(mod, holder)
        with pytest.warns(UserWarning, match=r"Expected Lazy\[Encoder \| Decoder\]"):
            cls.lazy(child=mod.OtherBlock.lazy())

    def test_optional_checks_the_inner_class(self, mod):
        assert _no_warnings(lambda: mod.OptionalHolder.lazy(child=mod.Encoder.lazy()))
        with pytest.warns(UserWarning, match=r"Expected Lazy\[Encoder\]"):
            mod.OptionalHolder.lazy(child=mod.Decoder.lazy())

    def test_generic_alias_union_member(self, mod):
        assert _no_warnings(
            lambda: mod.GenericAliasUnionHolder.lazy(child=mod.Encoder.lazy())
        )
        with pytest.warns(UserWarning, match=r"Expected Lazy\[list \| Encoder\]"):
            mod.GenericAliasUnionHolder.lazy(child=mod.Decoder.lazy())

    def test_unrelated_class_warns_but_is_accepted(self, mod):
        with pytest.warns(UserWarning, match=r"got Lazy\[OtherBlock\]") as caught:
            stack = mod.EncoderStack.lazy(blocks=[mod.OtherBlock.lazy()])
        assert caught[0].filename == __file__  # Points at the caller
        assert stack.blocks[0].dim == 8

    def test_mismatches_warn_once_per_location(self, mod):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                mod.EncoderStack.lazy(
                    blocks=[mod.OtherBlock.lazy(), mod.OtherBlock.lazy()]
                )
        assert len(caught) == 1

    def test_structural_abc_accepts_any_lazy(self, mod):
        # Iterable's subclass check only looks for __iter__, which doesn't reflect
        # how IndexedDataset (iterable via __getitem__) is used
        assert _no_warnings(lambda: mod.Loader.lazy(dataset=mod.IndexedDataset.lazy()))

    def test_user_abc_is_checked_nominally(self, mod):
        assert _no_warnings(
            lambda: mod.BlockStack.lazy(blocks=[mod.ConcreteBlock.lazy()])
        )
        with pytest.warns(UserWarning, match=r"Expected Lazy\[Block\]"):
            mod.BlockStack.lazy(blocks=[mod.OtherBlock.lazy()])

    def test_protocol_accepts_any_lazy(self, mod):
        assert _no_warnings(
            lambda: mod.ProtocolHolder.lazy(child=mod.OtherBlock.lazy())
        )

    def test_non_lazy_is_still_rejected(self, mod):
        with pytest.raises(ValidationError, match="Expected a Lazy"):
            mod.EncoderStack.lazy(blocks=[mod.Encoder()])


class TestReservedParameterNames:
    """Parameters named like pydantic BaseModel attributes are validated normally."""

    def test_model_config(self, mod):
        assert mod.Estimator(5).model_config == 5
        assert mod.Estimator(model_config=5).model_config == 5
        assert mod.Estimator.lazy(model_config=5).instantiate().model_config == 5
        assert mod.Estimator.lazy(5).get_config() == {"model_config": 5, "scale": 1}

    def test_model_config_is_type_checked(self, mod):
        with pytest.raises(ValidationError, match="model_config"):
            mod.Estimator("not an int")
        with pytest.raises(ValidationError, match="model_config"):
            mod.Estimator.lazy(model_config="not an int")

    def test_other_basemodel_names(self, mod):
        dumper = mod.Dumper(model_dump="yaml", model_validate=False)
        assert (dumper.model_dump, dumper.model_validate) == ("yaml", False)
        with pytest.raises(ValidationError, match="model_validate"):
            mod.Dumper(model_validate="not a bool")


def test_arguments_are_stored_in_signature_order():
    from confingy import track

    @track
    class Ordered:
        def __init__(self, a: int, b: int = 2, c: int = 3, **extra: int):
            pass

    obj = Ordered(z=9, c=30, a=1, y=8)
    assert list(obj._tracked_info["init_args"]) == ["a", "b", "c", "z", "y"]
    assert list(Ordered.lazy(c=30, a=1).get_config()) == ["a", "b", "c"]


def test_round_trip_through_transpile(mod):
    config = mod.Model.lazy(
        encoder=mod.Encoder.lazy(dim=4),
        head=mod.Decoder.lazy(),
        estimator=mod.Estimator(model_config=3),
        stack=mod.EncoderStack(blocks=[mod.Encoder.lazy(), mod.Encoder.lazy(dim=2)]),
    )
    serialized = serialize_fingy(config)

    namespace: dict = {}
    exec(transpile_fingy(serialized), namespace)
    assert serialize_fingy(namespace["config"]) == serialized
    assert serialize_fingy(deserialize_fingy(serialized)) == serialized
