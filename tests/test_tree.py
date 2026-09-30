"""Tests for the tree read, match, and transform utilities."""

import dataclasses
from collections import OrderedDict, namedtuple

import pytest

from confingy import (
    Lazy,
    ValidationError,
    get_init_args,
    is_fingy_of,
    lazy,
    lens,
    map_fingy,
    replace_args,
    serialize_fingy,
    track,
    update,
    walk_fingy,
)


class Scaler:
    def __init__(self, factor: float = 1.0, offset: float = 0.0):
        self.factor = factor
        self.offset = offset


class SpecialScaler(Scaler):
    pass


class Pipeline:
    def __init__(self, steps: list):
        self.steps = steps


class Registry:
    def __init__(self, entries: dict):
        self.entries = entries


class Pair:
    def __init__(self, left: object, right: object):
        self.left = left
        self.right = right


@dataclasses.dataclass
class Holder:
    item: object
    note: str = "hi"


def identity(node):
    return node


def build_registry():
    return track(Registry)(
        entries={
            "a": track(Pipeline)(steps=[track(Scaler)(), "passthrough"]),
            "b": track(Pipeline)(steps=[track(Scaler)(factor=2.0)]),
        }
    )


# get_init_args


def test_get_init_args_tracked_and_lazy_match():
    tracked = track(Scaler)(factor=2.0)
    lazy_obj = lazy(track(Scaler))(factor=2.0)
    assert get_init_args(tracked) == get_init_args(lazy_obj)
    assert get_init_args(tracked)["factor"] == 2.0


def test_get_init_args_returns_copy():
    for obj in (track(Scaler)(factor=2.0), lazy(track(Scaler))(factor=2.0)):
        args = get_init_args(obj)
        args["factor"] = 99.0
        assert get_init_args(obj)["factor"] == 2.0


def test_get_init_args_rejects_plain_objects():
    with pytest.raises(TypeError, match="tracked or Lazy"):
        get_init_args(Scaler())


# is_fingy_of


def test_is_fingy_of_matches_all_construction_styles():
    @track
    class Decorated:
        def __init__(self, x: int = 1):
            self.x = x

    tracked = track(Scaler)()
    assert is_fingy_of(tracked, Scaler)
    assert is_fingy_of(track(Scaler, factor=2.0), Scaler)
    assert is_fingy_of(lazy(track(Scaler))(), Scaler)
    assert is_fingy_of(lazy(Scaler)(), Scaler)
    assert is_fingy_of(lens(tracked), Scaler)
    assert is_fingy_of(Decorated(), Decorated)
    assert is_fingy_of(Decorated.lazy(), Decorated)
    assert is_fingy_of(lazy(Decorated, x=2), Decorated)


def test_is_fingy_of_matches_subclasses():
    assert is_fingy_of(track(SpecialScaler)(), Scaler)
    assert is_fingy_of(lazy(track(SpecialScaler))(), Scaler)
    assert not is_fingy_of(track(Scaler)(), SpecialScaler)


def test_is_fingy_of_rejects_other_values():
    assert not is_fingy_of(track(Scaler)(), Pipeline)
    assert not is_fingy_of(lazy(track(Scaler))(), Pipeline)
    assert not is_fingy_of(Scaler(), Scaler)  # Not tracked
    assert not is_fingy_of("text", str)
    assert not is_fingy_of(None, Scaler)


# walk_fingy


def test_walk_fingy_paths():
    registry = build_registry()
    walked = [(path, type(node).__name__) for path, node in walk_fingy(registry)]
    assert walked == [
        ("", "Registry"),
        ("entries['a']", "Pipeline"),
        ("entries['a'].steps[0]", "Scaler"),
        ("entries['b']", "Pipeline"),
        ("entries['b'].steps[0]", "Scaler"),
    ]


def test_walk_fingy_yields_shared_nodes_once():
    shared = track(Scaler)()
    pair = track(Pair)(left=shared, right=[shared])
    paths = [path for path, node in walk_fingy(pair) if node is shared]
    assert paths == ["left"]


def test_walk_fingy_containers_and_lazy():
    tree = track(Pair)(
        left=Holder(item=lazy(track(Scaler))()),
        right=(track(Scaler)(), {track(Scaler)(factor=3.0)}),
    )
    paths = [path for path, _ in walk_fingy(tree)]
    assert paths == ["", "left.item", "right[0]", "right[1]{}"]


def test_walk_fingy_root_container():
    nodes = [track(Scaler)(), lazy(track(Scaler))()]
    assert [path for path, _ in walk_fingy(nodes)] == ["[0]", "[1]"]


# map_fingy


def test_map_fingy_identity_returns_same_root():
    registry = build_registry()
    assert map_fingy(registry, identity) is registry
    containers = [registry, [registry], {"k": (registry,)}]
    for container in containers:
        assert map_fingy(container, identity) is container


def test_map_fingy_rebuilds_only_ancestors():
    registry = build_registry()
    target = registry.entries["a"].steps[0]

    def change_target(node):
        return update(node)(factor=5.0) if node is target else node

    new = map_fingy(registry, change_target)
    assert new is not registry
    assert new.entries["a"] is not registry.entries["a"]
    assert new.entries["a"].steps[0].factor == 5.0
    assert new.entries["a"].steps[1] == "passthrough"
    # Sibling subtree keeps its identity
    assert new.entries["b"] is registry.entries["b"]


def test_map_fingy_does_not_mutate_input():
    registry = build_registry()
    before = serialize_fingy(registry)
    replace_args(registry, Scaler, factor=7.0)
    assert serialize_fingy(registry) == before
    assert registry.entries["a"].steps[0].factor == 1.0


def test_map_fingy_rebuilt_node_attributes_match_config():
    registry = build_registry()
    new, _ = replace_args(registry, Scaler, factor=7.0)
    step = new.entries["a"].steps[0]
    assert step.factor == 7.0
    assert get_init_args(step)["factor"] == 7.0


def test_map_fingy_shared_node_transformed_once_and_stays_shared():
    shared = track(Scaler)()
    pair = track(Pair)(left=shared, right=[shared])
    calls = []

    def bump(node):
        if is_fingy_of(node, Scaler):
            calls.append(node)
            return update(node)(factor=2.0)
        return node

    new = map_fingy(pair, bump)
    assert len(calls) == 1
    assert new.left is new.right[0]
    assert new.left.factor == 2.0


def test_map_fingy_preserves_laziness_in_mixed_trees():
    tree = track(Pair)(
        left=lazy(track(Pipeline))(steps=[track(Scaler)()]),
        right=track(Pipeline)(steps=[lazy(track(Scaler))()]),
    )
    new, n = replace_args(tree, Scaler, factor=3.0)
    assert n == 2
    assert isinstance(new.left, Lazy)
    assert not isinstance(new.left.steps[0], Lazy)
    assert new.left.steps[0].factor == 3.0
    assert not isinstance(new.right, Lazy)
    assert isinstance(new.right.steps[0], Lazy)
    assert new.right.steps[0].factor == 3.0
    # The lazy parent still instantiates correctly
    assert new.left.instantiate().steps[0].factor == 3.0


def test_map_fingy_reaches_all_container_types():
    Point = namedtuple("Point", ["x", "y"])
    tree = track(Pair)(
        left=[
            (track(Scaler)(),),
            {"k": track(Scaler)()},
            Holder(item=track(Scaler)()),
            Point(x=track(Scaler)(), y=0),
        ],
        right=OrderedDict(z=track(Scaler)()),
    )
    new, n = replace_args(tree, Scaler, factor=4.0)
    assert n == 5
    assert new.left[0][0].factor == 4.0
    assert new.left[1]["k"].factor == 4.0
    assert new.left[2].item.factor == 4.0
    assert new.left[2].note == "hi"
    assert isinstance(new.left[3], Point)
    assert new.left[3].x.factor == 4.0
    assert isinstance(new.right, OrderedDict)
    assert new.right["z"].factor == 4.0


def test_map_fingy_reaches_sets():
    tree = track(Pair)(left={track(Scaler)()}, right=frozenset())
    new, n = replace_args(tree, Scaler, factor=4.0)
    assert n == 1
    assert [s.factor for s in new.left] == [4.0]
    assert new.right is tree.right


def test_map_fingy_untouched_containers_keep_identity():
    untouched = [track(Scaler)(factor=2.0), "x"]
    tree = track(Pair)(left=untouched, right=[track(Scaler)()])
    new, _ = replace_args(tree, Scaler, factor=2.0)
    assert new.left is untouched
    assert new.right is not tree.right


def test_map_fingy_serializes_like_hand_built_tree():
    registry = build_registry()
    new, _ = replace_args(registry, Scaler, offset=0.5)
    expected = track(Registry)(
        entries={
            "a": track(Pipeline)(steps=[track(Scaler)(offset=0.5), "passthrough"]),
            "b": track(Pipeline)(steps=[track(Scaler)(factor=2.0, offset=0.5)]),
        }
    )
    assert serialize_fingy(new) == serialize_fingy(expected)


def test_map_fingy_fn_sees_rebuilt_children():
    tree = track(Pipeline)(steps=[track(Scaler)()])
    seen = {}

    def record(node):
        if is_fingy_of(node, Scaler):
            return update(node)(factor=9.0)
        if is_fingy_of(node, Pipeline):
            seen["factor"] = get_init_args(node)["steps"][0].factor
        return node

    map_fingy(tree, record)
    assert seen["factor"] == 9.0


def test_map_fingy_replacement_is_not_traversed():
    tree = track(Pipeline)(steps=[track(Scaler)()])
    replacement = track(Pipeline)(steps=[track(Scaler)(factor=5.0)])
    visited = []

    def swap(node):
        visited.append(type(node).__name__)
        return replacement if is_fingy_of(node, Pipeline) else node

    assert map_fingy(tree, swap) is replacement
    assert visited == ["Scaler", "Pipeline"]


def test_map_fingy_detects_cycles():
    root = lazy(track(Pair))(left=None, right=None)
    root.left = [root]
    with pytest.raises(ValueError, match="Cycle detected"):
        map_fingy(root, identity)


def test_map_fingy_rejects_rebuilding_instance_tracked_nodes():
    class Legacy:
        def __init__(self, child: object):
            self.child = child
            self.label = "derived"

    legacy = track(Legacy(child=track(Scaler)()))
    assert get_init_args(legacy) == {"child": legacy.child, "label": "derived"}

    # Fine when the node is not on a changed path
    tree = track(Pair)(left=legacy, right=track(SpecialScaler)())
    new, _ = replace_args(tree, SpecialScaler, factor=2.0)
    assert new.left is legacy

    # Rebuilding it would pass attributes as constructor arguments
    tree = track(Pair)(left=[legacy], right=1)
    with pytest.raises(
        ValueError, match=r"Legacy at path 'left\[0\]'.*track\(existing_instance\)"
    ):
        replace_args(tree, Scaler, factor=2.0)

    with pytest.raises(ValueError, match="the root"):
        replace_args(legacy, Legacy, child=None)


def test_map_fingy_on_lensed_tree():
    tree = track(Pipeline)(steps=[track(Scaler)()])
    lensed = lens(tree)
    new, n = replace_args(lensed, Scaler, factor=6.0)
    assert n == 1
    rebuilt = new.unlens()
    assert rebuilt.steps[0].factor == 6.0


def test_map_fingy_reruns_post_config_hook_on_rebuilt_lazy():
    calls = []

    @track
    class Hooked:
        def __init__(self, child: object = None):
            self.child = child

        @staticmethod
        def __post_config__(lazy_obj, changed_key):
            calls.append(changed_key)
            return lazy_obj

    tree = Hooked.lazy(child=lazy(track(Scaler))())
    calls.clear()
    replace_args(tree, Scaler, factor=2.0)
    assert calls == [None]


# replace_args


def test_replace_args_counts_and_is_idempotent():
    registry = build_registry()
    new, n = replace_args(registry, Scaler, factor=2.0)
    assert n == 1  # The second Scaler already has factor=2.0
    again, n_again = replace_args(new, Scaler, factor=2.0)
    assert n_again == 0
    assert again is new


def test_replace_args_nested_targets():
    tree = track(Pair)(left=track(Pair)(left=1, right=2), right=3)
    new, n = replace_args(tree, Pair, right=0)
    assert n == 2
    assert new.right == 0
    assert new.left.right == 0


def test_replace_args_invalid_value_raises_validation_error():
    registry = build_registry()
    with pytest.raises(ValidationError):
        replace_args(registry, Scaler, factor="not a float")


def test_replace_args_unknown_arg_raises_validation_error():
    for tree in (
        build_registry(),
        lazy(track(Pipeline))(steps=[lazy(track(Scaler))()]),
    ):
        with pytest.raises(ValidationError, match="not_an_arg"):
            replace_args(tree, Scaler, not_an_arg=1)


# lens / unlens sharing


def test_unlens_preserves_sharing():
    shared = track(Scaler)()
    lensed = lens(track(Pair)(left=shared, right=shared))
    assert lensed.left is lensed.right
    lensed.left.factor = 3.0
    rebuilt = lensed.unlens()
    assert rebuilt.left is rebuilt.right
    assert rebuilt.right.factor == 3.0


def test_unlens_reuses_unmodified_nodes():
    original = track(Pair)(left=track(Scaler)(), right=[track(Scaler)()])
    assert lens(original).unlens() is original

    lensed = lens(original)
    lensed.left.factor = 2.0
    rebuilt = lensed.unlens()
    assert rebuilt is not original
    assert rebuilt.left.factor == 2.0
    assert rebuilt.right[0] is original.right[0]


def test_unlens_detects_container_edits():
    original = track(Pipeline)(steps=[track(Scaler)()])
    lensed = lens(original)
    lensed.steps.append(track(Scaler)(factor=2.0))
    rebuilt = lensed.unlens()
    assert rebuilt is not original
    assert len(rebuilt.steps) == 2
    assert len(original.steps) == 1


def test_unlens_lazy_root_with_tracked_children():
    original = lazy(track(Pair))(left=track(Scaler)(), right=None)
    assert lens(original).unlens() is original

    lensed = lens(original)
    lensed.left.factor = 2.0
    rebuilt = lensed.unlens()
    assert isinstance(rebuilt, Lazy)
    assert rebuilt.left.factor == 2.0
    assert original.left.factor == 1.0
