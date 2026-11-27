"""gsim on a gdsfactory whose layer levels name their layers.

In gdsf a ``LayerLevel`` refers to its layer by name: ``level.layer`` is a
``LogicalLayer`` whose ``.layer`` is ``"M3"``, or a ``DerivedLayer`` with
``layer1``/``layer2`` operands and no ``.layer`` at all. The level's
``derived_layer`` is a ``LogicalLayer`` that names the mask used for the
polygons. Its layer map (``stack.layers``) offers ``items``, ``keys`` and
``values`` and ``stack.layers[name]``, but plain iteration fails.

The fakes below copy those properties, as they were read from the built
packages. A fake ``get_layer_tuple`` stands in for gdsf's name lookup.
"""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest

from gsim.common.pdk import resolve_layer_stack
from gsim.common.stack._layer_utils import (
    _tuple_from_layer_expression,
    get_gds_layer_tuple,
)

NAMES = {"WG": (1, 0), "M1": (41, 0), "M3": (49, 0), "DEEP_ETCH": (3, 6), "WAFER": (999, 0)}


class FakeLogicalLayer:
    """gdsf ``LogicalLayer``: ``layer`` is the layer's name."""

    def __init__(self, name: str) -> None:
        self.layer = name


class FakeDerivedLayer:
    """gdsf ``DerivedLayer``: operands, and no ``layer`` attribute."""

    def __init__(self, layer1, layer2, operation: str = "not") -> None:
        self.layer1 = layer1
        self.layer2 = layer2
        self.operation = operation


class NoIterMap:
    """gdsf's ``LayerStackMap``: dict-like methods, plain iteration fails."""

    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("argument 'key': 'int' object is not an instance of 'str'")
        return self._data[key]

    def __iter__(self):
        return self[0]  # the failure gdsf has: a lookup by position

    def keys(self):
        return list(self._data)

    def values(self):
        return list(self._data.values())

    def items(self):
        return list(self._data.items())


@pytest.fixture(autouse=True)
def _name_lookup(monkeypatch):
    def get_layer_tuple(layer):
        if layer not in NAMES:
            raise KeyError(f'unknown layer name "{layer}"')
        return NAMES[layer]

    monkeypatch.setattr(gf, "get_layer_tuple", get_layer_tuple)


def _level(layer, *, derived=None, **fields):
    defaults = {
        "material": "aluminum",
        "zmin": 0.0,
        "thickness": 1.0,
        "info": None,
        "mesh_order": 2,
        "sidewall_angle": 0.0,
        "name": None,
    }
    return SimpleNamespace(layer=layer, derived_layer=derived, **{**defaults, **fields})


@pytest.mark.parametrize("name", ["WG", "M3", "M1", "DEEP_ETCH", "WAFER"])
def test_a_logical_layer_resolves_through_its_name(name):
    assert get_gds_layer_tuple(_level(FakeLogicalLayer(name))) == NAMES[name]


def test_a_two_letter_name_is_not_split_into_characters():
    # tuple("M3") == ("M", "3"): int("M") is the error gdsf first produced.
    assert get_gds_layer_tuple(_level(FakeLogicalLayer("M3"))) == (49, 0)


def test_a_derived_layer_uses_the_levels_derived_layer():
    core = FakeDerivedLayer(
        FakeDerivedLayer(FakeLogicalLayer("WG"), FakeLogicalLayer("DEEP_ETCH")),
        FakeLogicalLayer("M1"),
    )
    level = _level(core, derived=FakeLogicalLayer("WG"))
    assert get_gds_layer_tuple(level) == (1, 0)


def test_a_derived_layer_without_derived_layer_uses_its_first_leaf():
    expression = FakeDerivedLayer(FakeLogicalLayer("M3"), FakeLogicalLayer("WG"), "and")
    assert get_gds_layer_tuple(_level(expression)) == (49, 0)


def test_an_unknown_name_gives_none_without_raising():
    assert get_gds_layer_tuple(_level(FakeLogicalLayer("NOPE"))) is None
    assert get_gds_layer_tuple(_level("NOPE")) is None


def test_a_bare_name_and_a_pair_string():
    assert get_gds_layer_tuple(_level("M3")) == (49, 0)
    assert get_gds_layer_tuple(_level("3/2")) == (3, 2)
    assert _tuple_from_layer_expression("M3") == (49, 0)
    assert _tuple_from_layer_expression("3/2") == (3, 2)
    assert _tuple_from_layer_expression("a/b") is None


def test_the_name_lookup_follows_the_active_pdk_of_upstream_gdsfactory(monkeypatch):
    monkeypatch.undo()  # the real gf.get_layer_tuple
    gf.gpdk.PDK.activate()
    expected = tuple(gf.gpdk.LAYER.M3)
    assert get_gds_layer_tuple(_level(FakeLogicalLayer("M3"))) == expected


def _pdk(levels: dict):
    return SimpleNamespace(
        name="gdsf_like",
        layer_stack=SimpleNamespace(layers=NoIterMap(levels)),
    )


def _gdsf_like_levels() -> dict:
    return {
        "metal1": _level(FakeLogicalLayer("M1"), info={"layer_type": "conductor"}),
        "metal3": _level(
            FakeLogicalLayer("M3"), zmin=2.0, info={"layer_type": "conductor"}
        ),
        "clad": _level(
            FakeLogicalLayer("WAFER"),
            material="SiO2",
            zmin=-2.0,
            thickness=6.0,
            info={"layer_type": "dielectric"},
        ),
    }


def test_resolve_layer_stack_reads_a_layer_map_that_cannot_be_iterated():
    with pytest.raises(TypeError):
        list(_pdk(_gdsf_like_levels()).layer_stack.layers)  # the premise
    resolved = resolve_layer_stack(pdk=_pdk(_gdsf_like_levels()))
    assert resolved.layers["metal1"].gds_layer == (41, 0)
    assert resolved.layers["metal3"].gds_layer == (49, 0)
    assert resolved.layers["clad"].gds_layer == (999, 0)


def test_resolve_layer_stack_checks_overrides_against_a_non_iterable_map():
    pdk = _pdk(_gdsf_like_levels())
    resolved = resolve_layer_stack(pdk=pdk, role_overrides={"metal3": "via"})
    assert resolved.layers["metal3"].layer_type == "via"
    with pytest.raises(Exception, match="unknown layer"):
        resolve_layer_stack(pdk=pdk, role_overrides={"nope": "via"})
