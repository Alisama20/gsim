"""Derived (boolean) layers on a gdsfactory that is not backed by klayout.

gdsf's ``DerivedLayer.get_shapes`` reads only the top cell and combines polygons
of the *same* layer only (``boolean(a, b, "not")`` without ``layer1``/``layer2``),
so ``WG - SHALLOW_ETCH`` returns ``WG`` unchanged and ``SHALLOW_ETCH & WG`` is
empty. Its ``LayerStack`` has no ``get_component_with_derived_layers`` method,
and its ``LayerLevel`` has no ``width_to_z``, ``bias``, ``z_to_bias``,
``model_fields`` or ``model_copy``.

gsim therefore evaluates the layer expressions itself, on klayout regions made
from ``component.get_polygons()``. The fakes below have the properties listed
above; the real-component tests run on whichever gdsfactory is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import gdsfactory as gf
import klayout.db as kdb
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

from gsim.common.layout_access import layout_dbu, polygons_by_gds
from gsim.common.pdk import resolve_passive_pcell
from gsim.common.pdk.derived import (
    derived_layer_regions,
    get_component_with_derived_layers,
)
from gsim.common.pdk.ports import _layer_tuples
from gsim.common.pdk.wafer import with_wafer_films
from gsim.common.polygon import (
    evaluate_layer_expression,
    fuse_polygons,
    layer_level_region,
)
from gsim.common.stack._layer_utils import (
    level_bias,
    level_width_to_z,
    level_z_to_bias,
)

WG = (1, 0)
ETCH = (2, 6)


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


class _Kcl:
    dbu = 0.001


@dataclass
class FakeComponent:
    """gdsf-like: no ``kdb_cell``, um polygons, a ``dbu`` only."""

    polygons: dict = field(default_factory=dict)
    kcl: _Kcl = field(default_factory=_Kcl)
    ports: list = field(default_factory=list)

    def get_polygons(self) -> dict:
        return {key: [list(shape) for shape in shapes] for key, shapes in self.polygons.items()}

    @property
    def dxmin(self) -> float:
        return min(x for shapes in self.polygons.values() for s in shapes for x, _ in s)

    @property
    def dxmax(self) -> float:
        return max(x for shapes in self.polygons.values() for s in shapes for x, _ in s)

    @property
    def dymin(self) -> float:
        return min(y for shapes in self.polygons.values() for s in shapes for _, y in s)

    @property
    def dymax(self) -> float:
        return max(y for shapes in self.polygons.values() for s in shapes for _, y in s)


def _strip_with_etch_bar() -> FakeComponent:
    """WG 10 x 0.5 um, a bar 2 um wide across it on the etch layer."""
    return FakeComponent(
        polygons={WG: [_rect(0, 0, 10, 0.5)], ETCH: [_rect(2, -1, 4, 1)]}
    )


def leaf(gds: tuple[int, int], **sizing) -> SimpleNamespace:
    return SimpleNamespace(layer=gds, **sizing)


def derived(layer1, layer2, operation: str) -> SimpleNamespace:
    return SimpleNamespace(layer1=layer1, layer2=layer2, operation=operation)


def _regions(component: FakeComponent) -> dict:
    return {
        gds: kdb.Region(polys) for gds, polys in polygons_by_gds(component).items()
    }


def _area_um2(region: kdb.Region) -> float:
    return region.area() * 1e-6


# ---------------------------------------------------------------------------
# the expression evaluator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "area"),
    [
        ("not", 4.0),
        ("-", 4.0),
        ("and", 1.0),
        ("&", 1.0),
        ("or", 5.0 + 4.0 - 1.0),
        ("|", 5.0 + 4.0 - 1.0),
        ("xor", 5.0 + 4.0 - 2.0),
        ("^", 5.0 + 4.0 - 2.0),
    ],
)
def test_operations_combine_different_layers(operation: str, area: float) -> None:
    regions = _regions(_strip_with_etch_bar())
    expression = derived(leaf(WG), leaf(ETCH), operation)

    assert _area_um2(evaluate_layer_expression(expression, regions)) == pytest.approx(
        area
    )


def test_nested_expression_and_sources_untouched() -> None:
    component = FakeComponent(
        polygons={
            WG: [_rect(0, 0, 10, 1)],
            ETCH: [_rect(1, 0, 2, 1)],
            (3, 6): [_rect(5, 0, 6, 1)],
        }
    )
    regions = _regions(component)
    before = {gds: region.dup() for gds, region in regions.items()}
    expression = derived(derived(leaf(WG), leaf((3, 6)), "not"), leaf(ETCH), "not")

    result = evaluate_layer_expression(expression, regions)

    assert _area_um2(result) == pytest.approx(8.0)
    assert all((regions[gds] ^ region).is_empty() for gds, region in before.items())


def test_missing_layer_is_an_empty_region_and_unknown_operation_raises() -> None:
    regions = _regions(_strip_with_etch_bar())

    assert evaluate_layer_expression(leaf((77, 0)), regions).is_empty()
    assert evaluate_layer_expression(
        derived(leaf((77, 0)), leaf(WG), "not"), regions
    ).is_empty()
    with pytest.raises(ValueError, match="Unknown layer operation"):
        evaluate_layer_expression(derived(leaf(WG), leaf(ETCH), "nand"), regions)


def test_sizing_is_applied_in_database_units() -> None:
    regions = _regions(_strip_with_etch_bar())
    grown = leaf(WG, sizings_xoffsets=(250,), sizings_yoffsets=(250,), sizings_modes=(2,))

    result = evaluate_layer_expression(grown, regions)

    assert _area_um2(result) == pytest.approx((10 + 0.5) * (0.5 + 0.5))


# ---------------------------------------------------------------------------
# a component without a klayout cell
# ---------------------------------------------------------------------------


def test_layer_level_region_evaluates_booleans_without_get_shapes() -> None:
    component = _strip_with_etch_bar()
    level = SimpleNamespace(layer=derived(leaf(WG), leaf(ETCH), "not"))

    assert _area_um2(layer_level_region(component, level)) == pytest.approx(4.0)


def test_fuse_polygons_cuts_the_etched_part() -> None:
    component = _strip_with_etch_bar()
    level = SimpleNamespace(layer=derived(leaf(WG), leaf(ETCH), "not"))

    geometry = fuse_polygons(component, level, round_tol=6, simplify_tol=0.0)

    assert geometry.area == pytest.approx(4.0)
    assert len(geometry.geoms) == 2  # the bar splits the strip in two
    assert geometry.bounds == pytest.approx((0, 0, 10, 0.5))


def test_background_level_fills_the_bounding_box_minus_the_exclusions() -> None:
    component = _strip_with_etch_bar()
    level = SimpleNamespace(
        layer=leaf((999, 0)),
        background=True,
        background_exclude_layers=[WG],
        thickness=1.0,
        sidewall_angle=0.0,
        width_to_z=0.0,
    )

    geometry = fuse_polygons(component, level, round_tol=6, simplify_tol=0.0)

    # bbox x 0..10, y -1..1 is 20 um2; the WG strip is 5 um2
    assert geometry.area == pytest.approx(15.0)


def test_derived_layer_regions_unite_levels_that_share_a_destination() -> None:
    component = _strip_with_etch_bar()
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(
                layer=derived(leaf(WG), leaf(ETCH), "not"), derived_layer=leaf(WG)
            ),
            "etched": SimpleNamespace(
                layer=derived(leaf(ETCH), leaf(WG), "and"), derived_layer=leaf(WG)
            ),
            "slab": SimpleNamespace(layer=leaf(ETCH), derived_layer=None),
        }
    )

    regions = derived_layer_regions(stack, component)

    assert set(regions) == {WG, ETCH}
    assert _area_um2(regions[WG]) == pytest.approx(5.0)  # 4 + 1, united
    assert _area_um2(regions[ETCH]) == pytest.approx(4.0)


def test_a_level_with_an_expression_needs_a_derived_layer() -> None:
    stack = SimpleNamespace(
        layers={"core": SimpleNamespace(layer=derived(leaf(WG), leaf(ETCH), "not"))}
    )

    with pytest.raises(ValueError, match="no derived_layer"):
        derived_layer_regions(stack, _strip_with_etch_bar())


def test_component_with_derived_layers_for_a_stack_without_the_method() -> None:
    component = _strip_with_etch_bar()
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(
                layer=derived(leaf(WG), leaf(ETCH), "not"), derived_layer=leaf(WG)
            )
        }
    )

    result = get_component_with_derived_layers(stack, component)

    by_gds = polygons_by_gds(result)
    area = kdb.Region(by_gds[WG]).area() * layout_dbu(result) ** 2
    # upstream lists every registered layer, most of them empty
    assert {gds for gds, polygons in by_gds.items() if polygons} == {WG}
    assert area == pytest.approx(4.0)


def test_the_stack_method_is_used_when_there_is_one() -> None:
    sentinel = object()
    stack = SimpleNamespace(
        layers={}, get_component_with_derived_layers=lambda component: (sentinel, component)
    )

    assert get_component_with_derived_layers(stack, "c") == (sentinel, "c")


# ---------------------------------------------------------------------------
# level fields gdsf does not have
# ---------------------------------------------------------------------------


def test_level_field_helpers_default_when_the_level_has_no_field() -> None:
    bare = SimpleNamespace()
    full = SimpleNamespace(width_to_z=0.5, bias=0.1, z_to_bias=([0, 1], [0, 0.1]))

    assert (level_width_to_z(bare), level_bias(bare), level_z_to_bias(bare)) == (
        0.0,
        None,
        None,
    )
    assert level_width_to_z(full) == 0.5
    assert level_bias(full) == 0.1
    assert level_z_to_bias(full) == ([0, 1], [0, 0.1])


def test_layer_tuples_reads_a_name_as_a_name_not_as_characters() -> None:
    assert _layer_tuples("3/6") == {(3, 6)}
    assert _layer_tuples("no_such_layer_name") == set()


def test_layer_tuples_resolves_a_name_through_the_active_pdk() -> None:
    gf.gpdk.PDK.activate()

    assert _layer_tuples("WG") == {tuple(gf.get_layer_tuple("WG"))}


# ---------------------------------------------------------------------------
# real stacks and components, on whichever gdsfactory is installed
# ---------------------------------------------------------------------------


def _etch_stack() -> LayerStack:
    return LayerStack(
        layers={
            "box": LayerLevel(
                layer=LogicalLayer(layer=(999, 0)),
                material="sio2",
                thickness=2.0,
                zmin=-2.0,
            ),
            "core": LayerLevel(
                layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH),
                derived_layer=LogicalLayer(layer=(10, 0)),
                material="Si",
                thickness=0.22,
                zmin=0.0,
            ),
            "slab": LayerLevel(
                layer=LogicalLayer(layer=ETCH) & LogicalLayer(layer=WG),
                derived_layer=LogicalLayer(layer=(11, 0)),
                material="Si",
                thickness=0.07,
                zmin=0.0,
            ),
        }
    )


def _real_strip_with_etch_bar() -> gf.Component:
    component = gf.Component()
    component.add_polygon(_rect(0, 0, 10, 0.5), layer=WG)
    component.add_polygon(_rect(2, -1, 4, 1), layer=ETCH)
    return component


def test_with_wafer_films_marks_a_copy_and_leaves_the_stack_alone() -> None:
    stack = _etch_stack()
    if not hasattr(stack.layers["box"], "background"):
        pytest.skip("this gdsfactory has no LayerLevel.background")

    marked = with_wafer_films(stack, None)

    assert marked is not stack
    assert marked.layers["box"].background is True
    assert stack.layers["box"].background is False
    assert marked.layers["core"].background is False
    box = marked.layers["box"]
    assert (box.material, box.thickness, box.zmin) == ("sio2", 2.0, -2.0)


def test_resolve_applies_the_booleans_of_the_stack() -> None:
    pdk = gf.Pdk(name="derived_layers_test", layer_stack=_etch_stack())

    resolved = resolve_passive_pcell(_real_strip_with_etch_bar(), pdk=pdk)

    assert resolved.layers["core"].geometry.area == pytest.approx(4.0)
    assert resolved.layers["slab"].geometry.area == pytest.approx(1.0)
    assert resolved.layers["core"].material == "Si"
    assert resolved.layers["core"].z_bounds == (0.0, 0.22)
    assert resolved.layers["slab"].z_bounds == (0.0, 0.07)
    # the blanket film spans the bounding box of the layout: 10 x 2 um
    assert resolved.layers["box"].geometry.area == pytest.approx(20.0)
    assert (10, 0) in resolved.derived_component.layers
    assert (11, 0) in resolved.derived_component.layers
