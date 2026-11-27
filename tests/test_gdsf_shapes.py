"""gsim on a gdsfactory that is not backed by klayout.

gdsf's ``gdsfactory`` differs from upstream in three ways the Palace path
touches:

* ``component.kcl`` carries ``dbu``, ``to_dbu`` and ``to_um`` only, with no
  klayout ``layout``;
* ``component.get_polygons()`` returns ``{(layer, datatype): [[(x, y), ...]]}``
  in um, not klayout polygons keyed by layer index;
* ``Port.info`` builds a new dict on every read, so a write is lost, and a port
  has no setter for its center.

The fakes below have exactly those properties. They are written from the
description of gdsf's API, because its packages cannot be imported here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

import gdsfactory as gf
import klayout.db as kdb
import pytest

from gsim.common.layout_access import (
    default_dbu,
    layer_polygons,
    layout_dbu,
    polygons_by_gds,
)
from gsim.common.stack import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace._mesh_fingerprint import _geometry_digest
from gsim.palace.base import _flat_copy
from gsim.palace.mesh.auto_size import min_conductor_feature_size
from gsim.palace.mesh.geometry import extract_geometry, extract_pec_polygons
from gsim.palace.ports.config import (
    PortType,
    configure_inplane_port,
    configure_wave_port,
    extract_ports,
)

STRIP = [[(0, 0), (10, 0), (10, 2), (0, 2)]]


class FakeKcl:
    """gdsf's ``kcl`` stand-in: a database unit and two converters."""

    dbu = 0.001

    def to_dbu(self, value: float) -> int:
        return round(value / self.dbu)

    def to_um(self, value: int) -> float:
        return value * self.dbu


@dataclass(frozen=True)
class FakePort:
    """A port that is a value: ``info`` is built anew on every read."""

    name: str
    center: tuple[float, float]
    width: float
    orientation: float | None
    layer: tuple[int, int] = (49, 0)

    @property
    def info(self) -> dict:
        return {}


@dataclass
class FakeComponent:
    """A component with gdsf's surface and no klayout layout."""

    polygons: dict = field(default_factory=lambda: {(49, 0): STRIP})
    ports: list = field(default_factory=list)
    kcl: FakeKcl = field(default_factory=FakeKcl)
    flatten_returns_copy: bool = False

    def get_polygons(self) -> dict:
        # No ``layers=`` argument: klayout layer indexes do not exist here
        return {
            key: [[tuple(pt) for pt in shape] for shape in shapes]
            for key, shapes in self.polygons.items()
        }

    def copy(self) -> FakeComponent:
        return FakeComponent(
            polygons=dict(self.polygons),
            ports=list(self.ports),
            flatten_returns_copy=self.flatten_returns_copy,
        )

    def flatten(self):
        self.flattened = True
        return self.copy() if self.flatten_returns_copy else None


def _stack() -> LayerStack:
    return LayerStack(
        layers={
            "m1": Layer(
                name="m1",
                gds_layer=(49, 0),
                zmin=1,
                zmax=2,
                thickness=1,
                material="aluminum",
                layer_type="conductor",
            )
        }
    )


def _ports() -> list[FakePort]:
    return [
        FakePort("o1", (0.0, 1.0), 2.0, 180.0),
        FakePort("o2", (10.0, 1.0), 2.0, 0.0),
    ]


def test_a_fake_port_loses_info_writes_and_cannot_move() -> None:
    port = _ports()[0]
    port.info["palace_type"] = "lumped"
    assert port.info == {}
    with pytest.raises(AttributeError):
        port.center = (1.0, 1.0)  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# 3a: the layout adapter
# ---------------------------------------------------------------------------


def test_polygons_by_gds_builds_klayout_polygons_from_um_points() -> None:
    component = FakeComponent()
    assert layout_dbu(component) == 0.001
    grouped = polygons_by_gds(component)
    assert list(grouped) == [(49, 0)]
    (polygon,) = grouped[(49, 0)]
    assert isinstance(polygon, kdb.Polygon)
    assert polygon.bbox() == kdb.Box(0, 0, 10000, 2000)
    assert polygon.num_points_hull() == 4


def test_the_adapter_skips_degenerate_shapes_and_keeps_empty_keys() -> None:
    component = FakeComponent(
        polygons={(49, 0): [[(0, 0), (1, 1)], *STRIP], (50, 0): []}
    )
    grouped = polygons_by_gds(component)
    assert len(grouped[(49, 0)]) == 1
    assert grouped[(50, 0)] == []


def test_layout_dbu_falls_back_to_the_default_without_a_kcl_dbu() -> None:
    component = FakeComponent()
    component.kcl = object()  # ty: ignore[invalid-assignment]
    assert layout_dbu(component) == 0.001


def test_a_key_that_is_not_a_layer_pair_is_refused() -> None:
    with pytest.raises(TypeError, match="layer, datatype"):
        layer_polygons(FakeComponent(polygons={"WG": STRIP}))


def test_extract_geometry_reads_a_component_without_a_layout() -> None:
    data = extract_geometry(FakeComponent(), _stack())
    ((layernum, pts_x, pts_y, holes),) = data.polygons
    assert layernum == 49
    assert sorted(zip(pts_x, pts_y, strict=True)) == [
        (0.0, 0.0),
        (0.0, 2.0),
        (10.0, 0.0),
        (10.0, 2.0),
    ]
    assert holes == []
    assert data.bbox == (0.0, 0.0, 10.0, 2.0)


def test_extract_pec_polygons_reads_a_component_without_a_layout() -> None:
    assert len(extract_pec_polygons(FakeComponent(), (49, 0))) == 1
    assert extract_pec_polygons(FakeComponent(), (50, 0)) == []


def test_auto_size_measures_a_component_without_a_layout() -> None:
    # The strip is 2 um across its narrow side
    assert min_conductor_feature_size(FakeComponent(), _stack()) == pytest.approx(2.0)


def test_the_mesh_fingerprint_digest_follows_the_polygons() -> None:
    same = _geometry_digest(FakeComponent())
    assert same == _geometry_digest(FakeComponent())
    moved = FakeComponent(polygons={(49, 0): [[(0, 0), (10, 0), (10, 3), (0, 3)]]})
    assert _geometry_digest(moved) != same
    assert _geometry_digest(None) == "none"


def test_the_adapter_agrees_with_klayout_polygons_of_upstream_gdsfactory() -> None:
    """Round trip: upstream polygons -> um points -> adapter -> same polygons."""
    gf.gpdk.PDK.activate()
    upstream = gf.Component()
    upstream.add_polygon([(0, 0), (10, 0), (10, 2), (5, 4), (0, 2)], layer=(49, 0))
    upstream.add_polygon([(20, 0), (30, 0), (30, 5)], layer=(49, 0))
    dbu = layout_dbu(upstream)
    assert dbu == default_dbu()

    expected = polygons_by_gds(upstream)[(49, 0)]
    um_shapes = [
        [(pt.x * dbu, pt.y * dbu) for pt in poly.each_point_hull()]
        for poly in expected
    ]
    fake = FakeComponent(polygons={(49, 0): um_shapes})
    adapted = polygons_by_gds(fake)[(49, 0)]
    assert len(adapted) == len(expected) == 2
    assert (kdb.Region(adapted) ^ kdb.Region(expected)).is_empty()
    assert sorted(p.bbox().to_s() for p in adapted) == sorted(
        p.bbox().to_s() for p in expected
    )


# ---------------------------------------------------------------------------
# 3b: the port-settings table
# ---------------------------------------------------------------------------


def test_add_wave_port_then_extract_ports_gives_the_ports() -> None:
    component = FakeComponent(ports=_ports())
    sim = DrivenSim()
    sim.set_geometry(component)  # ty: ignore[invalid-argument-type]
    sim.add_wave_port("o1", layer="m1", z_margin=1.5)
    sim.add_wave_port("o2", layer="m1", z_margin=1.5)
    sim._configure_ports_on_component(_stack())

    # Without the table the settings are gone: a fake port's info is fresh
    assert extract_ports(component, _stack()) == []

    ports = extract_ports(component, _stack(), sim._port_info)
    assert [p.name for p in ports] == ["o1", "o2"]
    assert all(p.port_type is PortType.WAVEPORT for p in ports)
    assert all(p.z_margin == 1.5 for p in ports)
    assert [(p.zmin, p.zmax) for p in ports] == [(1, 2), (1, 2)]


def test_a_simulation_reads_its_ports_back_through_its_own_table() -> None:
    component = FakeComponent(ports=_ports())
    sim = DrivenSim()
    sim.set_geometry(component)  # ty: ignore[invalid-argument-type]
    sim.add_wave_port("o1", layer="m1")
    assert [p.name for p in sim._palace_ports(_stack())] == ["o1"]


def test_a_port_offset_goes_to_the_table_once() -> None:
    component = FakeComponent(ports=_ports())
    sim = DrivenSim()
    sim.set_geometry(component)  # ty: ignore[invalid-argument-type]
    sim.add_port("o1", layer="m1", length=1.0, offset=2.0)
    for _ in range(2):  # configuring again must not shift again
        (port,) = sim._palace_ports(_stack())
        # o1 points west (180 deg), so a positive offset moves it east
        assert port.center == pytest.approx((2.0, 1.0))
    assert component.ports[0].center == (0.0, 1.0)  # the component is untouched


def test_the_store_leaves_the_port_alone_and_info_is_the_default() -> None:
    table: dict = {}
    configure_inplane_port(_ports()[0], layer="m1", length=1.0, store=table)
    assert table["o1"]["palace_type"] == "lumped"

    class Upstream:
        """A port whose info is a persistent dict, as in kfactory."""

        name = "o1"
        center = (0.0, 1.0)
        orientation = 180.0
        width = 2.0

        def __init__(self) -> None:
            self.info: dict = {}

    port = Upstream()
    configure_inplane_port(port, layer="m1", length=1.0, offset=1.0)
    assert port.info["palace_type"] == "lumped"
    assert port.center == pytest.approx((1.0, 1.0))  # moved in place, as before


def test_a_table_entry_belongs_to_the_first_port_of_a_repeated_name() -> None:
    first, second = _ports()[0], FakePort("o1", (5.0, 5.0), 2.0, 0.0)
    component = FakeComponent(ports=[first, second])
    table: dict = {}
    configure_wave_port(first, layer="m1", store=table)
    ports = extract_ports(component, _stack(), table)
    assert [p.center for p in ports] == [(0.0, 1.0)]


def test_wave_port_checks_still_run_before_the_table_is_written() -> None:
    table: dict = {}
    with pytest.raises(ValueError):
        configure_wave_port(_ports()[0], layer="m1", mode=2, modes=[1], store=table)
    assert table == {}


# ---------------------------------------------------------------------------
# 3c: the flat copy
# ---------------------------------------------------------------------------


def test_flat_copy_flattens_in_place_without_a_kdb_cell() -> None:
    original = FakeComponent()
    copied = _flat_copy(original)  # ty: ignore[invalid-argument-type]
    assert copied is not original
    assert copied.flattened  # ty: ignore[unresolved-attribute]
    assert not hasattr(original, "flattened")


def test_flat_copy_uses_the_component_flatten_returns() -> None:
    original = FakeComponent(flatten_returns_copy=True)
    copied = _flat_copy(original)  # ty: ignore[invalid-argument-type]
    assert isinstance(copied, FakeComponent)
    assert copied is not original


def test_flat_copy_keeps_the_klayout_route_for_upstream_components() -> None:
    gf.gpdk.PDK.activate()
    child = gf.Component()
    child.add_polygon([(0, 0), (1, 0), (1, 1)], layer=(49, 0))
    parent = gf.Component()
    parent << child
    copied = _flat_copy(parent)
    if not hasattr(parent, "kdb_cell"):
        # gdsf's own components: no klayout cell, ``flatten()`` works in place.
        assert len(copied.insts) == 0
        assert len(parent.insts) == 1
        return
    assert copied.kdb_cell.child_instances() == 0
    assert parent.kdb_cell.child_instances() == 1


# ---------------------------------------------------------------------------
# 3d: the cloud client is optional
# ---------------------------------------------------------------------------


def test_the_cloud_client_is_loaded_on_first_use(monkeypatch) -> None:
    from gsim import gcloud
    from gsim._cloud_sdk import CloudClientNotInstalledError

    monkeypatch.setitem(sys.modules, "gdsfactoryplus", None)  # import fails
    with pytest.raises(CloudClientNotInstalledError, match="gdsfactoryplus"):
        gcloud.get_status("job-1")
    with pytest.raises(ImportError):
        gcloud.start("job-1", verbose=False)


def test_the_cloud_proxy_forwards_patches_to_the_real_module(monkeypatch) -> None:
    pytest.importorskip("gdsfactoryplus.sim")
    from gdsfactoryplus import sim as real

    from gsim import gcloud

    monkeypatch.setattr(gcloud.sim, "gsim_probe", 3, raising=False)
    assert real.gsim_probe == 3  # ty: ignore[unresolved-attribute]
    monkeypatch.undo()
    assert not hasattr(real, "gsim_probe")


def test_gsim_palace_imports_without_a_cloud_client() -> None:
    code = textwrap.dedent(
        """
        import sys
        sys.modules["gdsfactoryplus"] = None
        import gsim.palace
        import gsim.gcloud
        assert gsim.palace.DrivenSim is not None
        """
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and snippet
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
