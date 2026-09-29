"""Tests for the mesh statistics of high-order meshes.

Raising the element order only adds nodes on the edges of the same tetrahedra:
the number of elements, their corners, and so their shape quality and edge
lengths do not change. The statistics of a second- or third-order mesh must
therefore match those of the first-order mesh they come from.
"""

from __future__ import annotations

from collections.abc import Iterator

import gdsfactory as gf
import gmsh
import pytest

from gsim.palace import DrivenSim
from gsim.palace.base import _estimate_dofs
from gsim.palace.mesh.config_generator import collect_mesh_stats


@pytest.fixture
def _gmsh_session() -> Iterator[None]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def _mesh_a_box() -> int:
    """Mesh a unit box with linear tetrahedra and return how many there are."""
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
    gmsh.model.mesh.generate(3)
    return len(gmsh.model.mesh.getElementsByType(4)[0])


@pytest.mark.usefixtures("_gmsh_session")
@pytest.mark.parametrize("order", [2, 3])
def test_raising_the_order_does_not_change_the_tetrahedron_count(order) -> None:
    linear = _mesh_a_box()
    assert collect_mesh_stats()["tetrahedra"] == linear
    gmsh.model.mesh.setOrder(order)
    assert collect_mesh_stats()["tetrahedra"] == linear


@pytest.mark.usefixtures("_gmsh_session")
def test_quadrilaterals_are_not_counted_as_tetrahedra() -> None:
    """A quadrilateral has four corners too, but it is a surface element."""
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    plate = gmsh.model.occ.addRectangle(3, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setRecombine(2, plate)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
    gmsh.model.mesh.generate(3)
    tetrahedra = len(gmsh.model.mesh.getElementsByType(4)[0])
    quadrilaterals = len(gmsh.model.mesh.getElementsByType(3)[0])
    assert quadrilaterals > 0
    assert collect_mesh_stats()["tetrahedra"] == tetrahedra


def _cpw() -> gf.Component:
    """A small GSG electrode, as in the mesh integration tests."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def gsg_electrode(length: float = 500) -> gf.Component:
        layer = gf.gpdk.LAYER.M1
        c = gf.Component()
        r1 = c << gf.c.rectangle((length, 50), centered=True, layer=layer)
        r1.move((0, 38))
        c << gf.c.rectangle((length, 10), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, 50), centered=True, layer=layer)
        r3.move((0, -38))
        for name, x, angle in (("o1", -length / 2, 0), ("o2", length / 2, 180)):
            c.add_port(
                name=name,
                center=(x, 0),
                width=10,
                orientation=angle,
                port_type="electrical",
                layer=layer,
            )
        return c

    return gsg_electrode()


def _stats(tmp_path, order: int) -> dict:
    """Mesh statistics of the CPW at a geometric element order (1 = linear)."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / f"order-{order}"))
    sim.set_geometry(_cpw())
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    for port in ("o1", "o2"):
        sim.add_cpw_port(port, layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
    if order == 1:
        return sim.mesh(preset="coarse").mesh_stats
    return sim.mesh(
        preset="coarse",
        high_order_elements=True,
        high_order_order=order,
        high_order_optimize=False,
    ).mesh_stats


@pytest.fixture(scope="module")
def stats(tmp_path_factory) -> dict[int, dict]:
    tmp_path = tmp_path_factory.mktemp("high-order")
    return {order: _stats(tmp_path, order) for order in (1, 2, 3)}


@pytest.mark.parametrize("order", [2, 3])
def test_a_high_order_mesh_has_the_tetrahedra_of_the_first_order_mesh(
    stats, order
) -> None:
    assert stats[1]["tetrahedra"] > 0
    assert stats[order]["tetrahedra"] == stats[1]["tetrahedra"]
    assert stats[order]["elements"] == stats[1]["elements"]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("measure", ["quality", "sicn", "edge_length"])
def test_a_high_order_mesh_keeps_the_shape_statistics(stats, order, measure) -> None:
    """The corners and the straight edges are those of the first-order mesh."""
    assert stats[order][measure] == stats[1][measure]


@pytest.mark.parametrize("order", [2, 3])
def test_the_dof_estimate_does_not_depend_on_the_element_order(stats, order) -> None:
    """A 3D mesh stays 3D when its elements are raised to a higher order."""
    assert _estimate_dofs(stats[order]) == _estimate_dofs(stats[1])


@pytest.mark.usefixtures("_gmsh_session")
def test_a_planar_mesh_reports_no_tetrahedra() -> None:
    """Surface elements are counted as elements but never as tetrahedra."""
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(2)
    stats = collect_mesh_stats()
    assert stats["elements"] > 0
    assert "tetrahedra" not in stats
    assert "quality" not in stats
