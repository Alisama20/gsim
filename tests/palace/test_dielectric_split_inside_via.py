"""A dielectric boundary inside a via must not break the conductor shells (#424).

Volumetric conductors become shells (``m1_xy`` ...) whose faces are shared with
the via volumes where a via touches the conductor. The dim=3 fragment renumbers
those faces when a dielectric boundary cuts the via. The shells have to follow.
"""

from __future__ import annotations

import logging
from functools import cache
from itertools import pairwise

import gdsfactory as gf
import gmsh
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace.mesh.gmsh_utils import (
    Entity,
    _refresh_2d,
    _snapshot_2d,
    run_boolean_pipeline,
)

# --- full mesh of a small metal / via stack -------------------------------

_VIA_FOOTPRINT = (5.0, 3.0, 8.0, 6.0)
_METAL_FOOTPRINTS = {
    1: (0.0, 0.0, 20.0, 10.0),
    2: (3.0, 1.0, 12.0, 8.0),
    3: (4.0, 2.0, 10.0, 7.0),
}


def _rect(box):
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _layer(name, gds, z0, z1, kind, material):
    return Layer(
        name=name,
        gds_layer=(gds, 0),
        zmin=z0,
        zmax=z1,
        thickness=z1 - z0,
        material=material,
        layer_type=kind,
    )


def _mesh(output_dir, metals, split_z):
    """Mesh ``metals`` conductors joined by vias.

    Metal ``k`` spans z = 2(k-1)..2k-1 um and via ``k`` the next micrometer above it.

    The SiO2 is one region (``split_z`` empty) or is cut at every ``split_z``.
    """
    gf.gpdk.PDK.activate()
    component = gf.Component()
    layers = {}
    for k in range(1, metals + 1):
        component.add_polygon(_rect(_METAL_FOOTPRINTS[k]), layer=(k, 0))
        layers[f"m{k}"] = _layer(
            f"m{k}", k, 2.0 * (k - 1), 2.0 * (k - 1) + 1.0, "conductor", "aluminum"
        )
        # A via on top of every metal that has another metal above it; a single
        # metal carries one via on its own.
        if k < metals or metals == 1:
            component.add_polygon(_rect(_VIA_FOOTPRINT), layer=(10 + k, 0))
            layers[f"v{k}"] = _layer(
                f"v{k}", 10 + k, 2.0 * k - 1.0, 2.0 * k, "via", "tungsten"
            )
    component.add_port(
        "o1",
        center=(0, 5),
        width=4,
        orientation=180,
        layer=(1, 0),
        port_type="electrical",
    )
    top = 2.0 if metals == 1 else 2.0 * metals - 1.0
    edges = [-2.0, *sorted(split_z), top + 5.0]
    stack = LayerStack(
        layers=layers,
        materials={
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
            "tungsten": {"type": "conductor", "conductivity": 1.8e7},
            "sio2": {"type": "dielectric", "permittivity": 3.9},
        },
        dielectrics=[
            {"name": f"ox{i}", "zmin": a, "zmax": b, "material": "sio2"}
            for i, (a, b) in enumerate(pairwise(edges))
        ],
    )
    sim = DrivenSim()
    sim.set_output_dir(str(output_dir))
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3, margin_y=3, z_above=3, z_below=0)
    sim.add_port("o1", layer="m1", length=2.0)
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=3)
    sim.mesh(refined_mesh_size=2.0, max_mesh_size=10.0, verbose=False)
    return sim


def _shell_areas(sim):
    """Return ``({name: area of the shell group}, tetrahedron count)`` (um^2)."""
    groups = sim.mesh_groups["conductor_surfaces"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(sim.mesh_path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        position = np.zeros((int(node_tags.max()) + 1, 3))
        position[node_tags.astype(int)] = coords.reshape(-1, 3)
        areas = {}
        for name, info in groups.items():
            if not name.endswith("_xy"):
                continue
            total = 0.0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(2, info["phys_group"]):
                _, _, nodes = gmsh.model.mesh.getElements(2, int(entity))
                for block in nodes:
                    tri = position[np.asarray(block, dtype=int).reshape(-1, 3)]
                    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    total += 0.5 * float(np.linalg.norm(cross, axis=1).sum())
            areas[name] = total
        tets = int(sum(len(t) for t in gmsh.model.mesh.getElements(3)[1]))
        return areas, tets
    finally:
        gmsh.finalize()


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Shell areas of the same stack with one dielectric region, per metal count."""

    @cache
    def build(metals):
        sim = _mesh(tmp_path_factory.mktemp(f"ref{metals}"), metals, [])
        return _shell_areas(sim)[0]

    return build


@pytest.mark.parametrize(
    ("metals", "split_z"),
    [
        (1, [1.5]),
        (2, [1.5]),
        (2, [1.01, 1.99]),
        (3, [1.5, 3.5]),
    ],
)
def test_dielectric_split_inside_via_keeps_shells(tmp_path, reference, metals, split_z):
    """The mesh builds and every ``_xy`` shell has the area it has without a split."""
    expected = reference(metals)
    assert set(expected) == {f"m{k}_xy" for k in range(1, metals + 1)}
    assert all(area > 0 for area in expected.values())

    sim = _mesh(tmp_path, metals, split_z)
    areas, tets = _shell_areas(sim)

    assert tets > 0
    assert areas == pytest.approx(expected, rel=1e-6)


# --- the pipeline on a hand-built model -----------------------------------


def _conductor_shell_and_via(kernel, conductor_box, via_box):
    """Make a via volume on a conductor and turn the conductor into a shell.

    Like ``add_metals``: both volumes are fragmented so that they share the
    contact face, then the conductor volume is removed and its faces are kept.

    Returns the via volume tag and the tags of the conductor's faces.
    """
    conductor = kernel.addBox(*conductor_box)
    via = kernel.addBox(*via_box)
    _, out_map = kernel.fragment([(3, conductor)], [(3, via)])
    kernel.synchronize()
    (conductor_dimtag,) = out_map[0]
    (via_dimtag,) = out_map[1]
    boundary = gmsh.model.getBoundary(
        [conductor_dimtag], combined=False, oriented=False, recursive=False
    )
    via_faces = {
        t
        for _, t in gmsh.model.getBoundary(
            [via_dimtag], combined=False, oriented=False, recursive=False
        )
    }
    shell = [t for _, t in boundary]
    assert via_faces & set(shell), "the via must share a face with the shell"
    kernel.remove([conductor_dimtag])
    kernel.synchronize()
    return via_dimtag[1], shell


def _area(kernel, entity):
    return sum(kernel.getMass(d, t) for d, t in entity.dimtags if d == 2)


def test_pipeline_follows_shell_face_of_split_via(kernel):
    """A via cut by a dielectric boundary: its footprint stays in the shell."""
    # conductor 4 x 4 x 1; via 1 x 1, z 1..3 on top of it; oxide cut at z=2
    via, shell_faces = _conductor_shell_and_via(
        kernel, (0, 0, 0, 4, 4, 1), (1, 1, 1, 1, 1, 2)
    )
    lower = kernel.addBox(-2, -2, -2, 8, 8, 4)  # oxide up to z=2
    upper = kernel.addBox(-2, -2, 2, 8, 8, 4)  # oxide from z=2
    kernel.synchronize()
    entities = [
        Entity("via", 3, 1, [via]),
        Entity("lower", 3, 2, [lower]),
        Entity("upper", 3, 2, [upper]),
        Entity("m1", 2, 1, shell_faces),
    ]
    shell_area = _area(kernel, entities[3])
    assert shell_area == pytest.approx(48.0)

    pg_map = run_boolean_pipeline(entities)

    shell = entities[3]
    assert _area(kernel, shell) == pytest.approx(shell_area)
    assert len(shell.dimtags) == len(shell_faces)
    assert "m1" in pg_map
    for dim, tag in shell.dimtags:
        kernel.getBoundingBox(dim, tag)  # raises for a stale tag


def test_pipeline_follows_split_shell_face(kernel):
    """A shell face that the fragment cuts in two is replaced by both parts."""
    # the 4 x 2 contact face of block and conductor is the shell; the two
    # oxide blocks below it end at x=2, which cuts the block's bottom face
    upper, shell_faces = _conductor_shell_and_via(
        kernel, (0, 0, 0, 4, 2, 1), (0, 0, 1, 4, 2, 1)
    )
    (contact,) = [
        t for t in shell_faces if np.isclose(kernel.getCenterOfMass(2, t)[2], 1.0)
    ]
    assert kernel.getMass(2, contact) == pytest.approx(8.0)
    left = kernel.addBox(0, 0, 0, 2, 2, 1)
    right = kernel.addBox(2, 0, 0, 2, 2, 1)
    kernel.synchronize()
    entities = [
        Entity("upper", 3, 1, [upper]),
        Entity("left", 3, 2, [left]),
        Entity("right", 3, 2, [right]),
        Entity("shell", 2, 1, [contact]),
    ]
    run_boolean_pipeline(entities)

    shell = entities[3]
    assert len(shell.dimtags) == 2
    assert _area(kernel, shell) == pytest.approx(8.0)


# --- _refresh_2d on its own ------------------------------------------------


def test_refresh_leaves_unchanged_faces_alone(kernel):
    """Nothing changed: the tags stay and nothing is looked up."""
    a = kernel.addRectangle(0, 0, 0, 1, 1)
    b = kernel.addRectangle(3, 0, 0, 1, 1)
    kernel.synchronize()
    entities = [Entity("a", 2, 1, [a]), Entity("b", 2, 2, [b])]
    snapshot = _snapshot_2d(entities)
    assert set(snapshot) == {a, b}
    _refresh_2d(entities, snapshot, [])
    assert entities[0].dimtags == [(2, a)]
    assert entities[1].dimtags == [(2, b)]


def test_refresh_ignores_non_surface_entities(kernel):
    """Only dim=2 entities are snapshotted."""
    box = kernel.addBox(0, 0, 0, 1, 1, 1)
    kernel.synchronize()
    entities = [Entity("v", 3, 1, [box])]
    assert _snapshot_2d(entities) == {}
    _refresh_2d(entities, {}, [])
    assert entities[0].dimtags == [(3, box)]


def test_refresh_drops_lost_face_with_warning(kernel, caplog):
    """A face that is gone and has no counterpart is dropped and reported."""
    a = kernel.addRectangle(0, 0, 0, 1, 1)
    keep = kernel.addRectangle(3, 0, 0, 1, 1)
    kernel.synchronize()
    entities = [Entity("a", 2, 1, [a, keep])]
    snapshot = _snapshot_2d(entities)
    kernel.remove([(2, a)], recursive=True)
    kernel.synchronize()
    with caplog.at_level(logging.WARNING, logger="gsim.palace.mesh.gmsh_utils"):
        _refresh_2d(entities, snapshot, [])
    assert entities[0].dimtags == [(2, keep)]
    assert "was lost" in caplog.text


def test_refresh_does_not_trust_a_reused_tag(kernel, caplog):
    """A tag that is valid but now names a different face is not kept."""
    a = kernel.addRectangle(0, 0, 0, 1, 1)
    kernel.synchronize()
    entities = [Entity("a", 2, 1, [a])]
    snapshot = _snapshot_2d(entities)
    kernel.remove([(2, a)], recursive=True)
    other = kernel.addRectangle(5, 5, 0, 2, 2, tag=a)
    kernel.synchronize()
    assert other == a
    with caplog.at_level(logging.WARNING, logger="gsim.palace.mesh.gmsh_utils"):
        _refresh_2d(entities, snapshot, [])
    assert entities[0].dimtags == []
    assert "was lost" in caplog.text
