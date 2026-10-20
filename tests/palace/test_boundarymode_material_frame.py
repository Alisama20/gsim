"""A BoundaryMode config writes material tensors in the 2D mesh frame.

The native 2D mesh of an ``x=...`` cross-section has the layout's y on mesh x,
z on mesh y and the propagation direction x on mesh z; for ``y=...`` it is
(x, z, y). Palace reads ``Permittivity`` in the mesh frame, so a layout-axes
``[xx, yy, zz]`` has to be re-ordered or an anisotropic material gets the
wrong component along each axis.
"""

from __future__ import annotations

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import BoundaryModeSim
from gsim.palace.mesh.config_generator import _to_section_frame

EPS = [11.0, 22.0, 33.0]  # layout (xx, yy, zz), all different


@pytest.mark.parametrize(
    ("axis", "rows"),
    [
        # rows of R: the layout axis behind mesh x, y, z
        ("x", [(0, 1, 0), (0, 0, 1), (1, 0, 0)]),
        ("y", [(1, 0, 0), (0, 0, 1), (0, 1, 0)]),
    ],
)
def test_reordering_is_the_tensor_in_the_mesh_axes(axis, rows):
    rotation = np.array(rows, dtype=float)
    expected = np.diag(rotation @ np.diag(EPS) @ rotation.T)
    got = _to_section_frame({"permittivity": EPS}, axis)["permittivity"]
    assert got == pytest.approx(expected)


def test_scalars_other_planes_and_the_input_are_untouched():
    props = {"permittivity": 3.9, "loss_tangent": [1e-3, 2e-3, 3e-3]}
    assert _to_section_frame(props, "x")["permittivity"] == 3.9
    assert _to_section_frame(props, "x")["loss_tangent"] == [2e-3, 3e-3, 1e-3]
    assert props["loss_tangent"] == [1e-3, 2e-3, 3e-3]
    assert _to_section_frame(props, "z") is props


def test_rotated_material_axes_are_refused_not_guessed():
    props = {"permittivity": EPS, "material_axes": [[0, 1, 0], [1, 0, 0], [0, 0, 1]]}
    with pytest.raises(ValueError, match="material_axes"):
        _to_section_frame(props, "x")
    identity = {"permittivity": EPS, "material_axes": np.eye(3).tolist()}
    assert _to_section_frame(identity, "x")["permittivity"] == [22.0, 33.0, 11.0]


def _config(tmp_path, cut):
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    comp << gf.c.rectangle((40.0, 10.0), centered=True, layer=(1, 0))
    stack = LayerStack(pdk_name="aniso", units="um")
    stack.materials = {
        "aniso": {"permittivity": EPS},
        "sio2": {"permittivity": 3.9},
    }
    stack.layers = {
        "film": Layer(
            name="film",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=1.0,
            thickness=1.0,
            material="aniso",
            layer_type="dielectric",
        )
    }
    stack.dielectrics = [{"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "sio2"}]
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(comp)
    sim.set_stack(stack)
    sim.set_cross_section(cut)
    sim.set_boundary_mode(freq=10e9, num_modes=1)
    sim.set_airbox(margin_x=0.0, margin_y=5.0, z_above=3.0, z_below=0.0)
    sim.mesh(preset="coarse", verbose=False)
    sim.write_config(photonic=True)
    config = json.loads((tmp_path / "config.json").read_text())
    return sim, config


@pytest.mark.parametrize(
    ("cut", "expected", "mesh_first_axis_span"),
    [
        ("x=0", [22.0, 33.0, 11.0], 10.0),  # mesh x spans the layout's y extent
        ("y=0", [11.0, 33.0, 22.0], 40.0),  # mesh x spans the layout's x extent
    ],
)
def test_config_tensor_follows_the_mesh_frame(
    tmp_path, cut, expected, mesh_first_axis_span
):
    sim, config = _config(tmp_path, cut)
    points = np.asarray(meshio.read(sim.mesh_path).points)
    # the frame the tensor has to match: film width along mesh x
    assert points[:, 0].max() - points[:, 0].min() >= mesh_first_axis_span
    tensors = [
        m["Permittivity"]
        for m in config["Domains"]["Materials"]
        if isinstance(m["Permittivity"], list)
    ]
    assert tensors == [expected]
    # the stack the caller handed in is not rewritten
    assert sim.stack.materials["aniso"]["permittivity"] == EPS
