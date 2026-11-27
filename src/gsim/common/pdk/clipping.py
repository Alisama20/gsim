"""Diagnostics for explicit material windows, including sidewall footprints."""

from __future__ import annotations

from math import radians, tan
from typing import Any

from gsim.common.stack._layer_utils import (
    level_bias,
    level_width_to_z,
    level_z_to_bias,
)

TOLERANCE_UM = 1e-9


def material_clipping_axes(
    geometry: Any,
    level: Any,
    x_bounds: tuple[float, float] | None,
    y_bounds: tuple[float, float] | None,
    z_bounds: tuple[float, float] | None,
) -> tuple[str, ...]:
    """Return explicit axes that remove nonzero material volume.

    Blanket films have no finite XY footprint; their vertical truncation still
    matters. For supported sidewalls, inspect the widest extrusion section.
    """
    if (
        all(bounds is None for bounds in (x_bounds, y_bounds, z_bounds))
        or geometry.is_empty
        or not level.material
        or level.thickness == 0
    ):
        return ()
    xmin, ymin, xmax, ymax = geometry.bounds
    if (
        (x_bounds is not None or y_bounds is not None)
        and level_bias(level) in (None, 0, 0.0)
        and level_z_to_bias(level) is None
    ):
        slope = tan(radians(float(level.sidewall_angle))) * abs(float(level.thickness))
        width_to_z = level_width_to_z(level)
        offset = max(0, width_to_z * slope, (width_to_z - 1) * slope)
        if offset:
            xmin, ymin, xmax, ymax = geometry.buffer(offset, join_style=2).bounds
    z_start = float(level.zmin)
    z_end = z_start + float(level.thickness)
    intervals = ((xmin, xmax), (ymin, ymax), (min(z_start, z_end), max(z_start, z_end)))
    axes = []
    for axis, interval, bounds in zip(
        "xyz", intervals, (x_bounds, y_bounds, z_bounds), strict=True
    ):
        if bounds is None or (axis != "z" and getattr(level, "background", False)):
            continue
        if (
            interval[0] < bounds[0] - TOLERANCE_UM
            or interval[1] > bounds[1] + TOLERANCE_UM
        ):
            axes.append(axis)
    return tuple(axes)
