"""A component with the derived layers of a layer stack evaluated.

gdsfactory's ``LayerStack.get_component_with_derived_layers(component)`` returns
a component in which every stack level has been evaluated (boolean expressions
included) and drawn on the level's ``derived_layer``. That component is what
gdsfactory's own 3D view extrudes. This module gives the same component for
every gdsfactory implementation:

* upstream gdsfactory has the method, and it is used as is;
* gdsf (no klayout cell behind its components) has the function
  ``gdsfactory.technology.layer_stack.get_component_with_derived_layers(
  component, layer_stack)``, which fails on levels without a ``derived_layer``
  and combines polygons of the same layer only. Here the levels are evaluated
  with :func:`gsim.common.polygon.evaluate_layer_expression` instead.
"""

from __future__ import annotations

from typing import Any

from gsim.common.layout_access import layout_dbu
from gsim.common.polygon import _source_regions, evaluate_layer_expression
from gsim.common.stack._layer_utils import _tuple_from_layer_expression


def derived_layer_regions(
    layer_stack: Any, component: Any
) -> dict[tuple[int, int], Any]:
    """Evaluate every level of a stack on a component, as klayout regions.

    Each level is evaluated from its ``layer`` expression and added to the
    GDS layer of its ``derived_layer``, or of its own layer when it has none
    (then it must be a plain layer, as in gdsfactory). Levels that share a
    destination layer are united, as gdsfactory does.

    Args:
        layer_stack: A gdsfactory ``LayerStack``.
        component: A gdsfactory component.

    Returns:
        Merged regions by destination ``(layer, datatype)``, in the database
        units of the component.

    Raises:
        ValueError: A level without ``derived_layer`` whose ``layer`` is an
            expression, or a destination layer with no GDS pair.
    """
    import klayout.db as kdb

    sources = _source_regions(component)
    destinations: dict[tuple[int, int], Any] = {}
    for key, level in layer_stack.layers.items():
        declared = getattr(level, "derived_layer", None)
        if declared is None:
            if hasattr(level.layer, "layer1"):
                raise ValueError(
                    f"LayerStack level {key!r} has no derived_layer, so its layer "
                    "must be a plain layer, not a boolean expression."
                )
            declared = level.layer
        destination = _tuple_from_layer_expression(declared)
        if destination is None:
            raise ValueError(f"Cannot read a GDS layer from {declared!r} ({key!r}).")
        region = evaluate_layer_expression(level.layer, sources)
        destinations[destination] = destinations.get(destination, kdb.Region()) + region
    return {gds: region.merged() for gds, region in destinations.items()}


def get_component_with_derived_layers(layer_stack: Any, component: Any) -> Any:
    """Return *component* with the derived layers of *layer_stack* evaluated.

    Upstream gdsfactory's ``LayerStack.get_component_with_derived_layers`` is
    used when the stack has it. Otherwise a new component is built from
    :func:`derived_layer_regions`; its polygons are in micrometers, and the
    ports of *component* are copied when the implementation can.

    Args:
        layer_stack: A gdsfactory ``LayerStack``.
        component: A gdsfactory component.

    Returns:
        A component with one layer per destination layer of the stack.
    """
    method = getattr(layer_stack, "get_component_with_derived_layers", None)
    if callable(method):
        return method(component)

    import gdsfactory as gf

    dbu = layout_dbu(component)
    derived = gf.Component()
    for gds, region in derived_layer_regions(layer_stack, component).items():
        for polygon in region.each_merged():
            # A polygon with holes is written as one ring (keyholes), as in GDS.
            ring = polygon.to_simple_polygon()
            points = [(point.x * dbu, point.y * dbu) for point in ring.each_point()]
            derived.add_polygon(points, layer=gds)
    if hasattr(derived, "add_ports") and getattr(component, "ports", None) is not None:
        derived.add_ports(component.ports)
    return derived


__all__ = ["derived_layer_regions", "get_component_with_derived_layers"]
