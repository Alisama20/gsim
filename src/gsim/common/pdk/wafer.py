"""Blanket films declared on the PDK wafer layer.

gdsfactory PDKs such as the generic PDK declare their substrate, buried oxide
and cladding as ``LayerLevel`` entries on the ``WAFER`` layer, which no cell
draws. Their material fills the whole chip, so they are blanket films exactly
like levels marked ``background=True``. This module marks them as such on a
copy of the stack, leaving the PDK's own stack untouched.
"""

from __future__ import annotations

from typing import Any

from gsim.common.stack._layer_utils import (
    _tuple_from_layer_expression,
    get_gds_layer_tuple,
)

#: gdsfactory's wafer layer, used when a PDK's layer map has no ``WAFER`` entry.
GDSFACTORY_WAFER_LAYER = (999, 0)


def wafer_layer_tuple(pdk: Any | None) -> tuple[int, int]:
    """Return the GDS tuple of the PDK's wafer layer.

    The PDK's own ``WAFER`` layer is used when its layer map declares one;
    otherwise the gdsfactory convention ``(999, 0)`` applies.
    """
    wafer = getattr(getattr(pdk, "layers", None), "WAFER", None)
    parsed = _tuple_from_layer_expression(wafer) if wafer is not None else None
    return parsed if parsed is not None else GDSFACTORY_WAFER_LAYER


def is_wafer_level(level: Any, wafer_layer: tuple[int, int]) -> bool:
    """Return whether *level* is a plain, unsized level on the wafer layer."""
    from gdsfactory.technology import LogicalLayer

    layer = getattr(level, "layer", None)
    if not isinstance(layer, LogicalLayer):
        return False  # Boolean expressions are patterned masks.
    if any((*layer.sizings_xoffsets, *layer.sizings_yoffsets)):
        return False
    return get_gds_layer_tuple(level) == wafer_layer


def _has_background_field(level: Any) -> bool:
    """Whether the level type has a ``background`` field.

    gdsfactory's ``LayerLevel`` is a pydantic model and lists its fields; gdsf's
    builtin ``LayerLevel`` has no ``model_fields`` but a ``background`` getter.
    """
    fields = getattr(type(level), "model_fields", None)
    if fields is not None:
        return "background" in fields
    return hasattr(level, "background")


def _background_copy(level: Any) -> Any:
    """Return a copy of *level* with ``background=True``.

    gdsf's builtin ``LayerLevel`` has no ``model_copy``, no setters, and no
    getters for ``width_to_z``, ``bias`` and ``z_to_bias``: the copy is built
    again from the fields it does expose, so those three keep their defaults.
    A wafer film has no sidewall profile, which makes the loss immaterial.
    """
    model_copy = getattr(level, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"background": True})
    fields: dict[str, Any] = {
        "layer": level.layer,
        "thickness": level.thickness,
        "zmin": level.zmin,
        "material": level.material,
        "mesh_order": level.mesh_order,
        "sidewall_angle": level.sidewall_angle,
        "background": True,
        "background_exclude_layers": list(level.background_exclude_layers),
    }
    if level.derived_layer is not None:
        fields["derived_layer"] = level.derived_layer
    if level.info is not None:
        fields["info"] = level.info
    if level.name is not None:
        fields["name"] = level.name
    return type(level)(**fields)


def _stack_with_levels(layer_stack: Any, updated: dict[str, Any]) -> Any:
    """Return a copy of *layer_stack* in which the levels in *updated* replace theirs."""
    if hasattr(type(layer_stack), "model_fields"):
        return layer_stack.model_copy(
            update={"layers": {**layer_stack.layers, **updated}}
        )
    # gdsf: ``model_copy()`` takes no ``update``; ``layers`` is a live mapping.
    copied = layer_stack.model_copy()
    for key, level in updated.items():
        copied.layers[key] = level
    return copied


def with_wafer_films(layer_stack: Any, pdk: Any | None) -> Any:
    """Return *layer_stack* with wafer-layer levels marked ``background=True``.

    The input stack is never modified: when a level changes, the result is a
    shallow copy holding copied levels. Stacks of gdsfactory versions without
    the ``background`` field are returned unchanged.
    """
    wafer_layer = wafer_layer_tuple(pdk)
    updated = {
        key: _background_copy(level)
        for key, level in layer_stack.layers.items()
        if _has_background_field(level)
        and not level.background
        and is_wafer_level(level, wafer_layer)
    }
    if not updated:
        return layer_stack
    return _stack_with_levels(layer_stack, updated)


__all__ = [
    "GDSFACTORY_WAFER_LAYER",
    "is_wafer_level",
    "wafer_layer_tuple",
    "with_wafer_films",
]
