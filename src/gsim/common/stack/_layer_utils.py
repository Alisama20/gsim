"""Shared layer classification and GDS-layer extraction helpers.

Used by both ``extractor`` and ``visualization`` modules.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from gdsfactory.technology import LayerLevel

logger = logging.getLogger(__name__)


def _tuple_from_layer_string(layer: str) -> tuple[int, int] | None:
    """Resolve a layer written as ``"layer/datatype"`` or as a layer name.

    gdsf layer levels refer to their layer by name (``LogicalLayer.layer`` is
    ``"M3"``), not by a GDS pair, so a name is looked up through the active
    PDK's layer map. Returns ``None`` when the string is neither form.
    """
    if "/" in layer:
        parts = layer.split("/")
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return None
    try:
        import gdsfactory as gf

        pair = tuple(gf.get_layer_tuple(layer))
    except Exception:  # noqa: BLE001 - unknown name, no active PDK, API differences
        return None
    if len(pair) == 2 and all(isinstance(v, int) for v in pair):
        return (int(pair[0]), int(pair[1]))
    return None


def _layer_name(layer: Any) -> str | None:
    """Return the layer name held by *layer* (a str or ``.layer`` str), if any."""
    if isinstance(layer, str):
        return layer
    inner = getattr(layer, "layer", None)
    return inner if isinstance(inner, str) else None


def _tuple_from_layer_expression(layer: Any) -> tuple[int, int] | None:
    """Recursively extract a concrete ``(layer, datatype)`` from an expression.

    gdsfactory represents derived layers as objects with ``layer1``/``layer2``
    operands (and sometimes a nested ``layer`` attribute).  Resolve the first
    concrete leaf so metadata (drawn flags, source matching) remains stable even
    when the boolean expression itself has no single GDS tuple.
    """
    if isinstance(layer, str):
        return _tuple_from_layer_string(layer)

    if isinstance(layer, tuple) and len(layer) == 2:
        return (int(layer[0]), int(layer[1]))

    # LayerMap enums can subclass int (so their numeric value is a map index,
    # not the GDS layer), but expose the real pair via layer/datatype.
    inner_layer = getattr(layer, "layer", None)
    inner_datatype = getattr(layer, "datatype", None)
    if isinstance(inner_layer, int) and isinstance(inner_datatype, int):
        return (int(inner_layer), int(inner_datatype))

    if isinstance(layer, int) and not isinstance(layer, bool):
        return (int(layer), 0)

    nested = getattr(layer, "layer", None)
    if nested is not None and nested is not layer:
        result = _tuple_from_layer_expression(nested)
        if result is not None:
            return result

    for attribute in ("layer1", "layer2"):
        operand = getattr(layer, attribute, None)
        if operand is not None:
            result = _tuple_from_layer_expression(operand)
            if result is not None:
                return result

    value = getattr(layer, "value", None)
    if value is not None and value is not layer:
        return _tuple_from_layer_expression(value)

    return None


def get_gds_layer_tuple(layer_level: LayerLevel) -> tuple[int, int] | None:
    """Extract GDS layer tuple (layer, datatype) from a gdsfactory LayerLevel.

    Returns ``None`` when the layer cannot be parsed (callers decide fallback).

    When the primary ``layer`` is a ``DerivedLayer`` (e.g. ``WG - GRA`` in
    cspdk), falls back to a concrete leaf of the boolean expression and then to
    ``derived_layer``, which typically holds the simple GDS layer used for
    polygon extraction.
    """
    layer: Any = layer_level.layer

    if isinstance(layer, tuple):
        return (int(layer[0]), int(layer[1]))

    if isinstance(layer, int):
        return (int(layer), 0)

    name = _layer_name(layer)
    if name is not None:
        named = _tuple_from_layer_string(name)
        if named is not None:
            return named

    if hasattr(layer, "layer"):
        inner = layer.layer
        if isinstance(inner, tuple) and len(inner) == 2:
            return (int(inner[0]), int(inner[1]))
        if hasattr(inner, "layer") and hasattr(inner, "datatype"):
            return (int(inner.layer), int(inner.datatype))  # ty: ignore[invalid-argument-type]
        # LayerMap enums can expose the concrete GDS pair through iteration even
        # though ``int(enum)`` returns an unrelated map index (e.g. Cornerstone
        # WG is enum value 5 but GDS (3, 0)).
        try:
            # A str would iterate into its characters ("M3" -> ("M", "3")).
            pair = None if isinstance(inner, str) else tuple(inner)
        except TypeError:
            pair = None
        if pair is not None and len(pair) == 2:
            return (int(pair[0]), int(pair[1]))
        if isinstance(inner, int):
            datatype = getattr(layer, "datatype", 0)
            return (int(inner), int(datatype) if datatype else 0)
        if hasattr(inner, "layer"):
            innermost = inner.layer
            if isinstance(innermost, int):
                datatype = getattr(inner, "datatype", 0)
                return (int(innermost), int(datatype) if datatype else 0)

    if hasattr(layer, "layer") and hasattr(layer, "datatype"):
        return (int(layer.layer), int(layer.datatype))  # ty: ignore[invalid-argument-type]

    if hasattr(layer, "value"):
        if isinstance(layer.value, tuple):
            return (int(layer.value[0]), int(layer.value[1]))  # ty: ignore[invalid-argument-type]
        if isinstance(layer.value, int):
            return (int(layer.value), 0)

    if isinstance(layer, str):
        if "/" in layer:
            parts = layer.split("/")
            return (int(parts[0]), int(parts[1]))
        return None

    try:
        return (int(layer), 0)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        pass

    # Fallback: DerivedLayer (e.g. cspdk core = WG - GRA).  The declared
    # derived_layer is the authoritative concrete mask for the level (core ->
    # WG, grating -> GRA), which also keeps ports unambiguous when two derived
    # levels share boolean operands.
    derived = getattr(layer_level, "derived_layer", None)
    if derived is not None and derived is not layer:
        result = _tuple_from_layer_expression(derived)
        if result is not None:
            return result

    result = _tuple_from_layer_expression(layer)
    if result is not None:
        return result

    logger.warning("Could not parse layer %s", layer)
    return None


def level_width_to_z(level: Any) -> float:
    """Return ``LayerLevel.width_to_z``, ``0.0`` when the level has no such field.

    gdsfactory's ``LayerLevel`` always has it. gdsf's builtin ``LayerLevel``
    accepts ``width_to_z`` in its constructor but has no getter for it.
    """
    return float(getattr(level, "width_to_z", 0.0) or 0.0)


def level_bias(level: Any) -> Any:
    """Return ``LayerLevel.bias``, ``None`` when the level has no such field."""
    return getattr(level, "bias", None)


def level_z_to_bias(level: Any) -> Any:
    """Return ``LayerLevel.z_to_bias``, ``None`` when the level has no such field."""
    return getattr(level, "z_to_bias", None)


def classify_layer_type(
    layer_name: str,
    material: str | None = None,
) -> Literal["conductor", "via", "dielectric", "substrate"]:
    """Classify a layer as conductor, via, dielectric, or substrate.

    When *material* is provided, the materials DB is consulted for additional
    classification hints (conductor vs dielectric).  When ``None``, only
    name-based heuristics are used.
    """
    name_lower = layer_name.lower()

    if "via" in name_lower or "cont" in name_lower or name_lower == "vmim":
        return "via"

    if any(
        m in name_lower for m in ["metal", "topmetal", "m1", "m2", "m3", "m4", "m5"]
    ):
        return "conductor"

    if "substrate" in name_lower or name_lower == "sub":
        return "substrate"

    # Material-based lookup (only when caller supplies material)
    if material is not None:
        from gsim.common.stack.materials import get_material_properties

        props = get_material_properties(material)
        if props:
            cond = props.conductivity
            if isinstance(cond, list):
                cond = cond[0]
            if cond is not None and cond >= 1e4:
                return "conductor"
            if cond is not None and cond > 0 and "substrate" in name_lower:
                return "substrate"

        material_lower = material.lower()
        if material_lower in [
            "aluminum",
            "copper",
            "tungsten",
            "gold",
            "al",
            "cu",
            "w",
        ]:
            return "conductor"

    if "poly" in name_lower or "active" in name_lower:
        return "conductor"

    return "dielectric"
