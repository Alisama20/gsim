"""Read a component's layout polygons, whichever gdsfactory built it.

Upstream gdsfactory (kfactory) components expose a klayout layout:
``component.kcl.layout`` with layer indexes, and ``component.get_polygons``
returns klayout polygons keyed by layer index. The mesher is written against
that API.

Other gdsfactory implementations (for example gdsf) are not backed by klayout.
Their ``component.kcl`` carries a database unit only, and
``component.get_polygons()`` returns ``{(layer, datatype): [[(x, y), ...]]}``
with coordinates in um. This module hides the difference: the functions here
return klayout polygons in database units in both cases, so the rest of gsim
keeps working with ``klayout.db`` objects.

When ``component.kcl.layout`` exists, the klayout code path is the one gsim
always used, unchanged.
"""

from __future__ import annotations

from typing import Any

import klayout.db as kdb

DEFAULT_DBU = 0.001
"""Database unit (um) used when a component's ``kcl`` does not say."""

GdsTuple = tuple[int, int]


def _layout(component: Any) -> Any | None:
    """The klayout ``Layout`` behind a component, or None when there is none."""
    return getattr(getattr(component, "kcl", None), "layout", None)


def layout_dbu(component: Any) -> float:
    """Database unit of a component's layout, in um.

    Reads ``component.kcl.layout.dbu`` when there is a klayout layout, else
    ``component.kcl.dbu``, else :data:`DEFAULT_DBU`.
    """
    layout = _layout(component)
    if layout is not None:
        return layout.dbu
    dbu = getattr(getattr(component, "kcl", None), "dbu", None)
    return DEFAULT_DBU if dbu is None else float(dbu)


def default_dbu() -> float:
    """Database unit (um) of the active gdsfactory ``kcl``.

    Upstream gdsfactory has a klayout layout behind ``gf.kcl``; other
    implementations have ``gf.kcl.dbu`` only.
    """
    import gdsfactory as gf

    layout = getattr(gf.kcl, "layout", None)
    return layout.dbu if layout is not None else float(gf.kcl.dbu)


def _gds_key(key: Any) -> GdsTuple:
    """A ``(layer, datatype)`` tuple of ints from a ``get_polygons`` key."""
    message = f"Cannot read a (layer, datatype) pair from the polygon key {key!r}"
    if isinstance(key, str):  # a two-letter name would unpack as a pair
        raise TypeError(message)
    try:
        try:
            layer, datatype = key
        except (TypeError, ValueError):
            layer, datatype = key.layer, key.datatype
        return int(layer), int(datatype)
    except (TypeError, ValueError, AttributeError) as error:
        raise TypeError(message) from error


def _to_klayout_polygon(points: Any, dbu: float) -> kdb.Polygon | None:
    """A klayout polygon (database units) from ``(x, y)`` points in um."""
    dpoints = [kdb.DPoint(float(pt[0]), float(pt[1])) for pt in points]
    if len(dpoints) < 3:
        return None
    return kdb.DPolygon(dpoints).to_itype(dbu)


def _um_polygons(component: Any) -> list[tuple[GdsTuple, list[kdb.Polygon]]]:
    """Polygons of a component that has no klayout layout, in database units."""
    dbu = layout_dbu(component)
    result = []
    for key, shapes in component.get_polygons().items():
        polygons = [_to_klayout_polygon(points, dbu) for points in shapes]
        result.append((_gds_key(key), [p for p in polygons if p is not None]))
    return result


def layer_polygons(component: Any) -> list[tuple[GdsTuple, list[kdb.Polygon]]]:
    """The polygons of a component as ``(gds_tuple, polygons)`` entries.

    With a klayout layout, one entry per valid layer index in the order of
    ``component.get_polygons(layers=...)``. Named aliases share a GDS tuple,
    so a tuple can come up in more than one entry; ``polygons_by_gds`` joins
    them. Without one, one entry per ``(layer, datatype)``.

    Returns:
        Entries of GDS ``(layer, datatype)`` and klayout polygons in
        database units (``layout_dbu(component)``).
    """
    layout = _layout(component)
    if layout is None:
        return _um_polygons(component)
    by_index = component.get_polygons(layers=list(layout.layer_indexes()))
    entries = []
    for index, polygons in by_index.items():
        if not layout.is_valid_layer(index):
            continue
        info = layout.get_info(index)
        entries.append(((info.layer, info.datatype), polygons))
    return entries


def polygons_by_gds(component: Any) -> dict[GdsTuple, list[kdb.Polygon]]:
    """Polygons of a component grouped by GDS ``(layer, datatype)``.

    Args:
        component: A gdsfactory component, from upstream gdsfactory or from an
            implementation whose ``get_polygons()`` returns um coordinates.

    Returns:
        ``{(layer, datatype): [klayout.db.Polygon, ...]}`` in database units
        of ``layout_dbu(component)``. A tuple that is registered but empty is
        present with an empty list in the upstream case.
    """
    grouped: dict[GdsTuple, list[kdb.Polygon]] = {}
    for gds_tuple, polygons in layer_polygons(component):
        grouped.setdefault(gds_tuple, []).extend(polygons)
    return grouped


__all__ = [
    "DEFAULT_DBU",
    "default_dbu",
    "layer_polygons",
    "layout_dbu",
    "polygons_by_gds",
]
