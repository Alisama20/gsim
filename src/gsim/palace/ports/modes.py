"""Several wave-port modes on one port face (the multimode wave port).

Palace takes one ``WavePort`` entry per (face, mode): every entry runs its own
2D mode solve on the face and picks eigenvector ``Mode``. Entries may share a
boundary attribute, but only one of them may be ``"Active"`` (the one that
terminates the face), and the active entry must have the lowest ``Index`` of
the face, because Palace walks the entries in ``Index`` order and refuses an
active port whose attribute is already taken. The other entries are
observation-only: they are measured, and they can be excited, but they do not
load the face.

The config generator writes the entries of a multimode port and, next to
``config.json``, a sidecar ``port_modes.json`` that records for every Palace
``Index`` which gsim port (face) and mode it stands for and whether it is the
active one. The wave-port log checks and the later assembly of modal
S-parameters read the sidecar, because the S-parameter columns carry only the
Palace index.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PORT_MODES_FILENAME = "port_modes.json"
"""Name of the sidecar next to ``config.json``."""

FACE_POSITION_TOL_UM = 1e-6
"""Two ports whose centres agree to this distance and whose orientations agree
are on the same face."""

__all__ = [
    "FACE_POSITION_TOL_UM",
    "PORT_MODES_FILENAME",
    "PortModeEntry",
    "check_port_mode_entries",
    "faces_by_index",
    "read_port_modes",
    "same_face_pairs",
    "write_port_modes",
]


@dataclass(frozen=True)
class PortModeEntry:
    """One Palace ``WavePort`` entry of a multimode port.

    Attributes:
        index: The Palace ``Index`` of the entry (the column index in
            ``port-S.csv``).
        port: Name of the gsim port, i.e. of the face the entry sits on.
        portnumber: 1-based position of the gsim port, which is also the
            ``Index`` of its active entry.
        mode: The Palace ``Mode`` (eigenvector number) the entry follows.
        active: Whether the entry terminates the face.
        excited: Whether the entry has an ``Excitation`` of its own.
        attributes: Boundary attributes of the face.
    """

    index: int
    port: str
    portnumber: int
    mode: int
    active: bool
    excited: bool
    attributes: tuple[int, ...] = ()


def check_port_mode_entries(entries: Sequence[PortModeEntry]) -> None:
    """Refuse a set of entries Palace would reject or that gsim cannot decode.

    Raises:
        ValueError: For a repeated ``Index``; for a face (port) with no active
            entry or more than one; for an active entry that does not have the
            lowest ``Index`` of its face (Palace visits entries in ``Index``
            order and refuses an active port on an attribute already taken);
            or for a mode that appears twice on one face.
    """
    seen: set[int] = set()
    by_face: dict[int, list[PortModeEntry]] = defaultdict(list)
    for entry in entries:
        if entry.index in seen:
            raise ValueError(f"Palace WavePort Index {entry.index} is used twice.")
        seen.add(entry.index)
        by_face[entry.portnumber].append(entry)
    for group in by_face.values():
        name = group[0].port
        active = [entry for entry in group if entry.active]
        if len(active) != 1:
            raise ValueError(
                f"Wave port {name!r} has {len(active)} active entries; Palace "
                "accepts exactly one active entry per face."
            )
        if active[0].index != min(entry.index for entry in group):
            raise ValueError(
                f"The active entry of wave port {name!r} (Index "
                f"{active[0].index}) must have the lowest Index of its face; "
                "Palace refuses an active port whose boundary attribute is "
                "already assigned to an earlier entry."
            )
        modes = [entry.mode for entry in group]
        if len(set(modes)) != len(modes):
            raise ValueError(f"Wave port {name!r} lists a mode twice: {sorted(modes)}.")


def faces_by_index(entries: Iterable[PortModeEntry]) -> dict[int, str]:
    """Map each Palace ``Index`` to the name of the face (gsim port) it is on."""
    return {entry.index: entry.port for entry in entries}


def write_port_modes(directory: str | Path, entries: Sequence[PortModeEntry]) -> Path:
    """Write the sidecar ``port_modes.json`` into ``directory``.

    Args:
        directory: Folder holding ``config.json``.
        entries: The entries of every multimode port, in ``Index`` order.

    Returns:
        The path of the sidecar.

    Raises:
        ValueError: If the entries fail :func:`check_port_mode_entries`.
    """
    check_port_mode_entries(entries)
    path = Path(directory) / PORT_MODES_FILENAME
    payload = {
        "entries": [
            {**asdict(entry), "attributes": list(entry.attributes)}
            for entry in sorted(entries, key=lambda entry: entry.index)
        ]
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=4)
    return path


def read_port_modes(source: str | Path) -> tuple[PortModeEntry, ...]:
    """Read a sidecar written by :func:`write_port_modes`.

    Args:
        source: The sidecar file, or the folder that holds it.

    Returns:
        The entries in ``Index`` order.

    Raises:
        FileNotFoundError: If there is no sidecar.
        ValueError: If the file does not hold a list of entries or they fail
            :func:`check_port_mode_entries`.
    """
    path = Path(source)
    if path.is_dir():
        path = path / PORT_MODES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No {PORT_MODES_FILENAME} at {path}")
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if rows is None or not isinstance(rows, list):
        raise ValueError(f"{path} has no 'entries' list.")
    try:
        entries = tuple(
            PortModeEntry(
                index=int(row["index"]),
                port=str(row["port"]),
                portnumber=int(row["portnumber"]),
                mode=int(row["mode"]),
                active=bool(row["active"]),
                excited=bool(row["excited"]),
                attributes=tuple(int(a) for a in row.get("attributes", ())),
            )
            for row in rows
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} has a malformed entry: {error!r}") from error
    check_port_mode_entries(entries)
    return tuple(sorted(entries, key=lambda entry: entry.index))


def same_face_pairs(
    faces: Iterable[tuple[str, tuple[float, float], float]],
    *,
    tol_um: float = FACE_POSITION_TOL_UM,
) -> list[tuple[str, str]]:
    """Return the pairs of ports that sit on one face.

    Args:
        faces: ``(name, (x, y) centre in um, orientation in degrees)`` per
            port.
        tol_um: Distance below which two centres are the same place.

    Returns:
        The ``(name, name)`` pairs whose centres and orientations (modulo 360
        degrees) agree. A multimode port puts all its modes on one face with
        ``modes=``, so two ports there would be two surfaces on one place.
    """
    items = list(faces)
    pairs: list[tuple[str, str]] = []
    for i, (name_a, centre_a, angle_a) in enumerate(items):
        for name_b, centre_b, angle_b in items[i + 1 :]:
            turn = (angle_a - angle_b) % 360.0
            aligned = min(turn, 360.0 - turn) < 1e-6
            apart = math.dist(centre_a, centre_b)
            if aligned and apart <= tol_um:
                pairs.append((name_a, name_b))
    return pairs
