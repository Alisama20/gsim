"""Terminal (multi-pin) ports: which conductors, which paths, which ``T_V`` and ``T_I``.

:mod:`gsim.palace.terminal` turns *modal* S-parameters into *terminal* S-parameters
once it is given, for every physical port, the matrix of terminal voltages
``T_V`` and terminal currents ``T_I`` of the port's modes. It says nothing about
where those matrices come from. This module is the first half of that answer
for a Palace port:

* :class:`TerminalPortSpec` names a port's terminals and its reference conductor
  as rectangles on the port's cross-section (the
  :class:`gsim.common.modes.Conductor` the line-impedance code already uses);
* :func:`terminal_paths` draws one voltage path per terminal, from the terminal
  to the reference, that never crosses a conductor;
* :func:`declare_terminal_paths` registers those paths on a
  :class:`~gsim.palace.boundarymode.BoundaryModeSim`, so that Palace reports the
  voltage of every mode on every path;
* :func:`port_terminal_matrices` reads the saved fields of the solved modes and
  returns ``T_V`` and ``T_I`` in SI units, one column per mode, normalised to
  unit power and with a fixed phase.

Everything is stated in the cross-section's own ``(h, v)`` coordinates (um).

Conventions
-----------

* Time convention ``exp(+j w t)``, as in the rest of gsim.
* Column ``k`` of ``T_V`` holds the voltages of terminal ``i`` against the
  reference conductor for mode ``k``, from the line integral of the transverse
  electric field along the terminal's path (the path is declared from the
  terminal to the reference, so a terminal above the reference has a positive
  voltage when the field points from the terminal to the reference).
* Column ``k`` of ``T_I`` holds the terminal currents *into the network* for a
  forward wave of that mode. The contour integral of ``H`` around a conductor
  gives the current along the propagation axis; whichever way the port face
  points, the column is oriented so that ``Re(T_V[:, k]^H T_I[:, k]) > 0``,
  which is the statement that the incident wave carries power into the network.
* Each column is scaled so that ``|2 P| = 1`` with ``P`` the complex power flux
  of the saved mode, matching Palace's unit-power convention for wave-port
  modes, and rotated so that the voltage on its *pilot* terminal (by default the
  terminal with the largest voltage) is real and positive. Scaling a column of
  both matrices by one complex constant leaves the terminal S unchanged, so
  neither choice matters to :func:`gsim.palace.terminal.modal_to_terminal_s` on
  its own; they exist so that these columns can be compared with the wave-port
  modes of a 3D run, which carry their own scale and phase.

The second half joins these 2D matrices to the modal S of the driven runs:

* :func:`load_modal_run` and :func:`modal_s_from_runs` read ``port-S.csv`` by
  Palace index, decode the extra entries of a multimode port with the sidecar
  ``port_modes.json`` and assemble one modal S from a single run or from the
  active-mode cycling runs;
* :func:`align_gauge` pairs the 2D modes with the 3D port modes by ``n_eff`` and
  puts ``T_V`` and ``T_I`` into the gauge of the 3D modes (the pilot-terminal
  rule, or a fit to the voltages of Palace's own saved port modes);
  :func:`port_mode_reference_voltages` reads those saved port-mode fields off a
  driven run and returns the voltages to fit to;
* :func:`terminal_s` (and :func:`build_terminal_s`, which keeps the pieces) turns
  that into terminal S-parameters, with lumped ports passed through;
* :func:`short_terminals` and :func:`mixed_mode` act on the resulting network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from gsim.common.modes import Conductor, Extent
from gsim.palace._validation_log import port_mode_samples
from gsim.palace._validation_port_modes import (
    FACE_MODE_RTOL,
    PORT_AGREEMENT_RTOL,
    SPEED_OF_LIGHT_M_S,
)
from gsim.palace.line_impedance import PATH_CLEARANCE_FRACTION
from gsim.palace.mode_fields import (
    DEFAULT_PERIMETER_TOL_UM,
    VOLTAGE_NSAMPLES,
    contour_current,
    load_port_mode_fields,
    power_flux,
    reaction_matrix,
    voltage_line_integral,
)
from gsim.palace.ports.modes import (
    PORT_MODES_FILENAME,
    check_port_mode_entries,
    read_port_modes,
)
from gsim.palace.results import (
    SParam,
    _numeric_csv_column,
    _parse_sparam_col,
    _resolve_source,
)
from gsim.palace.terminal import (
    currents_from_reaction,
    degenerate_mode_groups,
    modal_to_terminal_s,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray
    from skrf import Network

    from gsim.palace._validation_models import ModeSample
    from gsim.palace.boundarymode import BoundaryModeSim
    from gsim.palace.mode_fields import BoundaryModeField
    from gsim.palace.ports.modes import PortModeEntry

__all__ = [
    "FREQ_RTOL",
    "INTERPOLATIONS",
    "LOG_FREQ_RTOL",
    "MAX_COND",
    "PAIRING_RTOL",
    "PORT_FIELD_TOL_UM",
    "AlignedPortModes",
    "DegenerateFlag",
    "LumpedTerminal",
    "ModalPortMode",
    "ModalRun",
    "ModalS",
    "PortModes2D",
    "PortTerminalMatrices",
    "TerminalPortSpec",
    "TerminalSResult",
    "align_gauge",
    "build_terminal_s",
    "declare_terminal_paths",
    "load_modal_run",
    "mixed_mode",
    "modal_s_from_runs",
    "port_mode_reference_voltages",
    "port_modes_2d",
    "port_terminal_matrices",
    "short_terminals",
    "terminal_paths",
    "terminal_s",
]

#: ``cond(T_V)`` above which the modes are taken not to span the terminals;
#: the same bound :func:`gsim.palace.terminal.modal_to_terminal_s` applies.
MAX_COND: float = 1e12

#: Palace's mode fields are in um coordinates against SI field values, so a
#: voltage or current integrated over um comes back a factor 1e6 too large and
#: a power integrated over um^2 a factor 1e12 too large.
_UM: float = 1e-6

#: How far a path sample may sit outside the mesh of a saved 3D port face (um)
#: and still be evaluated: Palace writes single-precision coordinates, and a
#: terminal path starts a hair inside a gap.
PORT_FIELD_TOL_UM: float = 1e-3

Point = tuple[float, float]


def _overlap_or_touch(a: Extent, b: Extent) -> bool:
    """Whether two rectangles share an interior point or an edge."""
    (a_h0, a_h1), (a_v0, a_v1) = a
    (b_h0, b_h1), (b_v0, b_v1) = b
    return a_h0 <= b_h1 and b_h0 <= a_h1 and a_v0 <= b_v1 and b_v0 <= a_v1


def _check_extent(name: str, extent: Extent) -> None:
    """Refuse a rectangle that is empty, inverted or not finite."""
    (h0, h1), (v0, v1) = extent
    if not all(np.isfinite([h0, h1, v0, v1])) or not (h0 < h1 and v0 < v1):
        raise ValueError(
            f"Conductor {name!r} has the extent {extent}, which is not a "
            "rectangle ((h_min, h_max), (v_min, v_max)) with h_min < h_max and "
            "v_min < v_max."
        )


@dataclass(frozen=True)
class TerminalPortSpec:
    """The conductors of one multi-pin port, as rectangles on its cross-section.

    A port with ``N`` terminals needs ``N`` modes: the terminals are all the
    conductors of the port except the reference, and a conductor that floats
    is a terminal like any other (see the section on grounds in the terminal
    wave port guide).

    Attributes:
        name: Name of the physical port; it prefixes the names of the voltage
            paths, as ``"{port}:{terminal}"``.
        terminals: The conductors whose voltage and current are wanted, in
            the order of the terminal S-parameters.
        reference: The conductor every voltage is measured against.
    """

    name: str
    terminals: Sequence[Conductor]
    reference: Conductor

    def __post_init__(self) -> None:
        """Validate the layout: unique names, real rectangles, no overlaps.

        Raises:
            ValueError: When the port or a conductor has no name, there is no
                terminal, names repeat, the reference is also a terminal, a
                conductor is not a proper rectangle, or two conductors
                overlap or touch.
        """
        object.__setattr__(self, "terminals", tuple(self.terminals))
        if not self.name:
            raise ValueError("A terminal port needs a name.")
        if not self.terminals:
            raise ValueError(f"Port {self.name!r} has no terminals.")
        everyone = [*self.terminals, self.reference]
        names = [c.name for c in everyone]
        if any(not n for n in names):
            raise ValueError(f"Port {self.name!r} has an unnamed conductor.")
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"Port {self.name!r}: conductor names must be unique, and the "
                f"reference may not also be a terminal; repeated: {repeated}."
            )
        for conductor in everyone:
            _check_extent(conductor.name, conductor.extent)
        for first, second in combinations(everyone, 2):
            if _overlap_or_touch(first.extent, second.extent):
                raise ValueError(
                    f"Port {self.name!r}: conductors {first.name!r} and "
                    f"{second.name!r} overlap or touch, so there is no gap to "
                    "integrate a voltage across."
                )

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """The terminal names, in port order."""
        return tuple(t.name for t in self.terminals)


def _segment_hits_box(p: Point, q: Point, box: Extent) -> bool:
    """Whether the segment ``p``-``q`` meets the closed rectangle ``box``."""
    (h0, h1), (v0, v1) = box
    t_lo, t_hi = 0.0, 1.0
    for start, delta, lo, hi in (
        (p[0], q[0] - p[0], h0, h1),
        (p[1], q[1] - p[1], v0, v1),
    ):
        if delta == 0.0:
            if start < lo or start > hi:
                return False
            continue
        t_a, t_b = (lo - start) / delta, (hi - start) / delta
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        t_lo, t_hi = max(t_lo, t_a), min(t_hi, t_b)
        if t_lo > t_hi:
            return False
    return True


def _mid(lo: float, hi: float) -> float:
    """Midpoint of an interval."""
    return 0.5 * (lo + hi)


def _direct_path(term: Extent, ref: Extent, fraction: float) -> list[Point] | None:
    """Straight path across the gap between two conductors, if they have one.

    Across the horizontal gap from face to face at each conductor's
    mid-height when the two are side by side, else across the vertical gap
    at the middle of their common width. Each end keeps ``fraction`` of the
    smallest of the gap and the two conductors' thicknesses off its face, in
    the gap, as :func:`gsim.palace.line_impedance.line_impedance_paths` does.

    Args:
        term: The terminal's rectangle.
        ref: The reference conductor's rectangle.
        fraction: Inset fraction.

    Returns:
        Two points, or ``None`` when the rectangles are separated along
        neither axis.
    """
    (th0, th1), (tv0, tv1) = term
    (rh0, rh1), (rv0, rv1) = ref
    gap_h = max(rh0 - th1, th0 - rh1)
    gap_v = max(rv0 - tv1, tv0 - rv1)
    if gap_h > 0.0:
        a, b, sign = (th1, rh0, 1.0) if rh0 > th1 else (th0, rh1, -1.0)
        inset = fraction * min(gap_h, tv1 - tv0, rv1 - rv0)
        return [
            (a + sign * inset, _mid(tv0, tv1)),
            (b - sign * inset, _mid(rv0, rv1)),
        ]
    if gap_v > 0.0:
        lo, hi = max(th0, rh0), min(th1, rh1)
        if hi <= lo:
            return None
        a, b, sign = (tv1, rv0, 1.0) if rv0 > tv1 else (tv0, rv1, -1.0)
        inset = fraction * min(gap_v, th1 - th0, rh1 - rh0)
        h_mid = _mid(lo, hi)
        return [(h_mid, a + sign * inset), (h_mid, b - sign * inset)]
    return None


def _detour_path(
    index: int,
    boxes: Sequence[Extent],
    ref: Extent,
    domain: Extent,
    fraction: float,
) -> list[Point] | None:
    """Path out of a terminal's gap, over all the metal, down onto the reference.

    The path leaves the terminal at mid-height, into the gap on the side the
    reference lies on, rises there to a height above every conductor, runs
    across to the middle of the reference, and comes down onto its top face.
    The rise above the metal is the smaller of the gap it rose through and half
    the room the domain leaves above the metal, so the path stays close to the
    conductors rather than wandering out to the domain wall.

    Args:
        index: Position of the terminal in ``boxes``.
        boxes: Every conductor's rectangle (terminals, then the reference).
        ref: The reference conductor's rectangle.
        domain: The meshed domain.
        fraction: Inset fraction.

    Returns:
        The polyline (consecutive points distinct), or ``None`` when the
        domain leaves no room above the metal or the terminal has no gap on
        that side.
    """
    (th0, th1), (tv0, tv1) = boxes[index]
    (rh0, rh1), (rv0, rv1) = ref
    (d_h0, d_h1), (_, d_v1) = domain
    v_mid = _mid(tv0, tv1)
    direction = 1.0 if _mid(rh0, rh1) >= _mid(th0, th1) else -1.0
    face = th1 if direction > 0 else th0
    gaps = [(d_h1 - face) if direction > 0 else (face - d_h0)]
    for other, ((b_h0, b_h1), (b_v0, b_v1)) in enumerate(boxes):
        if other == index or not b_v0 <= v_mid <= b_v1:
            continue
        if direction > 0 and b_h0 >= face:
            gaps.append(b_h0 - face)
        elif direction < 0 and b_h1 <= face:
            gaps.append(face - b_h1)
    gap = min(gaps)
    top = max(v1 for _, (_, v1) in boxes)
    headroom = d_v1 - top
    if gap <= 0.0 or headroom <= 0.0:
        return None
    x_exit = face + direction * fraction * min(gap, tv1 - tv0)
    rise = min(0.5 * headroom, gap)
    y_top = top + rise
    x_down = _mid(rh0, rh1)
    y_end = rv1 + fraction * min(rise, rv1 - rv0)
    path = [(x_exit, v_mid), (x_exit, y_top), (x_down, y_top), (x_down, y_end)]
    return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]


def _inside(point: Point, domain: Extent) -> bool:
    """Whether a point is strictly inside the domain rectangle."""
    (h0, h1), (v0, v1) = domain
    return h0 < point[0] < h1 and v0 < point[1] < v1


def terminal_paths(
    spec: TerminalPortSpec,
    domain: Extent,
    *,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> dict[str, tuple[Point, ...]]:
    """Draw one voltage path per terminal, from the terminal to the reference.

    Where the terminal and the reference face each other across a clear gap the
    path is the straight line across it, the same path
    :func:`gsim.palace.line_impedance.line_impedance_paths` uses. Where another
    conductor stands in the way, the path leaves the terminal at mid-height into
    the gap beside it, rises above all the metal, runs across and comes down
    onto the reference. Every path starts and ends a hair inside the gap
    (``clearance_fraction`` of the gap, as in
    :data:`gsim.palace.line_impedance.PATH_CLEARANCE_FRACTION`), on the meshed
    domain rather than on a conductor's face, and no segment of it meets any
    conductor's rectangle.

    The voltage of a full-wave mode depends a little on the path; for the
    quasi-TEM modes of an on-chip line the dependence is small, and the rule
    above is fixed so that it is the same in every run.

    Args:
        spec: The port's conductors.
        domain: ``((h_min, h_max), (v_min, v_max))`` of the meshed domain
            the paths are sampled on (um).
        clearance_fraction: Inset of the path ends from the conductor faces,
            as a fraction of the gap.

    Returns:
        ``{terminal name: ((h, v), ...)}`` in terminal order.

    Raises:
        ValueError: When the domain is not a rectangle, a conductor is not
            inside it, or no path from a terminal to the reference avoids
            the other conductors and stays in the domain.
    """
    _check_extent("domain", domain)
    conductors = [*spec.terminals, spec.reference]
    (d_h0, d_h1), (d_v0, d_v1) = domain
    for conductor in conductors:
        (h0, h1), (v0, v1) = conductor.extent
        if h0 < d_h0 or h1 > d_h1 or v0 < d_v0 or v1 > d_v1:
            raise ValueError(
                f"Conductor {conductor.name!r} at {conductor.extent} is not inside "
                f"the domain {domain}."
            )
    boxes = [c.extent for c in conductors]
    ref = spec.reference.extent
    paths: dict[str, tuple[Point, ...]] = {}
    for index, terminal in enumerate(spec.terminals):
        candidates = (
            _direct_path(terminal.extent, ref, clearance_fraction),
            _detour_path(index, boxes, ref, domain, clearance_fraction),
        )
        for path in candidates:
            if (
                path is not None
                and len(path) >= 2
                and all(_inside(p, domain) for p in path)
                and not any(
                    _segment_hits_box(p, q, box)
                    for p, q in pairwise(path)
                    for box in boxes
                )
            ):
                paths[terminal.name] = tuple(path)
                break
        else:
            raise ValueError(
                f"No voltage path from terminal {terminal.name!r} to the reference "
                f"{spec.reference.name!r} of port {spec.name!r} avoids the other "
                "conductors inside the domain: a conductor spans the way over the "
                "metal, or the domain leaves no room above it."
            )
    return paths


def declare_terminal_paths(
    sim: BoundaryModeSim,
    spec: TerminalPortSpec,
    domain: Extent,
    *,
    nsamples: int = 100,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> list[int]:
    """Register every terminal's voltage path on a boundary-mode solve.

    The paths are postprocessing only: they load nothing and leave the
    eigenproblem as it was. Palace then reports the complex voltage of each mode
    on each path in ``mode-V.csv``, under the index this returns.

    Args:
        sim: The boundary-mode simulation about to be solved.
        spec: The port's conductors.
        domain: The meshed domain (um), as for :func:`terminal_paths`.
        nsamples: Quadrature order of each line integral in Palace.
        clearance_fraction: Inset of the path ends, as for :func:`terminal_paths`.

    Returns:
        The 1-based Palace index of each terminal's path, in terminal order.
        A path named ``"{port}:{terminal}"`` that was declared before is
        replaced and keeps its index.
    """
    paths = terminal_paths(spec, domain, clearance_fraction=clearance_fraction)
    return [
        sim.add_impedance_path(
            f"{spec.name}:{name}",
            voltage=[[h, v] for h, v in points],
            nsamples=nsamples,
        )
        for name, points in paths.items()
    ]


@dataclass(frozen=True, eq=False)
class PortTerminalMatrices:
    """Terminal voltages and currents of one port's modes.

    Attributes:
        t_v: ``(N, N)`` terminal voltages in volts, column ``k`` for mode ``k``
            (row ``i`` is terminal ``i``), normalised to ``|2 P| = 1`` with the
            pilot voltage real and positive.
        t_i: ``(N, N)`` terminal currents into the network in amperes, same
            layout, same scale and phase as ``t_v``.
        pilot: Terminal row used as the phase reference of each mode.
        power: ``|2 P|`` of each saved mode before normalisation, in watts for
            the amplitude the field was saved with.
        cond: Condition number of ``t_v``.
    """

    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]
    pilot: tuple[int, ...]
    power: NDArray[np.float64]
    cond: float


def _pilot_rows(
    t_v: NDArray[np.complex128],
    names: Sequence[str],
    pilot: Sequence[int | str] | None,
) -> tuple[int, ...]:
    """The terminal each mode's phase is referred to.

    Default: the terminal with the largest voltage magnitude, the lowest index
    among those within 1e-6 of it so that a symmetric mode picks the same
    terminal on every run.
    """
    count = t_v.shape[0]
    if pilot is None:
        rows = []
        for k in range(count):
            magnitude = np.abs(t_v[:, k])
            near_max = np.flatnonzero(magnitude >= (1.0 - 1e-6) * magnitude.max())
            rows.append(int(near_max[0]))
        return tuple(rows)
    if len(pilot) != count:
        raise ValueError(f"pilot names {len(pilot)} terminals for {count} modes.")
    rows = []
    for entry in pilot:
        if isinstance(entry, str):
            if entry not in names:
                known = list(names)
                raise ValueError(f"pilot terminal {entry!r} is not one of {known}.")
            rows.append(names.index(entry))
        elif 0 <= int(entry) < count:
            rows.append(int(entry))
        else:
            raise ValueError(f"pilot terminal index {entry} is outside 0..{count - 1}.")
    return tuple(rows)


def port_terminal_matrices(
    fields: Sequence[BoundaryModeField],
    spec: TerminalPortSpec,
    *,
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    current: Literal["contour", "reaction"] = "contour",
    pilot: Sequence[int | str] | None = None,
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> PortTerminalMatrices:
    """``T_V`` and ``T_I`` of a port's modes from their saved 2D fields.

    ``T_V[i, k]`` is the line integral of ``E_t`` of mode ``k`` along terminal
    ``i``'s path. ``T_I`` comes one of two ways:

    * ``"contour"``: Ampere's contour integral of ``H`` around each terminal's
      rectangle, :func:`gsim.palace.mode_fields.contour_current`;
    * ``"reaction"``: from the unconjugated reaction of the modes,
      ``T_I = T_V^-T R`` with ``R_jk = integral (E_j x H_k) . z dA``
      (:func:`gsim.palace.terminal.currents_from_reaction`), which needs no
      conductor outline on the mesh and covers a degenerate group in any basis.

    Each column is then scaled to ``|2 P| = 1`` (``P`` the complex power flux of
    the mode, in SI), oriented so that the incident wave carries power into the
    network (``Re(T_V^H T_I) > 0`` per column, by flipping the sign of ``T_I``),
    and rotated so that its pilot terminal's voltage is real and positive.

    Args:
        fields: The saved fields of the modes, one per terminal, all on the same
            mesh. Mode ``k`` becomes column ``k``.
        spec: The port's conductors.
        domain: The meshed domain (um), used to draw the paths. Not needed when
            ``paths`` is given.
        paths: Voltage paths per terminal name in the cross-section's
            coordinates (um), for example the ones declared on the boundary-mode
            solve. Overrides ``domain``.
        current: How to obtain ``T_I``.
        pilot: Per mode, the terminal (index or name) whose voltage sets the
            phase. ``None`` takes the terminal with the largest voltage.
        nsamples: Sample points per path segment for the voltage integrals
            (default 800: at 200 the integral along a path that passes close to
            a metal corner is off by about 1 %).
        tol_um: How far outside the mesh a path sample may sit (um).

    Returns:
        The matrices and the numbers needed to audit them.

    Raises:
        ValueError: When the number of modes differs from the number of
            terminals, ``current`` is not one of the two routes, neither
            ``domain`` nor ``paths`` is given or a terminal has no path, a mode
            carries no power or no real power into the port (so it cannot be
            oriented), the pilot voltage is zero, or the modes do not span the
            terminals (``cond(T_V) > MAX_COND``).
    """
    modes = list(fields)
    count = len(spec.terminals)
    if len(modes) != count:
        raise ValueError(
            f"Port {spec.name!r} has {count} terminals and needs one saved mode "
            f"per terminal, but {len(modes)} were given."
        )
    if current not in ("contour", "reaction"):
        raise ValueError(f"current must be 'contour' or 'reaction', got {current!r}.")
    if paths is None:
        if domain is None:
            raise ValueError("Pass domain= (to draw the paths) or paths=.")
        paths = dict(terminal_paths(spec, domain))
    absent = [n for n in spec.terminal_names if n not in paths]
    if absent:
        raise ValueError(f"No voltage path was given for the terminals {absent}.")

    t_v = np.empty((count, count), dtype=np.complex128)
    for k, mode in enumerate(modes):
        for i, name in enumerate(spec.terminal_names):
            t_v[i, k] = voltage_line_integral(
                mode, paths[name], nsamples=nsamples, tol_um=tol_um
            )
    t_v *= _UM

    power = np.array([abs(2.0 * power_flux(mode)) for mode in modes]) * _UM**2
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        bad = [k + 1 for k in range(count) if not power[k] > 0.0]
        raise ValueError(f"Mode(s) {bad} of port {spec.name!r} carry no power.")
    scale = 1.0 / np.sqrt(power)
    t_v = t_v * scale[None, :]
    cond = float(np.linalg.cond(t_v))
    if not np.isfinite(cond) or cond > MAX_COND:
        raise ValueError(
            f"T_V of port {spec.name!r} is ill-conditioned (cond = {cond:.1e}): the "
            "modes do not span the terminals."
        )

    if current == "contour":
        t_i = np.empty((count, count), dtype=np.complex128)
        for k, mode in enumerate(modes):
            for i, terminal in enumerate(spec.terminals):
                t_i[i, k] = contour_current(
                    mode,
                    h_span=terminal.extent[0],
                    v_span=terminal.extent[1],
                    tol_um=tol_um,
                )
        t_i = t_i * _UM * scale[None, :]
    else:
        reaction = reaction_matrix(modes) * _UM**2 * np.outer(scale, scale)
        t_i = np.asarray(currents_from_reaction(t_v, reaction), dtype=np.complex128)

    rows = _pilot_rows(t_v, spec.terminal_names, pilot)
    for k in range(count):
        real_power = float(np.real(np.vdot(t_v[:, k], t_i[:, k])))
        apparent = float(np.linalg.norm(t_v[:, k]) * np.linalg.norm(t_i[:, k]))
        if abs(real_power) <= 1e-9 * apparent:
            raise ValueError(
                f"Mode {k + 1} of port {spec.name!r} carries no real power into the "
                "port, so the direction of its current cannot be fixed."
            )
        if real_power < 0.0:
            t_i[:, k] = -t_i[:, k]
        pilot_voltage = t_v[rows[k], k]
        if abs(pilot_voltage) == 0.0:
            raise ValueError(
                f"Mode {k + 1} of port {spec.name!r} has zero voltage on its pilot "
                f"terminal {spec.terminal_names[rows[k]]!r}."
            )
        phase = np.conj(pilot_voltage) / abs(pilot_voltage)
        t_v[:, k] = t_v[:, k] * phase
        t_i[:, k] = t_i[:, k] * phase
    return PortTerminalMatrices(
        t_v=t_v, t_i=t_i, pilot=rows, power=np.asarray(power), cond=cond
    )


# ---------------------------------------------------------------------------
# Modal S-parameters from the driven runs
# ---------------------------------------------------------------------------

#: Largest relative difference between the frequency axes of runs that are
#: combined into one modal S.
FREQ_RTOL: float = 1e-6

#: Largest relative distance between the frequency of a logged mode solve and
#: a frequency of ``port-S.csv`` for the two to be the same point (the log
#: rounds the frequency, as ``ModeExpectation.frequency_rtol`` assumes).
LOG_FREQ_RTOL: float = 5e-4

#: Largest relative difference of ``n_eff`` between a 2D mode and a 3D port mode
#: for the two to be paired. Modes of one face closer than this are reported
#: as degenerate, because the pairing cannot tell them apart.
PAIRING_RTOL: float = PORT_AGREEMENT_RTOL

_PORT_S = "port-S.csv"


class ModalPortMode(NamedTuple):
    """One row (and column) of a modal S-parameter matrix.

    Attributes:
        port: Name of the gsim port, i.e. of the face (the name of a lumped
            terminal for a lumped port).
        portnumber: Palace number of the port: the ``Index`` of its active
            entry, which orders the physical ports.
        mode: The Palace ``Mode`` of the entry; ``0`` for a lumped port, which
            has no modes.
    """

    port: str
    portnumber: int
    mode: int


@dataclass(frozen=True)
class LumpedTerminal:
    """A lumped port that is passed through to the terminal S unchanged.

    A lumped port is one terminal with one mode whose voltage and current are
    defined by its own reference resistance, so its block of ``T_V`` and
    ``T_I`` is ``sqrt(R)`` and ``1 / sqrt(R)`` and the terminal reference
    impedance is ``R``. That makes ``A = 1`` and ``B = 0`` for the port: it
    keeps the S-parameters Palace reported, referred to ``R``.

    Attributes:
        name: Name of the terminal in the output network.
        index: The Palace port ``Index`` (the number in ``S[i][j]``).
        resistance: Reference resistance of the port in ohm, the value Palace
            used for its S-parameters.
    """

    name: str
    index: int
    resistance: float = 50.0

    def __post_init__(self) -> None:
        """Validate the name, index and resistance.

        Raises:
            ValueError: For an empty name, an index below 1 or a resistance
                that is not finite and positive.
        """
        if not self.name:
            raise ValueError("A lumped terminal needs a name.")
        if int(self.index) != self.index or self.index < 1:
            raise ValueError(
                f"Lumped terminal {self.name!r}: the Palace index must be an "
                f"integer >= 1, got {self.index!r}."
            )
        if not (np.isfinite(self.resistance) and self.resistance > 0.0):
            raise ValueError(
                f"Lumped terminal {self.name!r}: the resistance must be finite "
                f"and positive, got {self.resistance!r}."
            )


@dataclass(frozen=True, eq=False)
class ModalRun:
    """The S-parameters of one driven run, keyed by Palace index.

    Attributes:
        freq_ghz: Frequencies of ``port-S.csv`` in GHz.
        s: ``{(i, j): S[i][j] vs frequency}``: the response at Palace port
            index ``i`` to the excitation of index ``j``. Only the excited
            columns are present.
        entries: The ``port_modes.json`` entries of the run (which port, mode
            and active flag each Palace index stands for).
        samples: The port mode solves printed in the run's log, if it was
            read; they carry the wave number of every entry.
        label: A name for messages.
    """

    freq_ghz: NDArray[np.float64]
    s: Mapping[tuple[int, int], NDArray[np.complex128]]
    entries: tuple[PortModeEntry, ...]
    samples: tuple[ModeSample, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        """Coerce the arrays and check the entries.

        Raises:
            ValueError: When the frequencies are not a non-empty finite 1-D
                array, a column does not have one value per frequency, or the
                entries are not a valid set (see ``check_port_mode_entries``).
        """
        freq = np.asarray(self.freq_ghz, dtype=float)
        if freq.ndim != 1 or freq.size == 0 or not np.all(np.isfinite(freq)):
            raise ValueError(
                f"Run {self.label!r}: the frequencies must be a non-empty finite "
                "1-D array."
            )
        columns: dict[tuple[int, int], NDArray[np.complex128]] = {}
        for key, values in self.s.items():
            array = np.asarray(values, dtype=np.complex128)
            if array.shape != freq.shape:
                raise ValueError(
                    f"Run {self.label!r}: S{list(key)} has shape {array.shape}, "
                    f"expected {freq.shape} (one value per frequency)."
                )
            columns[(int(key[0]), int(key[1]))] = array
        entries = tuple(self.entries)
        check_port_mode_entries(entries)
        object.__setattr__(self, "freq_ghz", freq)
        object.__setattr__(self, "s", columns)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "samples", tuple(self.samples))


def _find_sidecar(base: Path, csv_path: Path) -> Path | None:
    """Search the places a run folder may keep ``port_modes.json``."""
    candidates = [
        csv_path.parent / PORT_MODES_FILENAME,
        base / PORT_MODES_FILENAME,
        base / "output" / PORT_MODES_FILENAME,
        csv_path.parent.parent / "input" / PORT_MODES_FILENAME,
        csv_path.parent.parent / PORT_MODES_FILENAME,
        csv_path.parent.parent.parent / PORT_MODES_FILENAME,
    ]
    return next((path for path in candidates if path.exists()), None)


def _read_port_s(
    csv_path: Path,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], NDArray[np.complex128]]]:
    """Read ``port-S.csv`` by Palace index, with no port names in between.

    ``load_sparams`` names the columns from ``port_information.json``, which
    knows the ports but not the extra entries of a multimode port. The
    headers carry the Palace index, so they are read directly.

    Raises:
        ValueError: If there is no frequency column or a magnitude column has
            no phase column (or the other way round).
    """
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    freq_col = next((c for c in frame.columns if c.startswith("f")), None)
    if freq_col is None:
        raise ValueError(f"{csv_path} has no frequency column.")
    freq = _numeric_csv_column(frame[freq_col], csv_path, freq_col)
    raw: dict[tuple[int, int], dict[str, NDArray[np.float64]]] = {}
    for col in frame.columns:
        parsed = _parse_sparam_col(col)
        if parsed is None:
            continue
        i, j, kind = parsed
        raw.setdefault((i, j), {})[kind] = _numeric_csv_column(
            frame[col], csv_path, col, allow_negative_infinity=kind == "db"
        )
    s: dict[tuple[int, int], NDArray[np.complex128]] = {}
    for (i, j), parts in sorted(raw.items()):
        if "db" not in parts or "deg" not in parts:
            raise ValueError(
                f"{csv_path}: S[{i}][{j}] needs both a magnitude and a phase column."
            )
        s[(i, j)] = np.asarray(
            SParam(db=parts["db"], deg=parts["deg"]).complex, dtype=np.complex128
        )
    return freq, s


def load_modal_run(
    source: str | Path | dict,
    *,
    sidecar: str | Path | None = None,
    log: str | Path | Sequence[str] | None = None,
) -> ModalRun:
    """Read one driven run: ``port-S.csv``, the sidecar and, optionally, the log.

    Args:
        source: The run folder (the simulation folder, ``output/palace`` or a
            downloaded result folder), a ``port-S.csv`` file, or the results
            dict of ``sim.run()``.
        sidecar: The ``port_modes.json`` of the run. By default it is looked
            for next to ``port-S.csv`` and in the usual input folders.
        log: The Palace log of the run (a file, or its lines). It carries the
            wave number of every port mode, which the gauge alignment pairs the
            2D modes with; without it :attr:`ModalS.wave_numbers` is ``None``.

    Returns:
        The run.

    Raises:
        FileNotFoundError: When there is no ``port-S.csv`` or no sidecar.
        ValueError: When a file is malformed.
    """
    if isinstance(source, Mapping):
        csv_path, base = _resolve_source(dict(source))
    else:
        path = Path(source)
        if path.is_file():
            csv_path, base = path, path.parent
        else:
            csv_path, base = _resolve_source(path)
    if csv_path is None:
        raise FileNotFoundError(f"{_PORT_S} not found in {source!r}")
    if sidecar is not None:
        sidecar_path: Path | None = Path(sidecar)
    else:
        sidecar_path = _find_sidecar(base, csv_path)
    if sidecar_path is None or not sidecar_path.exists():
        raise FileNotFoundError(
            f"No {PORT_MODES_FILENAME} found for {csv_path}. It is written next "
            "to config.json by a simulation with a multimode wave port; pass "
            "sidecar= if it was moved."
        )
    entries = read_port_modes(sidecar_path)
    freq, s = _read_port_s(csv_path)
    samples: tuple[ModeSample, ...] = ()
    if log is not None:
        if isinstance(log, str | Path):
            lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            lines = list(log)
        samples = port_mode_samples(lines)
    return ModalRun(
        freq_ghz=freq,
        s=s,
        entries=entries,
        samples=samples,
        label=str(csv_path.parent),
    )


@dataclass(frozen=True)
class DegenerateFlag:
    """Modes of one face whose wave numbers coincide at some frequencies.

    Attributes:
        port: The face.
        modes: The Palace mode numbers that are within the pairing tolerance of
            each other.
        freq_ghz: The frequencies at which they are.
    """

    port: str
    modes: tuple[int, ...]
    freq_ghz: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class ModalS:
    """The modal S-parameters assembled from one or several runs.

    Attributes:
        freq_ghz: Frequencies in GHz.
        s_modal: ``(nf, M, M)`` modal S, ordered as ``order`` (port by port in
            Palace port order, modes in ascending order inside a port), in
            the gauge Palace used for each (face, mode).
        order: The row and column labels.
        source_run: Index of the run each column was taken from.
        exact: Whether each column comes from a run in which its entry was the
            active one. The other columns were measured with the face
            terminated for a different mode: the entry is excited and measured
            as in the active case, but the face reflects part of it, and so
            do the other modes arriving there.
        wave_numbers: ``(nf, M)`` complex wave numbers (1/m) of the port modes
            from the run logs, NaN where no solve was logged and for lumped
            ports; ``None`` if no log was read.
        degenerate: Modes of a face whose wave numbers coincide.
    """

    freq_ghz: NDArray[np.float64]
    s_modal: NDArray[np.complex128]
    order: tuple[ModalPortMode, ...]
    source_run: tuple[int, ...]
    exact: tuple[bool, ...]
    wave_numbers: NDArray[np.complex128] | None
    degenerate: tuple[DegenerateFlag, ...]

    def port_columns(self, port: str) -> list[int]:
        """The rows (and columns) of ``s_modal`` that belong to a port, in order."""
        return [i for i, item in enumerate(self.order) if item.port == port]


def _as_run(run: ModalRun | str | Path | dict) -> ModalRun:
    """Accept a loaded run or anything :func:`load_modal_run` takes."""
    return run if isinstance(run, ModalRun) else load_modal_run(run)


def _layout(runs: Sequence[ModalRun]) -> dict[int, tuple[str, tuple[int, ...]]]:
    """``{port number: (name, modes)}``, the same in every run.

    Raises:
        ValueError: When two runs do not have the same ports and modes.
    """
    layout: dict[int, tuple[str, tuple[int, ...]]] | None = None
    for run in runs:
        this: dict[int, tuple[str, list[int]]] = {}
        for entry in run.entries:
            name, modes = this.setdefault(entry.portnumber, (entry.port, []))
            if name != entry.port:
                raise ValueError(
                    f"Run {run.label!r}: port number {entry.portnumber} is named "
                    f"{name!r} and {entry.port!r}."
                )
            modes.append(entry.mode)
        found = {
            number: (name, tuple(sorted(modes)))
            for number, (name, modes) in this.items()
        }
        if layout is None:
            layout = found
        elif found != layout:
            raise ValueError(
                f"Run {run.label!r} has the ports and modes {found}, but the first "
                f"run has {layout}: the runs must be the same simulation."
            )
    return layout or {}


def _choose_source(
    runs: Sequence[ModalRun],
    column: ModalPortMode,
    lumped_index: int | None,
) -> tuple[int, int, bool]:
    """Pick the run and Palace excitation index that supply a modal column.

    A multimode entry is taken from the run in which it is the active entry
    (the face is terminated for exactly that mode); if there is none, from the
    first run in which it is excited. A lumped port is taken from the first run
    that has a column for it.

    Returns:
        ``(run position, Palace index of the excitation, exact)``.

    Raises:
        ValueError: When no run excites the column, or two runs both have the
            entry as the active one.
    """
    if lumped_index is not None:
        for k, run in enumerate(runs):
            if any(j == lumped_index for _, j in run.s):
                terminated = all(entry.active for entry in run.entries)
                return k, lumped_index, terminated
        raise ValueError(
            f"No run has an S column for the lumped port {column.port!r} "
            f"(Palace index {lumped_index})."
        )
    active: list[tuple[int, int]] = []
    excited: list[tuple[int, int]] = []
    for k, run in enumerate(runs):
        for entry in run.entries:
            if (
                entry.port == column.port
                and entry.mode == column.mode
                and entry.excited
            ):
                (active if entry.active else excited).append((k, entry.index))
    if len(active) > 1:
        labels = [runs[k].label for k, _ in active]
        raise ValueError(
            f"Mode {column.mode} of port {column.port!r} is the active mode in "
            f"{len(active)} runs ({labels}); give each mode one run."
        )
    if active:
        return active[0][0], active[0][1], True
    if excited:
        return excited[0][0], excited[0][1], False
    raise ValueError(
        f"No run excites mode {column.mode} of port {column.port!r}: its modal "
        "column is missing."
    )


def _k0(freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Free-space wave number in 1/m at frequencies in GHz."""
    return np.asarray(2.0 * np.pi * freq_ghz * 1e9 / SPEED_OF_LIGHT_M_S, dtype=float)


def _wave_numbers(
    runs: Sequence[ModalRun],
    order: Sequence[ModalPortMode],
    freq_ghz: NDArray[np.float64],
) -> NDArray[np.complex128] | None:
    """The logged wave number of every (port, mode) at every frequency.

    The log's ``Port <index>`` is the Palace index of an entry, which differs
    from run to run for the observation entries, so each run decodes its own
    samples with its own sidecar. Repeated solves (one per excitation) are
    reduced to their median.
    """
    series: dict[tuple[str, int], list[tuple[float, complex]]] = {}
    for run in runs:
        keys = {entry.index: (entry.port, entry.mode) for entry in run.entries}
        for sample in run.samples:
            key = keys.get(sample.port)
            frequency = sample.frequency_ghz
            if key is None or frequency is None or not np.isfinite(frequency):
                continue
            series.setdefault(key, []).append((float(frequency), sample.wave_number))
    if not series:
        return None
    out = np.full((freq_ghz.size, len(order)), np.nan + 0j, dtype=np.complex128)
    for c, column in enumerate(order):
        points = series.get((column.port, column.mode))
        if not points:
            continue
        logged = np.array([f for f, _ in points])
        values = np.array([k for _, k in points])
        for f_index, frequency in enumerate(freq_ghz):
            near = np.abs(logged - frequency) <= LOG_FREQ_RTOL * abs(frequency)
            if near.any():
                out[f_index, c] = complex(
                    float(np.median(values[near].real)),
                    float(np.median(values[near].imag)),
                )
    return out


def _degenerate_flags(
    order: Sequence[ModalPortMode],
    wave_numbers: NDArray[np.complex128] | None,
    freq_ghz: NDArray[np.float64],
) -> tuple[DegenerateFlag, ...]:
    """Groups of modes of one face whose ``n_eff`` coincide, per frequency."""
    if wave_numbers is None:
        return ()
    found: dict[tuple[str, tuple[int, ...]], list[float]] = {}
    ports = dict.fromkeys(item.port for item in order if item.mode > 0)
    k0 = _k0(freq_ghz)
    for port in ports:
        cols = [c for c, item in enumerate(order) if item.port == port]
        modes = [order[c].mode for c in cols]
        for f_index, frequency in enumerate(freq_ghz):
            n_eff = wave_numbers[f_index, cols].real / k0[f_index]
            if len(cols) < 2 or not np.all(np.isfinite(n_eff)):
                continue
            for group in degenerate_mode_groups(n_eff, rtol=FACE_MODE_RTOL):
                if len(group) > 1:
                    key = (port, tuple(modes[g] for g in group))
                    found.setdefault(key, []).append(float(frequency))
    return tuple(
        DegenerateFlag(port=port, modes=modes, freq_ghz=tuple(freqs))
        for (port, modes), freqs in found.items()
    )


def modal_s_from_runs(
    runs: Sequence[ModalRun | str | Path | dict],
    *,
    lumped: Sequence[LumpedTerminal] = (),
) -> ModalS:
    """Assemble the modal S-parameters of all faces from one run or several.

    Palace measures every wave-port entry, but only the active entry of a face
    terminates it. The recommended way (active-mode cycling) therefore runs once
    per mode ``k``, with mode ``k`` active on every face and only the mode-``k``
    entries excited. Column ``(face p, mode k)`` of the modal S then comes from
    run ``k`` and has every entry of every face as its rows. A single run with
    mode 1 active and every entry excited gives the same matrix with the
    columns of modes 2 and up approximate.

    The rule per column: take it from the run in which its entry is active, else
    from the first run that excites it. :attr:`ModalS.exact` says which.

    Args:
        runs: Loaded runs (see :func:`load_modal_run`), or what it accepts.
        lumped: Lumped ports of the simulation that are part of the terminal
            network (their Palace index is the port number).

    Returns:
        The assembled matrix, ordered port by port in Palace port number,
        modes ascending inside a port (the order of
        :func:`gsim.palace.terminal.modal_to_terminal_s`).

    Raises:
        ValueError: When no run is given, the runs disagree on their ports,
            modes or frequencies, a lumped port clashes with a multimode port,
            a column is supplied by no run or by two active runs, or a needed
            S entry is absent from the file.
    """
    loaded = [_as_run(run) for run in runs]
    if not loaded:
        raise ValueError("modal_s_from_runs needs at least one run.")
    freq = loaded[0].freq_ghz
    for run in loaded[1:]:
        if run.freq_ghz.shape != freq.shape or not np.allclose(
            run.freq_ghz, freq, rtol=FREQ_RTOL, atol=0.0
        ):
            raise ValueError(
                f"Run {run.label!r} has a different frequency axis from run "
                f"{loaded[0].label!r}; the runs must share one sweep."
            )
    layout = _layout(loaded)
    taken_names = {name for name, _ in layout.values()}
    if len({item.name for item in lumped}) != len(lumped) or any(
        item.name in taken_names for item in lumped
    ):
        raise ValueError(
            "Lumped terminal names must be unique and different from the names "
            "of the multimode ports."
        )
    taken_indices = {entry.index for run in loaded for entry in run.entries}
    if len({item.index for item in lumped}) != len(lumped) or any(
        item.index in taken_indices for item in lumped
    ):
        raise ValueError(
            "Lumped terminal Palace indices must be unique and different from "
            "the indices of the multimode entries."
        )

    columns: list[ModalPortMode] = [
        ModalPortMode(name, number, mode)
        for number, (name, modes) in layout.items()
        for mode in modes
    ]
    lumped_index: dict[str, int] = {}
    for item in lumped:
        columns.append(ModalPortMode(item.name, item.index, 0))
        lumped_index[item.name] = item.index
    order = tuple(sorted(columns, key=lambda c: (c.portnumber, c.mode)))

    row_index: list[dict[tuple[str, int], int]] = []
    for run in loaded:
        rows = {(entry.port, entry.mode): entry.index for entry in run.entries}
        rows.update({(name, 0): index for name, index in lumped_index.items()})
        row_index.append(rows)

    size = len(order)
    s_modal = np.empty((freq.size, size, size), dtype=np.complex128)
    chosen = [
        _choose_source(
            loaded, column, lumped_index.get(column.port) if column.mode == 0 else None
        )
        for column in order
    ]
    source = [k for k, _, _ in chosen]
    exact = [is_exact for _, _, is_exact in chosen]
    for c, (k, j, _) in enumerate(chosen):
        column = order[c]
        run = loaded[k]
        for r, row in enumerate(order):
            i = row_index[k][(row.port, row.mode)]
            values = run.s.get((i, j))
            if values is None:
                raise ValueError(
                    f"Run {run.label!r} has no S[{i}][{j}] (response of "
                    f"{row.port!r} mode {row.mode} to the excitation of "
                    f"{column.port!r} mode {column.mode})."
                )
            s_modal[:, r, c] = values
    wave_numbers = _wave_numbers(loaded, order, freq)
    return ModalS(
        freq_ghz=freq,
        s_modal=s_modal,
        order=order,
        source_run=tuple(source),
        exact=tuple(exact),
        wave_numbers=wave_numbers,
        degenerate=_degenerate_flags(order, wave_numbers, freq),
    )


# ---------------------------------------------------------------------------
# Gauge alignment of the 2D modes with the 3D port modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PortModes2D:
    """``T_V`` and ``T_I`` of one port from one 2D boundary-mode solve.

    Attributes:
        freq_ghz: Frequency of the 2D solve in GHz.
        n_eff: ``(N,)`` effective indices of the 2D modes, in the 2D solve's
            mode order (the real part is used).
        t_v: ``(N, N)`` terminal voltages, column ``l`` for 2D mode ``l``
            (see :func:`port_terminal_matrices`).
        t_i: ``(N, N)`` terminal currents, same layout.
    """

    freq_ghz: float
    n_eff: NDArray[np.float64]
    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]

    def __post_init__(self) -> None:
        """Coerce and validate the arrays.

        Raises:
            ValueError: For a frequency that is not positive and finite, or
                arrays that are not finite, ``(N,)`` and ``(N, N)``.
        """
        if not (np.isfinite(self.freq_ghz) and self.freq_ghz > 0.0):
            raise ValueError(
                f"freq_ghz must be positive and finite, got {self.freq_ghz}."
            )
        n_eff = np.real(np.asarray(self.n_eff, dtype=np.complex128))
        t_v = np.asarray(self.t_v, dtype=np.complex128)
        t_i = np.asarray(self.t_i, dtype=np.complex128)
        count = n_eff.size
        if (
            n_eff.ndim != 1
            or t_v.shape != (count, count)
            or t_i.shape != (count, count)
        ):
            raise ValueError(
                f"Expected n_eff (N,), t_v and t_i (N, N); got {n_eff.shape}, "
                f"{t_v.shape} and {t_i.shape}."
            )
        if not (
            np.all(np.isfinite(n_eff))
            and np.all(np.isfinite(t_v))
            and np.all(np.isfinite(t_i))
        ):
            raise ValueError("n_eff, t_v and t_i must be finite.")
        object.__setattr__(self, "n_eff", n_eff)
        object.__setattr__(self, "t_v", t_v)
        object.__setattr__(self, "t_i", t_i)


def port_modes_2d(
    fields: Sequence[BoundaryModeField],
    spec: TerminalPortSpec,
    *,
    freq_ghz: float,
    n_eff: ArrayLike,
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    current: Literal["contour", "reaction"] = "contour",
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> PortModes2D:
    """``T_V`` and ``T_I`` of one 2D solve, ready for :func:`align_gauge`.

    A thin wrapper of :func:`port_terminal_matrices` that keeps the frequency
    and the effective indices with the matrices; the phase reference is chosen
    later, per 3D mode, by :func:`align_gauge`.

    Args:
        fields: The saved fields of the modes of the solve, one per terminal.
        spec: The port's conductors.
        freq_ghz: Frequency of the solve.
        n_eff: ``(N,)`` effective indices of the modes, in the order of
            ``fields`` (from ``mode-kn.csv``).
        domain: The meshed domain, to draw the paths (see
            :func:`port_terminal_matrices`).
        paths: The paths declared on the solve instead of ``domain``.
        current: How ``T_I`` is obtained.
        nsamples: Sample points per path segment (default 800, see
            :func:`port_terminal_matrices`).
        tol_um: Distance a path sample may sit outside the mesh (um).

    Returns:
        The matrices with their frequency and effective indices.
    """
    matrices = port_terminal_matrices(
        fields,
        spec,
        domain=domain,
        paths=paths,
        current=current,
        nsamples=nsamples,
        tol_um=tol_um,
    )
    return PortModes2D(
        freq_ghz=float(freq_ghz),
        n_eff=np.asarray(n_eff),
        t_v=matrices.t_v,
        t_i=matrices.t_i,
    )


def port_mode_reference_voltages(
    source: str | Path | dict,
    spec: TerminalPortSpec,
    entries: Sequence[PortModeEntry] | str | Path,
    *,
    cycles: Sequence[int],
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    excitation: int = 1,
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = PORT_FIELD_TOL_UM,
) -> list[NDArray[np.complex128]]:
    """Terminal voltages of Palace's own port modes, as ``reference_voltages``.

    A driven run that saves fields (``set_driven(save_fields_at=...)``) writes,
    for every ``WavePort`` entry, the unit-power field of its port mode
    (``E0_<index>``, see :func:`gsim.palace.mode_fields.load_port_mode_fields`).
    This integrates those fields along the terminal paths, one ``(N, N)``
    matrix per saved cycle, column ``k`` for mode ``k`` of the port in
    ascending mode number: what :func:`align_gauge` (and
    :func:`terminal_s`) take as ``reference_voltages`` to fit the 2D modes to the
    3D ones instead of assuming the pilot-terminal gauge.

    Palace writes one cycle per saved frequency, in ascending order, so
    ``cycles`` is the position (1-based) of each 2D solve's frequency among the
    saved ones, in the order of the 2D solves handed to :func:`align_gauge`.
    Use the same paths as for the 2D matrices.

    Args:
        source: The run that saved the fields (results dict or directory); any
            run with the port's entries will do, the port modes do not depend on
            the excitation.
        spec: The port's conductors; its name is the name of the port in the
            sidecar.
        entries: The sidecar ``port_modes.json`` (file or folder), or its
            entries as read by
            :func:`gsim.palace.ports.modes.read_port_modes`.
        cycles: ParaView cycle of each reference, 1-based.
        domain: The meshed domain (um), to draw the paths. Not needed when
            ``paths`` is given.
        paths: Voltage paths per terminal name (um), overriding ``domain``.
        excitation: Excitation whose output is read.
        nsamples: Sample points per path segment.
        tol_um: How far outside the face mesh a path sample may sit (um).

    Returns:
        One ``(N, N)`` complex matrix of volts per cycle.

    Raises:
        ValueError: When the sidecar has not one entry per terminal for the
            port, the face has no attribute, no cycle is given, neither
            ``domain`` nor ``paths`` is given or a terminal has no path, or the
            saved fields cannot be read (see
            :func:`gsim.palace.mode_fields.port_mode_fields_from_grid`).
    """
    rows = (
        read_port_modes(entries) if isinstance(entries, str | Path) else tuple(entries)
    )
    face = sorted(
        (row for row in rows if row.port == spec.name), key=lambda row: row.mode
    )
    count = len(spec.terminals)
    if len(face) != count:
        raise ValueError(
            f"Port {spec.name!r} has {len(face)} entries in the sidecar but "
            f"{count} terminals; the reference needs one saved mode per terminal."
        )
    attributes = sorted({a for row in face for a in row.attributes})
    if not attributes:
        raise ValueError(
            f"The entries of port {spec.name!r} list no boundary attribute."
        )
    if not cycles:
        raise ValueError("cycles is empty: give the ParaView cycle of each 2D solve.")
    if paths is None:
        if domain is None:
            raise ValueError("Pass domain= (to draw the paths) or paths=.")
        paths = dict(terminal_paths(spec, domain))
    absent = [n for n in spec.terminal_names if n not in paths]
    if absent:
        raise ValueError(f"No voltage path was given for the terminals {absent}.")

    references: list[NDArray[np.complex128]] = []
    for cycle in cycles:
        fields = load_port_mode_fields(
            source,
            indices=[row.index for row in face],
            attributes=attributes,
            cycle=int(cycle),
            excitation=excitation,
        )
        volts = np.empty((count, count), dtype=np.complex128)
        for k, field in enumerate(fields):
            for i, name in enumerate(spec.terminal_names):
                volts[i, k] = voltage_line_integral(
                    field, paths[name], nsamples=nsamples, tol_um=tol_um
                )
        references.append(volts * _UM)
    return references


@dataclass(frozen=True, eq=False)
class AlignedPortModes:
    """``T_V`` and ``T_I`` of one port in the gauge of its 3D port modes.

    Attributes:
        freq_ghz: Frequencies of the target grid in GHz.
        t_v: ``(nf, N, N)`` terminal voltages, column ``k`` for the 3D mode
            ``k`` of the port (ascending Palace mode number).
        t_i: ``(nf, N, N)`` terminal currents, same layout.
        pilot: The terminal row that fixes the phase of each mode (empty when
            the gauge came from ``reference_voltages``).
        pairing: For each 2D solve (in ascending frequency), the 2D mode
            index paired with each 3D mode.
        mismatch: Largest relative difference of ``n_eff`` over all pairs.
        degenerate: Groups of 3D modes (0-based positions) that the pairing
            could not tell apart at some 2D frequency.
        gauge_residual: ``(n_2d, N)`` relative residual of the fit to
            ``reference_voltages`` per mode, ``None`` without them.
        gauge_scale: ``(n_2d, N)`` magnitude of the fitted gauge factor per mode
            (1 when the 2D and 3D modes are normalised alike), ``None`` without
            ``reference_voltages``.
    """

    freq_ghz: NDArray[np.float64]
    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]
    pilot: tuple[int, ...]
    pairing: tuple[tuple[int, ...], ...]
    mismatch: float
    degenerate: tuple[tuple[int, ...], ...]
    gauge_residual: NDArray[np.float64] | None = None
    gauge_scale: NDArray[np.float64] | None = None


def _check_grid(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """A strictly increasing, positive, finite 1-D frequency grid."""
    freq = np.asarray(freq_ghz, dtype=float)
    if (
        freq.ndim != 1
        or freq.size == 0
        or not np.all(np.isfinite(freq))
        or np.any(freq <= 0.0)
        or np.any(np.diff(freq) <= 0.0)
    ):
        raise ValueError(
            "freq_ghz must be a strictly increasing 1-D array of positive "
            "finite frequencies."
        )
    return freq


def _fit_gauge(
    t_v: NDArray[np.complex128],
    t_i: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    groups: Sequence[tuple[int, ...]],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit ``T_V`` to the voltages of the 3D port modes, mode by mode.

    For every mode (or group of indistinguishable modes) the least-squares
    factor (matrix) ``M`` with ``reference ~ T_V M``. ``T_I`` gets the same
    ``M``, so voltage and current keep their shape and take the 3D gauge.

    Returns:
        ``T_V M``, ``T_I M``, the relative residual per mode and the column
        norm of ``M`` per mode.

    Raises:
        ValueError: When a reference mode has no voltage at all.
    """
    count = t_v.shape[0]
    fit = np.zeros((count, count), dtype=np.complex128)
    residual = np.zeros(count)
    for group in groups:
        cols = list(group)
        v2 = t_v[:, cols]
        v3 = reference[:, cols]
        norms = np.linalg.norm(v3, axis=0)
        if np.any(norms == 0.0):
            raise ValueError("A reference mode has zero terminal voltage.")
        sol, *_ = np.linalg.lstsq(v2, v3, rcond=None)
        fit[np.ix_(cols, cols)] = sol
        residual[cols] = np.linalg.norm(v3 - v2 @ sol, axis=0) / norms
    return (
        np.asarray(t_v @ fit, dtype=np.complex128),
        np.asarray(t_i @ fit, dtype=np.complex128),
        residual,
        np.asarray(np.linalg.norm(fit, axis=0), dtype=float),
    )


def align_gauge(
    modes_2d: Sequence[PortModes2D],
    *,
    freq_ghz: ArrayLike,
    wave_numbers: ArrayLike,
    terminal_names: Sequence[str] = (),
    pilot: Sequence[int | str] | None = None,
    reference_voltages: Sequence[ArrayLike] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
) -> AlignedPortModes:
    """Bring the 2D ``T_V`` and ``T_I`` of a port into the gauge of its 3D modes.

    The modal S-parameters from Palace and the terminal matrices must describe
    the same modes with the same scale and phase at every port; a scaling that
    differs from port to port changes the terminal S. This does three things:

    1. **Pairing.** At every 2D frequency the 3D modes (the wave numbers
       Palace printed, interpolated to that frequency) are paired with the 2D
       modes by ``n_eff`` (an optimal assignment), so a swap of the mode order
       with frequency does no harm. The pair must agree within ``rtol``.
    2. **Magnitude.** The 2D columns already have ``|2 P| = 1``
       (:func:`port_terminal_matrices`), Palace's unit-power convention.
    3. **Phase and sign.** Every column is rotated so that the voltage on its
       *pilot* terminal is real and positive. Palace's mode phase is set by a
       fixed rule and its sign so that the voltage along the port's
       ``VoltagePath`` is positive; for a low-loss quasi-TEM mode the two
       agree when the 3D pilot path is the 2D pilot terminal's path. That is
       an assumption, checked only by a comparison with a run (see the
       guide); where it is doubtful pass ``reference_voltages`` instead.

    With ``reference_voltages`` (the terminal voltages of Palace's own saved
    port modes, one ``(N, N)`` matrix per 2D solve, column ``k`` for the 3D mode
    ``k``) the gauge comes from the 3D fields: every mode (a degenerate group as
    a block) is fitted ``V_3D ~ V_2D M`` and ``T_V``, ``T_I`` are multiplied by
    ``M``. The fit residual measures the agreement of the 2D and 3D modes.

    The aligned matrices are then interpolated (real and imaginary parts, linearly
    in the coordinate ``interpolation`` names) from the 2D frequencies to
    ``freq_ghz``. The default is ``1 / f``: the modes of a lossy line depart
    from those of the lossless one by terms like ``R / (w L)`` and
    ``G / (w C)``, so ``T_V`` varies most at low frequency and is nearly flat
    above. On a synthetic lossy line (1-100 GHz, 5 solves) the largest ``T_V``
    error was 1e-4 with ``1 / f``, 1.7e-2 with ``"log"`` and 4.8e-2 with
    ``"linear"``. A real line also has dispersion and skin effect, so check the
    choice with a 2D solve at a frequency not used for the interpolation.

    Args:
        modes_2d: The 2D solves of the port, at one or more frequencies.
        freq_ghz: Target frequencies in GHz, strictly increasing.
        wave_numbers: ``(nf, N)`` complex 3D wave numbers (1/m) of the port's
            modes at ``freq_ghz``, column ``k`` for mode ``k``
            (:attr:`ModalS.wave_numbers` for the port's columns).
        terminal_names: Names of the terminals, to give ``pilot`` as names.
        pilot: Per 3D mode, the terminal (index or name) that sets the phase.
            Default: the terminal with the largest voltage in the lowest 2D
            frequency's paired mode, kept for all frequencies. Use the same
            terminal for the 3D ``VoltagePath`` of that mode.
        reference_voltages: See above; same length and order as ``modes_2d``.
        rtol: Largest relative ``n_eff`` difference of a 2D mode and the 3D mode
            it is paired with. It may be opened (for example to 0.15) to pair
            2D modes that differ from the 3D ones by more than the default,
            which says nothing about whether two 3D modes can be told apart.
        degenerate_rtol: Distance under which two 3D modes count as
            indistinguishable, so that their 2D basis need not be Palace's
            (default: the face-mode tolerance of the port-mode checks, 0.5 %).
            Separate from ``rtol`` on purpose: opening ``rtol`` does not
            declare modes a few percent apart degenerate.
        allow_degenerate: Accept indistinguishable 3D modes without
            ``reference_voltages``. Their 2D basis then need not be the 3D
            one and the terminal S can be wrong; a ``ValueError`` is the default.
        extrapolate: Hold the end values for target frequencies outside the
            range of the 2D solves (otherwise that is an error).
        interpolation: ``"inverse"`` (``1 / f``, default), ``"linear"`` or ``"log"``.

    Returns:
        The aligned matrices on the target grid, and the audit numbers.

    Raises:
        ValueError: For an unknown ``interpolation``, malformed input,
            missing wave numbers, a 2D solve
            outside the 3D wave-number range or target frequencies outside
            the 2D range without ``extrapolate``, a pair of modes that do not
            agree within ``rtol``, a negative or non-finite ``degenerate_rtol``,
            indistinguishable modes (within ``degenerate_rtol``) without a way
            to fix their basis, or a zero pilot voltage.
    """
    from scipy.optimize import linear_sum_assignment

    _axis(np.ones(1), interpolation)  # refuse an unknown name before any work
    if not (np.isfinite(degenerate_rtol) and degenerate_rtol >= 0.0):
        raise ValueError(
            f"degenerate_rtol must be a finite number >= 0, got {degenerate_rtol}."
        )
    freq = _check_grid(freq_ghz)
    kn = np.asarray(wave_numbers, dtype=np.complex128)
    if not modes_2d:
        raise ValueError("align_gauge needs at least one 2D solve.")
    count = modes_2d[0].n_eff.size
    if kn.shape != (freq.size, count):
        raise ValueError(
            f"wave_numbers has shape {kn.shape}, expected {(freq.size, count)}: one "
            "column per mode of the port."
        )
    if not np.all(np.isfinite(kn)):
        missing = sorted({float(f) for f in freq[~np.all(np.isfinite(kn), axis=1)]})
        raise ValueError(
            f"The 3D wave numbers are missing at {missing} GHz: read the run's log "
            "(load_modal_run(..., log=...)) for every frequency."
        )
    n_3d = kn.real / _k0(freq)[:, None]
    if np.any(n_3d <= 0.0):
        raise ValueError("A 3D mode has a non-positive effective index.")
    if reference_voltages is not None and len(reference_voltages) != len(modes_2d):
        raise ValueError(
            f"reference_voltages has {len(reference_voltages)} entries for "
            f"{len(modes_2d)} 2D solves."
        )
    refs = list(reference_voltages) if reference_voltages is not None else None
    ranked = sorted(range(len(modes_2d)), key=lambda p: modes_2d[p].freq_ghz)
    solves = [modes_2d[p] for p in ranked]
    f_2d = np.array([m.freq_ghz for m in solves])
    if np.any(np.diff(f_2d) <= 0.0):
        raise ValueError("Two 2D solves have the same frequency.")
    if any(m.n_eff.size != count for m in solves):
        raise ValueError("The 2D solves do not all have the same number of modes.")
    outside = freq[0] < f_2d[0] * (1 - 1e-9) or freq[-1] > f_2d[-1] * (1 + 1e-9)
    if outside and not extrapolate:
        raise ValueError(
            f"The 2D solves cover {f_2d[0]:g} to {f_2d[-1]:g} GHz but the S-parameters "
            f"go from {freq[0]:g} to {freq[-1]:g} GHz; add 2D solves, or pass "
            "extrapolate=True to hold the end values."
        )

    pairing: list[tuple[int, ...]] = []
    degenerate: set[tuple[int, ...]] = set()
    mismatch = 0.0
    aligned_v: list[NDArray[np.complex128]] = []
    aligned_i: list[NDArray[np.complex128]] = []
    residuals: list[NDArray[np.float64]] = []
    scales: list[NDArray[np.float64]] = []
    rows: tuple[int, ...] | None = None
    for position, solve in enumerate(solves):
        n_at = np.array(
            [np.interp(solve.freq_ghz, freq, n_3d[:, k]) for k in range(count)]
        )
        cost = np.abs(n_at[:, None] - solve.n_eff[None, :]) / n_at[:, None]
        three_d, two_d = linear_sum_assignment(cost)
        worst = float(cost[three_d, two_d].max())
        if worst > rtol:
            bad = [int(k) + 1 for k in three_d[cost[three_d, two_d] > rtol]]
            raise ValueError(
                f"At {solve.freq_ghz:g} GHz the 2D modes cannot be paired with the 3D "
                f"modes {bad}: the closest n_eff differs by {worst:.2%} "
                f"(limit {rtol:.2%}). 3D n_eff {np.round(n_at, 4).tolist()}, "
                f"2D n_eff {np.round(solve.n_eff, 4).tolist()}."
            )
        mismatch = max(mismatch, worst)
        perm = tuple(int(c) for c in two_d)
        pairing.append(perm)
        groups = degenerate_mode_groups(n_at, rtol=degenerate_rtol)
        indistinct = [g for g in groups if len(g) > 1]
        degenerate.update(indistinct)
        t_v = solve.t_v[:, perm]
        t_i = solve.t_i[:, perm]
        if refs is not None:
            reference = np.asarray(refs[ranked[position]], dtype=np.complex128)
            if reference.shape != (count, count):
                raise ValueError(
                    f"reference_voltages[{ranked[position]}] has shape "
                    f"{reference.shape}, expected {(count, count)}."
                )
            t_v, t_i, residual, scale = _fit_gauge(t_v, t_i, reference, groups)
            residuals.append(residual)
            scales.append(scale)
        else:
            if indistinct and not allow_degenerate:
                raise ValueError(
                    f"At {solve.freq_ghz:g} GHz the 3D modes "
                    f"{[[k + 1 for k in g] for g in indistinct]} have the same n_eff "
                    f"within {degenerate_rtol:.2%}: their 2D basis need not be "
                    "Palace's, which "
                    "changes the terminal S. Pass reference_voltages= (from the "
                    "saved port-mode fields) or allow_degenerate=True."
                )
            if rows is None:
                rows = _pilot_rows(t_v, terminal_names, pilot)
            t_v = t_v.copy()
            t_i = t_i.copy()
            for k in range(count):
                voltage = t_v[rows[k], k]
                if abs(voltage) == 0.0:
                    raise ValueError(
                        f"Mode {k + 1} has zero voltage on its pilot terminal at "
                        f"{solve.freq_ghz:g} GHz."
                    )
                phase = np.conj(voltage) / abs(voltage)
                t_v[:, k] *= phase
                t_i[:, k] *= phase
        aligned_v.append(t_v)
        aligned_i.append(t_i)

    return AlignedPortModes(
        freq_ghz=freq,
        t_v=_interpolate_matrices(f_2d, aligned_v, freq, interpolation),
        t_i=_interpolate_matrices(f_2d, aligned_i, freq, interpolation),
        pilot=() if rows is None else rows,
        pairing=tuple(pairing),
        mismatch=mismatch,
        degenerate=tuple(sorted(degenerate)),
        gauge_residual=np.array(residuals) if residuals else None,
        gauge_scale=np.array(scales) if scales else None,
    )


#: The frequency axes :func:`align_gauge` can interpolate the 2D matrices on.
INTERPOLATIONS: tuple[str, ...] = ("inverse", "linear", "log")


def _axis(freq: NDArray[np.float64], interpolation: str) -> NDArray[np.float64]:
    """The increasing coordinate that the matrices are interpolated along."""
    if interpolation == "inverse":
        return np.asarray(-1.0 / freq, dtype=float)
    if interpolation == "linear":
        return freq
    if interpolation == "log":
        return np.asarray(np.log(freq), dtype=float)
    raise ValueError(
        f"interpolation must be one of {list(INTERPOLATIONS)}, got {interpolation!r}."
    )


def _interpolate_matrices(
    f_2d: NDArray[np.float64],
    stack: Sequence[NDArray[np.complex128]],
    freq: NDArray[np.float64],
    interpolation: str,
) -> NDArray[np.complex128]:
    """Interpolate ``(N, N)`` matrices given at ``f_2d`` onto ``freq``.

    Real and imaginary parts are interpolated linearly in the chosen
    coordinate, entry by entry; outside ``f_2d`` the end values are held.
    """
    data = np.asarray(stack, dtype=np.complex128)
    count = data.shape[1]
    x_2d = _axis(f_2d, interpolation)
    x = _axis(freq, interpolation)
    out = np.empty((freq.size, count, count), dtype=np.complex128)
    for i in range(count):
        for j in range(count):
            out[:, i, j] = np.interp(x, x_2d, data[:, i, j].real) + 1j * np.interp(
                x, x_2d, data[:, i, j].imag
            )
    return out


# ---------------------------------------------------------------------------
# Terminal S-parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TerminalSResult:
    """Terminal S-parameters of a simulation, with what they were built from.

    Attributes:
        freq_ghz: Frequencies in GHz.
        s: ``(nf, Nt, Nt)`` terminal S, ordered port by port (Palace port
            number), terminals in the order of each :class:`TerminalPortSpec`.
        port_names: ``"{port}:{terminal}"`` per terminal, the name of a lumped
            terminal for a lumped port.
        z_ref: ``(Nt,)`` real reference impedance of each terminal.
        modal: The assembled modal S.
        aligned: The aligned ``T_V``, ``T_I`` of every multimode port.
    """

    freq_ghz: NDArray[np.float64]
    s: NDArray[np.complex128]
    port_names: tuple[str, ...]
    z_ref: NDArray[np.float64]
    modal: ModalS
    aligned: dict[str, AlignedPortModes]

    def network(self, name: str | None = None) -> Network:
        """The terminal S as a scikit-rf network (one port per terminal).

        Args:
            name: Name of the network.

        Returns:
            The network, with the port names and per-port reference impedances.
        """
        import skrf as rf

        z0 = np.broadcast_to(self.z_ref, self.s.shape[:2]).copy()
        net = rf.Network(f=self.freq_ghz * 1e9, s=self.s, z0=z0, f_unit="Hz")
        net.name = name
        net.port_names = list(self.port_names)
        return net

    def write_touchstone(self, path: str | Path, *, name: str | None = None) -> Path:
        """Write the terminal S to a Touchstone ``.sNp`` file.

        The reference impedance of every port is written as a comment as well
        (``write_z0``), and the values with 16 digits so that a round trip
        keeps the S-parameters to far below 1e-9.

        Args:
            path: Destination; ``.sNp`` is appended when the suffix is missing.
            name: Name of the network.

        Returns:
            The file written.
        """
        target = Path(path)
        count = len(self.port_names)
        if target.suffix.lower() != f".s{count}p":
            target = Path(f"{target}.s{count}p")
        fmt = "{:.16e}"
        self.network(name).write_touchstone(
            target,
            form="ri",
            format_spec_A=fmt,
            format_spec_B=fmt,
            format_spec_freq=fmt,
            write_z0=True,
        )
        return target


def _port_z_ref(
    z_ref: float | Mapping[str, float | Sequence[float]], port: str, count: int
) -> NDArray[np.float64]:
    """The reference impedances of a port's terminals, validated."""
    if isinstance(z_ref, Mapping):
        if port not in z_ref:
            raise ValueError(f"z_ref has no entry for port {port!r}.")
        value = z_ref[port]
    else:
        value = z_ref
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        if np.any(arr.imag != 0):
            raise ValueError("Complex reference impedances are not supported.")
        arr = arr.real
    out = np.broadcast_to(np.asarray(arr, dtype=float), (count,)).copy()
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"z_ref of port {port!r} must be finite and positive.")
    return out


def build_terminal_s(
    runs: Sequence[ModalRun | str | Path | dict],
    specs: Mapping[str, TerminalPortSpec] | Sequence[TerminalPortSpec],
    modes_2d: Mapping[str, Sequence[PortModes2D]],
    *,
    z_ref: float | Mapping[str, float | Sequence[float]] = 50.0,
    lumped: Sequence[LumpedTerminal] = (),
    pilot: Mapping[str, Sequence[int | str]] | None = None,
    reference_voltages: Mapping[str, Sequence[ArrayLike]] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
) -> TerminalSResult:
    """Terminal S-parameters from driven runs and 2D mode solves.

    The pipeline: :func:`modal_s_from_runs` assembles the modal S;
    :func:`align_gauge` brings each port's 2D ``T_V`` and ``T_I`` into the gauge
    of its 3D modes on the S-parameter frequencies; lumped ports get the
    pass-through block; :func:`gsim.palace.terminal.modal_to_terminal_s` does the
    transformation.

    Args:
        runs: The driven runs (one for a single run, one per mode for cycling).
            Read with :func:`load_modal_run` including the log, for the wave
            numbers.
        specs: The terminal layout of each multimode port, by port name (or a
            list; the port names are the specs' names).
        modes_2d: Per port name, its 2D solves (see :func:`port_modes_2d`).
        z_ref: Reference impedance of the terminals in ohm: a number for all, or
            ``{port: number or one value per terminal}``. Lumped terminals
            use their own resistance.
        lumped: Lumped ports to include in the network.
        pilot: Per port name, the pilot terminal per mode (see
            :func:`align_gauge`).
        reference_voltages: Per port name, the reference voltages of the 3D
            modes per 2D solve (see :func:`align_gauge`).
        rtol: Pairing tolerance on ``n_eff``.
        degenerate_rtol: Tolerance under which 3D modes count as degenerate (see
            :func:`align_gauge`).
        allow_degenerate: See :func:`align_gauge`.
        extrapolate: See :func:`align_gauge`.
        interpolation: See :func:`align_gauge`.

    Returns:
        The terminal S with its modal S and aligned matrices.

    Raises:
        ValueError: When specs or 2D solves do not match the ports of the runs,
            a port has a different number of modes and terminals, the runs carry
            no wave numbers, or any step above refuses its input.
    """
    modal = modal_s_from_runs(runs, lumped=lumped)
    spec_by_name = (
        dict(specs)
        if isinstance(specs, Mapping)
        else {spec.name: spec for spec in specs}
    )
    lumped_by_number = {item.index: item for item in lumped}
    ports = {item.portnumber: item.port for item in modal.order if item.mode > 0}
    wanted = set(ports.values())
    if wanted != set(spec_by_name):
        raise ValueError(
            f"The terminal specs are for ports {sorted(spec_by_name)} but the runs "
            f"have the multimode ports {sorted(wanted)}."
        )
    if missing := sorted(wanted - set(modes_2d)):
        raise ValueError(f"No 2D solves were given for the ports {missing}.")

    t_vs: list[NDArray[np.complex128]] = []
    t_is: list[NDArray[np.complex128]] = []
    names: list[str] = []
    zr: list[NDArray[np.float64]] = []
    aligned: dict[str, AlignedPortModes] = {}
    nf = modal.freq_ghz.size
    for number in sorted({item.portnumber for item in modal.order}):
        if number in lumped_by_number and number not in ports:
            item = lumped_by_number[number]
            root = np.sqrt(item.resistance)
            t_vs.append(np.full((nf, 1, 1), root, dtype=np.complex128))
            t_is.append(np.full((nf, 1, 1), 1.0 / root, dtype=np.complex128))
            names.append(item.name)
            zr.append(np.array([item.resistance]))
            continue
        port = ports[number]
        spec = spec_by_name[port]
        cols = modal.port_columns(port)
        if len(cols) != len(spec.terminals):
            raise ValueError(
                f"Port {port!r} has {len(cols)} modes in the runs but "
                f"{len(spec.terminals)} terminals: they must match."
            )
        if modal.wave_numbers is None:
            raise ValueError(
                "The runs carry no wave numbers: load them with "
                "load_modal_run(..., log=<the Palace log>)."
            )
        result = align_gauge(
            modes_2d[port],
            freq_ghz=modal.freq_ghz,
            wave_numbers=modal.wave_numbers[:, cols],
            terminal_names=spec.terminal_names,
            pilot=None if pilot is None else pilot.get(port),
            reference_voltages=(
                None if reference_voltages is None else reference_voltages.get(port)
            ),
            rtol=rtol,
            degenerate_rtol=degenerate_rtol,
            allow_degenerate=allow_degenerate,
            extrapolate=extrapolate,
            interpolation=interpolation,
        )
        aligned[port] = result
        t_vs.append(result.t_v)
        t_is.append(result.t_i)
        names.extend(f"{port}:{terminal}" for terminal in spec.terminal_names)
        zr.append(_port_z_ref(z_ref, port, len(spec.terminals)))
    z_vec = np.concatenate(zr)
    s_terminal = modal_to_terminal_s(modal.s_modal, t_vs, t_is, z_ref=z_vec)
    return TerminalSResult(
        freq_ghz=modal.freq_ghz,
        s=s_terminal,
        port_names=tuple(names),
        z_ref=z_vec,
        modal=modal,
        aligned=aligned,
    )


def terminal_s(
    runs: Sequence[ModalRun | str | Path | dict],
    specs: Mapping[str, TerminalPortSpec] | Sequence[TerminalPortSpec],
    modes_2d: Mapping[str, Sequence[PortModes2D]],
    *,
    z_ref: float | Mapping[str, float | Sequence[float]] = 50.0,
    lumped: Sequence[LumpedTerminal] = (),
    pilot: Mapping[str, Sequence[int | str]] | None = None,
    reference_voltages: Mapping[str, Sequence[ArrayLike]] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
    name: str | None = None,
) -> Network:
    """Terminal S-parameters as a scikit-rf network, one port per terminal.

    The public entry point: see :func:`build_terminal_s` for the arguments and
    what is done, and use that function when the modal S, the pairing and the
    aligned matrices are wanted as well. Ports are named
    ``"{port}:{terminal}"``; write a Touchstone file with
    ``network.write_touchstone`` or :meth:`TerminalSResult.write_touchstone`.

    Args:
        runs: The driven runs.
        specs: The terminal layout of each multimode port.
        modes_2d: Per port name, its 2D solves.
        z_ref: Reference impedance of the terminals in ohm.
        lumped: Lumped ports to include.
        pilot: Per port, the pilot terminal per mode.
        reference_voltages: Per port, the 3D reference voltages per 2D solve.
        rtol: Pairing tolerance on ``n_eff``.
        degenerate_rtol: Tolerance under which 3D modes count as degenerate.
        allow_degenerate: See :func:`align_gauge`.
        extrapolate: See :func:`align_gauge`.
        interpolation: See :func:`align_gauge`.
        name: Name of the network.

    Returns:
        The terminal network.
    """
    return build_terminal_s(
        runs,
        specs,
        modes_2d,
        z_ref=z_ref,
        lumped=lumped,
        pilot=pilot,
        reference_voltages=reference_voltages,
        rtol=rtol,
        degenerate_rtol=degenerate_rtol,
        allow_degenerate=allow_degenerate,
        extrapolate=extrapolate,
        interpolation=interpolation,
    ).network(name)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _named_ports(
    net: Network, names: Sequence[str], what: str
) -> tuple[list[str], list[int]]:
    """The port names of a network and the positions of the named ports.

    Raises:
        ValueError: If the network has no port names, a name is unknown or a
            name is repeated.
    """
    known = [str(n) for n in (net.port_names or [])]
    if len(known) != net.nports:
        raise ValueError(f"{what} needs a network with one name per port.")
    if len(set(names)) != len(list(names)):
        raise ValueError(f"{what}: a port name is repeated in {list(names)}.")
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"{what}: unknown ports {unknown}; the network has {known}.")
    return known, [known.index(n) for n in names]


def _real_z0(net: Network, what: str) -> NDArray[np.float64]:
    """The reference impedances of a network, which must be real."""
    z0 = np.asarray(net.z0)
    if np.any(z0.imag != 0.0):
        raise ValueError(f"{what}: complex port impedances are not supported.")
    return np.asarray(z0.real, dtype=float)


def short_terminals(net: Network, names: Sequence[str]) -> Network:
    """Short the named terminals to the reference: ``V = 0`` on them.

    The result is the network seen from the remaining terminals when the named
    ones are tied to the reference conductor. It equals the sub-block of the
    admittance matrix of the remaining terminals, computed here directly on
    the S-parameters (a short is the reflection ``-1`` for real reference
    impedances): ``S' = S_kk - S_kn (I + S_nn)^-1 S_nk`` with ``n`` the shorted
    and ``k`` the kept ports. That also works when ``Y`` does not exist.

    Shorting terminals at the ports of a line ties them at its two ends only,
    not along its length; the two agree for an electrically short line.

    Args:
        net: A network with named ports and real reference impedances.
        names: The terminals to short, at least one and not all.

    Returns:
        A new network with the other ports, in their order, named as before.

    Raises:
        ValueError: For unknown, repeated or all port names, complex reference
            impedances, or a set of shorted ports whose ``I + S_nn`` is
            singular.
    """
    import skrf as rf

    known, short = _named_ports(net, names, "short_terminals")
    if not short:
        raise ValueError("short_terminals needs at least one port name.")
    keep = [i for i in range(net.nports) if i not in short]
    if not keep:
        raise ValueError("short_terminals cannot short every port.")
    z0 = _real_z0(net, "short_terminals")
    s = np.asarray(net.s, dtype=np.complex128)
    s_kk = s[:, keep][:, :, keep]
    s_kn = s[:, keep][:, :, short]
    s_nk = s[:, short][:, :, keep]
    s_nn = s[:, short][:, :, short]
    try:
        solved = np.linalg.solve(np.eye(len(short)) + s_nn, s_nk)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The shorted ports cannot be tied to the reference: I + S_nn is singular."
        ) from error
    reduced = s_kk - s_kn @ solved
    out = rf.Network(frequency=net.frequency, s=reduced, z0=z0[:, keep])
    out.s_def = net.s_def
    out.port_names = [known[i] for i in keep]
    return out


def mixed_mode(
    net: Network,
    pairs: Sequence[tuple[str, str]],
    *,
    z0_mm: ArrayLike | None = None,
) -> Network:
    """Mixed-mode network of terminal pairs, by name.

    Wraps ``Network.se2gmm``, which works in place on ports in a fixed order and
    returns ``None``: the ports are taken by name, put in the order scikit-rf
    wants (pair 1, pair 2, ..., the others), converted on a copy, and returned.
    The result has the ``p`` differential ports first, the ``p`` common-mode
    ports next and the unpaired terminals last; scikit-rf names the mixed ports
    ``d(<first>,<second>)`` and ``c(<first>,<second>)``. The differential
    voltage is ``first - second`` (swap a pair to flip the sign of ``S_dc``).

    By default the reference impedances are twice (differential) and half
    (common) the mean of the pair's single-ended ones.

    Args:
        net: A network with named ports.
        pairs: ``(first, second)`` port names per differential port.
        z0_mm: Optional mixed-mode reference impedances, as in ``se2gmm``.

    Returns:
        A new mixed-mode network; ``net`` is unchanged.

    Raises:
        ValueError: For no pairs, a pair that is not two names, or unknown
            or repeated names.
    """
    import skrf as rf

    pair_list = [tuple(pair) for pair in pairs]
    if not pair_list or any(len(pair) != 2 for pair in pair_list):
        raise ValueError("mixed_mode needs one or more (first, second) pairs of names.")
    flat = [name for pair in pair_list for name in pair]
    known, idx = _named_ports(net, flat, "mixed_mode")
    rest = [i for i in range(net.nports) if i not in idx]
    z0 = _real_z0(net, "mixed_mode")
    # The transform is written out instead of calling Network.se2gmm, whose
    # pairing and port naming differ between scikit-rf versions. For a pair
    # (a, b) the waves are d = (a - b)/sqrt(2) and c = (a + b)/sqrt(2); with
    # the reference impedances 2 z0 and z0/2 this is exact when a and b share
    # z0, so the pair's mean is used, as se2gmm does.
    n_pairs, n = len(pair_list), net.nports
    m = np.zeros((n, n))
    z0_out = np.zeros((z0.shape[0], n))
    names = []
    for k, (ia, ib) in enumerate(zip(idx[0::2], idx[1::2], strict=True)):
        m[k, ia], m[k, ib] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        m[n_pairs + k, ia], m[n_pairs + k, ib] = 1 / np.sqrt(2), 1 / np.sqrt(2)
        mean = 0.5 * (z0[:, ia] + z0[:, ib])
        z0_out[:, k], z0_out[:, n_pairs + k] = 2 * mean, mean / 2
        names.append((k, f"d({known[ia]},{known[ib]})"))
        names.append((n_pairs + k, f"c({known[ia]},{known[ib]})"))
    for j, i in enumerate(rest):
        m[2 * n_pairs + j, i] = 1.0
        z0_out[:, 2 * n_pairs + j] = z0[:, i]
        names.append((2 * n_pairs + j, known[i]))
    s_mm = np.einsum("ij,fjk,lk->fil", m, np.asarray(net.s), m)
    out = rf.Network(frequency=net.frequency, s=s_mm, z0=z0_out)
    out.s_def = net.s_def
    out.port_names = [name for _, name in sorted(names)]
    if getattr(out, "port_modes", None) is not None:  # scikit-rf 2.x only
        out.port_modes = np.array(["D"] * n_pairs + ["C"] * n_pairs + ["S"] * len(rest))
    if z0_mm is not None:
        out.renormalize(np.asarray(z0_mm, dtype=complex))
    return out
