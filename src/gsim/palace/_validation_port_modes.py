"""Continuity check of the wave-port mode that Palace selects at each frequency.

Palace picks the port mode only by its index, so a spurious or closed-box
solution (gsim#375, gsim#408) is silent in the S-parameters. A physical port
mode changes slowly with frequency, so an isolated jump of its effective index
``n_eff = Re(k_n) / k0`` is evidence that a different mode was selected at that
frequency. The check reads the samples from the driven log and never reruns a
simulation. A hit is a ``warning``: a mode that really changes quickly, for
example close to cutoff or on a very coarse sweep, is flagged in the same way.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

from gsim.palace._validation_models import ModeSample, SolutionCheck

SPEED_OF_LIGHT_M_S = 299_792_458.0
"""Vacuum speed of light in m/s, used for ``k0 = 2 pi f / c``."""

CONTINUITY_RTOL = 0.01
"""Largest relative distance of ``n_eff`` from the median of its nearest
neighbours (in frequency) before a point is flagged. The good points of the
real test logs scatter by at most 0.05 %, the spurious solves differ by 17 % or
more. A mode that really changes by more than this between neighbouring
frequencies (close to cutoff, very coarse sweep) is flagged as well."""

PORT_AGREEMENT_RTOL = 0.005
"""Largest relative difference of ``n_eff`` between two ports solved at the same
frequency (only when the ports share a cross-section). The two identical ports
of the real test logs agree exactly in the printed digits; the spurious pairs
differ by 11 % and 49 %."""

NEIGHBOUR_COUNT = 5
"""Number of nearest points (by frequency, same port, mode and excitation) whose
median is the continuity reference. Up to two spurious neighbours are tolerated."""

MIN_DISTINCT_FREQUENCIES = 6
"""A series with fewer distinct frequencies is not judged: continuity cannot be
tested on so few points, so the result is ``unknown``, never ``passed``."""

MAX_EVIDENCE = 5
"""Number of flagged points listed in a check; the rest are only counted."""

CONTINUITY_CHECK = "port_mode_continuity"
PORT_AGREEMENT_CHECK = "port_mode_ports"

_Key = tuple[int, int, int | None]
_Point = tuple[float, float]  # (frequency in GHz, effective index)


def _effective_index(sample: ModeSample, frequency_ghz: float) -> float:
    """Return ``Re(k_n) / k0``; NaN if the wave number is not finite."""
    wave_number = sample.wave_number
    if not (math.isfinite(wave_number.real) and math.isfinite(wave_number.imag)):
        return math.nan
    k0 = 2 * math.pi * frequency_ghz * 1e9 / SPEED_OF_LIGHT_M_S
    return wave_number.real / k0


def _series(samples: Sequence[ModeSample]) -> dict[_Key, list[_Point]]:
    """Group samples by (port, mode, excitation), sorted by frequency.

    Samples without a positive finite frequency are skipped: eigenmode and
    boundary-mode observations carry none, so they are ignored here.
    """
    groups: dict[_Key, list[_Point]] = defaultdict(list)
    for sample in samples:
        frequency = sample.frequency_ghz
        if frequency is None or not math.isfinite(frequency) or frequency <= 0:
            continue
        key = (sample.port, sample.mode, sample.excitation)
        groups[key].append((frequency, _effective_index(sample, frequency)))
    return {key: sorted(points) for key, points in groups.items()}


def _sort_key(key: _Key) -> tuple[int, int, int]:
    """Order series by port, mode and excitation (None sorts first)."""
    port, mode, excitation = key
    return (-1 if excitation is None else excitation, port, mode)


def _label(key: _Key) -> str:
    """Human-readable name of a (port, mode, excitation) series."""
    port, mode, excitation = key
    where = "" if excitation is None else f", excitation {excitation}"
    return f"port {port}, mode {mode}{where}"


def _neighbour_median(points: Sequence[_Point], index: int) -> float | None:
    """Median ``n_eff`` of the nearest other points in frequency.

    ``points`` must be sorted by frequency and finite, so the nearest points
    lie within ``NEIGHBOUR_COUNT`` positions on either side.
    """
    frequency = points[index][0]
    lower = max(0, index - NEIGHBOUR_COUNT)
    candidates = [
        (abs(points[other][0] - frequency), other)
        for other in range(lower, min(len(points), index + NEIGHBOUR_COUNT + 1))
        if other != index
    ]
    candidates.sort()
    nearest = [points[other][1] for _, other in candidates[:NEIGHBOUR_COUNT]]
    return statistics.median(nearest) if nearest else None


def _distinct_frequencies(points: Sequence[_Point]) -> int:
    """Number of different frequencies in a series."""
    return len({frequency for frequency, _ in points})


def _summary(flagged: Sequence[str], total: int, what: str) -> tuple[str, ...]:
    """Return evidence lines: a count, the first points, then the remainder."""
    lines = [f"{len(flagged)} of {total} {what} flagged."]
    lines.extend(flagged[:MAX_EVIDENCE])
    if len(flagged) > MAX_EVIDENCE:
        lines.append(f"... and {len(flagged) - MAX_EVIDENCE} more.")
    return tuple(lines)


def _continuity(series: dict[_Key, list[_Point]], rtol: float) -> SolutionCheck:
    """Flag points whose n_eff departs from the median of their frequency neighbours."""
    flagged: list[str] = []
    not_judged: list[str] = []
    judged = 0
    solves = sum(len(points) for points in series.values())
    for key in sorted(series, key=_sort_key):
        points = series[key]
        for frequency, n_eff in points:
            if not math.isfinite(n_eff):
                flagged.append(
                    f"{_label(key)}: nonfinite wave number at {frequency:.4g} GHz"
                )
        finite = [point for point in points if math.isfinite(point[1])]
        distinct = _distinct_frequencies(finite)
        if distinct < MIN_DISTINCT_FREQUENCIES:
            not_judged.append(
                f"{_label(key)}: only {distinct} distinct frequencies "
                f"(need {MIN_DISTINCT_FREQUENCIES}); not judged."
            )
            continue
        judged += 1
        for index, (frequency, n_eff) in enumerate(finite):
            reference = _neighbour_median(finite, index)
            if reference is None or abs(n_eff - reference) <= rtol * abs(reference):
                continue
            deviation = 100 * abs(n_eff - reference) / abs(reference)
            flagged.append(
                f"{_label(key)}: n_eff = {n_eff:.4f} at {frequency:.4g} GHz, "
                f"neighbour median {reference:.4f} "
                f"({deviation:.2f} % > {100 * rtol:.2g} %)"
            )
    if flagged:
        return SolutionCheck(
            CONTINUITY_CHECK, "warning", _summary(flagged, solves, "port solves")
        )
    if not judged:
        return SolutionCheck(CONTINUITY_CHECK, "unknown", tuple(not_judged))
    return SolutionCheck(
        CONTINUITY_CHECK,
        "passed",
        (
            f"{judged} series, {solves} solves: no n_eff differs more than "
            f"{100 * rtol:.2g} % from its neighbour median.",
            *not_judged,
        ),
    )


def _port_agreement(series: dict[_Key, list[_Point]], rtol: float) -> SolutionCheck:
    """Flag frequencies where ports solved for the same mode disagree in n_eff."""
    groups: dict[tuple[int, int | None], dict[int, list[_Point]]] = defaultdict(dict)
    for (port, mode, excitation), points in series.items():
        groups[mode, excitation][port] = points
    flagged: list[str] = []
    notes: list[str] = []
    compared = 0
    ordered = sorted(
        groups.items(),
        key=lambda item: (-1 if item[0][1] is None else item[0][1], item[0][0]),
    )
    for (mode, excitation), ports in ordered:
        if len(ports) < 2:
            continue
        first, *others = sorted(ports)
        where = "" if excitation is None else f", excitation {excitation}"
        for other in others:
            reference = defaultdict(list)
            for frequency, n_eff in ports[first]:
                reference[frequency].append(n_eff)
            shared = 0
            for frequency, n_eff in ports[other]:
                for n_first in reference.get(frequency, ()):
                    shared += 1
                    scale = max(abs(n_first), abs(n_eff))
                    if not (math.isfinite(n_first) and math.isfinite(n_eff)):
                        continue  # reported by the continuity check
                    if abs(n_first - n_eff) > rtol * scale:
                        flagged.append(
                            f"mode {mode}{where}: ports {first} and {other} differ "
                            f"by {100 * abs(n_first - n_eff) / scale:.2f} % at "
                            f"{frequency:.4g} GHz (n_eff {n_first:.4f} vs {n_eff:.4f}, "
                            f"limit {100 * rtol:.2g} %)"
                        )
            if shared:
                compared += shared
            else:
                notes.append(
                    f"mode {mode}{where}: ports {first} and {other} share no "
                    "solve frequency; not compared."
                )
    if flagged:
        return SolutionCheck(
            PORT_AGREEMENT_CHECK,
            "warning",
            _summary(flagged, compared, "port pairs"),
        )
    if not compared:
        return SolutionCheck(
            PORT_AGREEMENT_CHECK,
            "unknown",
            (*notes, "No two ports were solved at a common frequency.")
            if notes
            else ("Only one port found; nothing to compare.",),
        )
    return SolutionCheck(
        PORT_AGREEMENT_CHECK,
        "passed",
        (
            f"{compared} port pairs agree within {100 * rtol:.2g} %.",
            *notes,
        ),
    )


def check_port_mode_continuity(
    samples: Sequence[ModeSample],
    *,
    compare_ports: bool = False,
    rtol: float = CONTINUITY_RTOL,
    port_rtol: float = PORT_AGREEMENT_RTOL,
) -> tuple[SolutionCheck, ...]:
    """Check that the selected wave-port mode varies smoothly with frequency.

    For every (port, mode, excitation) series the effective index
    ``n_eff = Re(k_n) / k0`` is compared with the median of its
    ``NEIGHBOUR_COUNT`` nearest points in frequency. Points further than
    ``rtol`` (relative) from that median, and nonfinite wave numbers, give a
    ``"warning"`` whose evidence lists port, excitation, frequency, ``n_eff``
    and the neighbour median. A series with fewer than
    ``MIN_DISTINCT_FREQUENCIES`` frequencies is ``"unknown"``.

    Args:
        samples: Port-mode observations, normally ``SolutionReport.mode_samples``.
            Observations without a frequency (eigenmode, boundary mode) are
            ignored.
        compare_ports: Also require ports solved at the same frequency to agree
            within ``port_rtol``. Only set this when all ports share one
            cross-section: ports of different width or stack have different
            modes and would be flagged wrongly.
        rtol: Relative continuity tolerance of ``n_eff``.
        port_rtol: Relative port-to-port tolerance of ``n_eff``.

    Returns:
        ``()`` when no sample has a frequency (no driven wave-port solve),
        otherwise the ``port_mode_continuity`` check followed by the
        ``port_mode_ports`` check when requested. Statuses are ``"passed"``,
        ``"warning"`` or ``"unknown"``; this check never reports ``"failed"``.

    Raises:
        ValueError: If a tolerance is negative or not finite.
    """
    for tolerance in (rtol, port_rtol):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Port-mode tolerances must be finite and nonnegative.")
    series = _series(samples)
    if not series:
        return ()
    checks = [_continuity(series, rtol)]
    if compare_ports:
        checks.append(_port_agreement(series, port_rtol))
    return tuple(checks)
