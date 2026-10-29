r"""Step 6 (PLAN.md): a pure-numpy synthetic model of Palace's multimode wave-port extraction on a lossy uniform two-mode line.  No gsim, no skrf, no solver.

Physics of the model (every assumption is one of the stated ones of PLAN.md section 0.5 / 2; it is a model, not a derivation from Palace's code):

* The line has per-unit-length ``Z = R + jwL``, ``Y = G + jwC`` (2 x 2, symmetric: two terminals ``S_hi``, ``G_hi`` over the reference, i.e. the half model).  Forward waves ``x(z) = exp(-gamma z) [v; i]`` with
  ``(Z Y) v = gamma^2 v``, ``i = Y v / gamma``.  The terminal voltages / currents of a mode are the columns of ``T_V = [v_1 v_2]``, ``T_I = [i_1 i_2]`` (currents flow into the line at both ends; the same
  ``T_V``, ``T_I`` describe both ports).  ``kn = -1j * gamma`` (``gamma = 1j * kn``, the convention of the logged wave numbers): ``Re kn = beta``, ``Im kn = -alpha``.
* Mode fields of the port: ``e_k <-> v_k``, ``h_k <-> i_k`` (terminal proxy of the fields), so the cross-power matrix of the face is ``P_mk = int (e_k x h_m^*) . n  <->  (i_m^P)^H v_k``.
  Palace builds the modal ``n x H`` of the driven path with ``Re(kn)`` (waveportoperator.cpp 350-356, 1096, 1989-2009), so its ``h_m`` is the true ``h_m`` times ``rho_m = Re(kn_m) / kn_m``:
  ``i^P_m = rho_m i_m``.  Every mode is normalised to ``|P_mm| = 1`` by modulus only (``Normalize``, line 156), so ``P_mm`` is a unit-modulus complex number, not 1.
* The port is a PSEUDO-WAVE port with the reference admittance ``rho_m`` (in units of the mode's own admittance): waves ``a = (v + i / rho)/2``, ``b = (v - i / rho)/2``, i.e. ``v = a + b``, ``i = rho (a - b)``.
  The modal reflection / transmission matrix ``B`` of the line between two such ports (``b = B a``) is mode by mode ``B = (rho 1 + Y_l)^-1 (rho 1 - Y_l)`` with the normalised line admittance
  matrix ``Y_l = [[coth gl, -1/sinh gl], [-1/sinh gl, coth gl]]``; for ``rho = 1`` this is ``[[0, exp(-gl)], [exp(-gl), 0]]`` (``gsim.palace.terminal.uniform_line_modal_s``).
* What Palace writes: the conjugated projection of the total field, ``S' = blockdiag(P, P) (I + B) - I`` (the ``-1`` only on the driven entry, i.e. on the diagonal of ``I + B``).

Functions: ``make_line``, ``line_modes``, ``palace_port_modes``, ``ideal_modal_s``, ``pseudo_modal_s``, ``direct_terminal_s`` (truth, from the chain matrix), ``to_terminal_s`` (V = T_V (a + b),
I = T_I (a - b), power waves in ``z_ref``: the transform of ``gsim.palace.terminal.modal_to_terminal_s``, written again here), ``palace_reading``, ``undo_projection``, ``gates``, ``conj_gram``.
"""

from __future__ import annotations

import numpy as np

C0 = 299792458.0


# ---------------------------------------------------------------------------------------------------------------
# the line
# ---------------------------------------------------------------------------------------------------------------
def make_line(f_hz: float, L, C, R=None, G=None) -> dict:
    """Per-unit-length ``Z``, ``Y`` (2 x 2 each) at ``f_hz``; ``R``, ``G`` default to 0 (lossless)."""
    w = 2 * np.pi * f_hz
    L, C = np.asarray(L, float), np.asarray(C, float)
    R = np.zeros_like(L) if R is None else np.asarray(R, float)
    G = np.zeros_like(C) if G is None else np.asarray(G, float)
    return {"f": float(f_hz), "w": w, "Z": R + 1j * w * L, "Y": G + 1j * w * C}


def default_line(lossy: bool = True, f_hz: float = 28e9) -> dict:
    """Two terminals over a reference, inhomogeneous dielectric (two different phase velocities), series resistances that are NOT proportional to L (so the lossy modes are not
    orthogonal under the conjugated product).  Generic numbers chosen to look like a quasi-TEM line (n_eff about 2.1 and 1.9, alpha/beta about 2e-2 and 1e-2)."""
    L = np.array([[0.55e-6, 0.20e-6], [0.20e-6, 0.60e-6]])
    C = np.array([[95e-12, -30e-12], [-30e-12, 70e-12]])
    R = np.diag([3600.0, 1400.0]) if lossy else None
    return make_line(f_hz, L, C, R)


def line_modes(line: dict) -> dict:
    """Eigenmodes of the forward waves, ordered by decreasing ``n_eff`` ... mode 1 = larger Im(gamma) / (k0).  Returns ``gamma`` (2,), ``kn = -1j gamma``, ``v`` (2 x 2, columns), ``i`` (2 x 2).

    Gauge: the voltage of terminal 0 (``S_hi``) of every mode is real and positive (the pilot rule); the scale is the one of the eigen-solver (``palace_port_modes`` fixes it)."""
    z, y = line["Z"], line["Y"]
    w2, vec = np.linalg.eig(z @ y)
    g = np.sqrt(w2.astype(complex))
    g = np.where(g.real < 0, -g, g)  # forward wave: Re gamma >= 0
    order = np.argsort(-g.imag)
    g, vec = g[order], vec[:, order]
    for k in range(vec.shape[1]):
        vec[:, k] = vec[:, k] * (np.conj(vec[0, k]) / abs(vec[0, k]))  # pilot terminal real positive
    cur = (y @ vec) / g[None, :]
    return {"gamma": g, "kn": -1j * g, "v": vec, "i": cur, "n_eff": g.imag * C0 / line["w"], "alpha_over_beta": g.real / g.imag}


def palace_port_modes(modes: dict, scale_h: bool = True) -> dict:
    """The port modes of the face as Palace sees them.

    ``rho = Re(kn) / kn`` (``scale_h = False``: rho = 1, a port that uses the complex kn).  Palace's h of a mode is ``rho`` times the true one.  Each mode is scaled (real, positive) to ``|P_mm| = 1``.
    Returns ``T_V``, ``T_I`` (true terminal currents of the scaled modes), ``rho``, ``q = 1/rho = kn / Re(kn)`` and ``P`` (2 x 2, ``P[m, k] = (rho_m i_m)^H v_k``)."""
    kn = modes["kn"]
    rho = (kn.real / kn) if scale_h else np.ones_like(kn)
    ip = modes["i"] * rho[None, :]
    p_raw = ip.conj().T @ modes["v"]
    s = 1.0 / np.sqrt(np.abs(np.diag(p_raw)))
    t_v, t_i = modes["v"] * s[None, :], modes["i"] * s[None, :]
    p = (t_i * rho[None, :]).conj().T @ t_v
    return {"T_V": t_v, "T_I": t_i, "rho": rho, "q": 1.0 / rho, "P": p}


# ---------------------------------------------------------------------------------------------------------------
# modal S of the line between two ports (4 x 4, order [face 1 mode 1, face 1 mode 2, face 2 mode 1, face 2 mode 2])
# ---------------------------------------------------------------------------------------------------------------
def pseudo_modal_s(gamma, length: float, rho=None) -> np.ndarray:
    """Modal S of a uniform line between pseudo-wave ports of reference admittance ``rho`` (mode by mode; ``rho = 1``: the ideal line)."""
    g = np.asarray(gamma, complex)
    n = g.size
    rho = np.ones(n, complex) if rho is None else np.asarray(rho, complex)
    s = np.zeros((2 * n, 2 * n), complex)
    for k in range(n):
        gl = g[k] * length
        yl = np.array([[1 / np.tanh(gl), -1 / np.sinh(gl)], [-1 / np.sinh(gl), 1 / np.tanh(gl)]])
        sk = np.linalg.solve(rho[k] * np.eye(2) + yl, rho[k] * np.eye(2) - yl)
        for a in range(2):
            for b in range(2):
                s[a * n + k, b * n + k] = sk[a, b]
    return s


def ideal_modal_s(gamma, length: float) -> np.ndarray:
    """``[[0, E], [E, 0]]``, ``E = diag(exp(-gamma l))`` (the formula of ``uniform_line_modal_s``)."""
    g = np.asarray(gamma, complex)
    n = g.size
    e = np.diag(np.exp(-g * length))
    return np.block([[np.zeros((n, n)), e], [e, np.zeros((n, n))]])


# ---------------------------------------------------------------------------------------------------------------
# truth: the terminal S of the line, from the chain matrix, independent of any mode
# ---------------------------------------------------------------------------------------------------------------
def direct_terminal_s(line: dict, length: float, z_ref: float = 50.0) -> np.ndarray:
    """4-port terminal S (ports ``[face 1 terminal 0, terminal 1, face 2 terminal 0, terminal 1]``, currents into the line, equal real ``z_ref``) of the line of ``length`` metres."""
    z, y = line["Z"], line["Y"]
    m = np.block([[np.zeros((2, 2)), z], [y, np.zeros((2, 2))]])
    w, vec = np.linalg.eig(m)
    phi = vec @ np.diag(np.exp(-w * length)) @ np.linalg.inv(vec)  # x(l) = phi x(0), x = [V; I] (I along +z)
    p11, p12, p21, p22 = phi[:2, :2], phi[:2, 2:], phi[2:, :2], phi[2:, 2:]
    # V2 = p11 V1 + p12 I1;  -I2 = p21 V1 + p22 I1   (I2 = current into the line at face 2 = -I(l))
    p12i = np.linalg.inv(p12)
    i1_v1 = -p12i @ p11
    i1_v2 = p12i
    i2_v1 = -p21 - p22 @ p12i @ (-p11)
    i2_v2 = -p22 @ p12i
    ym = np.block([[i1_v1, i1_v2], [i2_v1, i2_v2]])
    eye = np.eye(4)
    return (eye - z_ref * ym) @ np.linalg.inv(eye + z_ref * ym)


# ---------------------------------------------------------------------------------------------------------------
# terminal S from a modal S (the transform of gsim.palace.terminal.modal_to_terminal_s)
# ---------------------------------------------------------------------------------------------------------------
def to_terminal_s(s_modal, t_v, t_i, z_ref: float = 50.0) -> np.ndarray:
    """``V = T_V (a + b)``, ``I = T_I (a - b)`` on both faces (the same 2 x 2 blocks), power waves in the real ``z_ref``.  ``t_i`` may be modified by the caller (e.g. ``T_I diag(rho)``)."""
    s_modal = np.asarray(s_modal, complex)
    n = s_modal.shape[0]
    t = np.kron(np.eye(2), np.asarray(t_v, complex))
    w = np.kron(np.eye(2), np.asarray(t_i, complex))
    f = 1.0 / (2.0 * np.sqrt(z_ref))
    a_mat, b_mat = f * (t + z_ref * w), f * (t - z_ref * w)
    num, den = b_mat + a_mat @ s_modal, a_mat + b_mat @ s_modal
    return np.linalg.solve(den.T, num.T).T if n else num


# ---------------------------------------------------------------------------------------------------------------
# Palace's reading and its inverse
# ---------------------------------------------------------------------------------------------------------------
def block_p(p) -> np.ndarray:
    return np.kron(np.eye(2), np.asarray(p, complex))


def palace_reading(p, b) -> np.ndarray:
    """``S' = blockdiag(P, P) (I + B) - I``."""
    b = np.asarray(b, complex)
    return block_p(p) @ (np.eye(b.shape[0]) + b) - np.eye(b.shape[0])


def undo_projection(p, s_prime) -> np.ndarray:
    """``B = blockdiag(P, P)^-1 (S' + I) - I``."""
    s_prime = np.asarray(s_prime, complex)
    n = s_prime.shape[0]
    return np.linalg.solve(block_p(p), s_prime + np.eye(n)) - np.eye(n)


# ---------------------------------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------------------------------
def gates(s) -> dict:
    """Gauge-invariant numbers of a modal S: ``sigma_max`` and ``max | |S_ij| - |S_ji| |``; and the plain ``max |S - S^T|`` (terminal S: reciprocity)."""
    s = np.asarray(s, complex)
    a = np.abs(s)
    return {"sigma_max": float(np.linalg.svd(s, compute_uv=False).max()), "recip_mag": float(np.abs(a - a.T).max()), "recip": float(np.abs(s - s.T).max())}


def conj_gram(p) -> float:
    """Largest off-diagonal of the (unit-diagonal-modulus) conjugated cross-power matrix, ``|P_mk| / sqrt(|P_mm P_kk|)``."""
    p = np.asarray(p, complex)
    d = np.sqrt(np.abs(np.diag(p)))
    g = np.abs(p) / np.outer(d, d)
    return float(g[~np.eye(p.shape[0], dtype=bool)].max())


def reaction_gram(t_v, t_i) -> float:
    """Largest off-diagonal of the unconjugated reaction ``R = T_V^T T_I`` (normalised by the diagonal moduli)."""
    r = np.asarray(t_v).T @ np.asarray(t_i)
    d = np.sqrt(np.abs(np.diag(r)))
    g = np.abs(r) / np.outer(d, d)
    return float(g[~np.eye(r.shape[0], dtype=bool)].max())


def full_chain(line: dict, length: float, z_ref: float = 50.0, ti_scale: str = "rho") -> dict:
    """Truth -> Palace reading -> undo P -> terminal S with the chosen ``T_I`` renormalisation (``"none"``, ``"rho"`` = diag(Re kn / kn) = diag(1/q), ``"q"`` = diag(kn / Re kn)).  Also returns the pieces."""
    modes = line_modes(line)
    pm = palace_port_modes(modes)
    b = pseudo_modal_s(modes["gamma"], length, pm["rho"])
    sp = palace_reading(pm["P"], b)
    b_back = undo_projection(pm["P"], sp)
    scale = {"none": np.ones(2), "rho": pm["rho"], "q": pm["q"]}[ti_scale]
    s_t = to_terminal_s(b_back, pm["T_V"], pm["T_I"] * scale[None, :], z_ref)
    truth = direct_terminal_s(line, length, z_ref)
    return {"modes": modes, "port": pm, "B": b, "S_prime": sp, "B_back": b_back, "S_term": s_t, "S_true": truth, "err": float(np.abs(s_t - truth).max())}
