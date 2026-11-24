"""
Numerical core of the CMAP builder.

- Histograms on grids centred on the map nodes.
- Evaluation of a map exactly as OpenMM's CMAPTorsionForce does: derivatives
  at the nodes from periodic cubic splines, then bicubic patches. Verified
  against OpenMM in tests (``test_cmap_builder.py``).
- Geometry helper placing five backbone atoms at given (phi, psi).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

KB = 0.008314462618  # kJ/mol/K


def nodes(n):
    return -180.0 + np.arange(n) * 360.0 / n


def node_histogram(phi, psi, n, weights=None):
    """
    2D histogram with bins centred on the map nodes, rows = psi, cols = phi.

    Bin k covers [node_k - d/2, node_k + d/2), d = 360/n, periodic.
    """
    d = 360.0 / n
    i = np.floor(((np.asarray(psi) + 180.0 + d / 2) % 360.0) / d).astype(int) % n
    j = np.floor(((np.asarray(phi) + 180.0 + d / 2) % 360.0) / d).astype(int) % n
    h = np.zeros((n, n))
    np.add.at(h, (i, j), 1.0 if weights is None else weights)
    return h


def coarsen_target(grid144, n):
    """
    Sum a 144x144 Choi-Pappu grid (2.5 deg bins from -180, rows = psi) into
    node-centred bins of an n x n map. Requires 144/n even.
    """
    f = grid144.shape[0] // n
    if grid144.shape != (144, 144) or 144 % n or f % 2:
        raise ValueError(f"cannot coarsen a {grid144.shape} target to {n} node-centred bins")
    g = np.roll(grid144, f // 2, axis=(0, 1))  # first bin now starts at -180 - d/2
    return g.reshape(n, f, n, f).sum(axis=(1, 3))


def smooth_periodic(u, sigma_cells):
    """Periodic Gaussian smoothing (FFT); sigma in grid cells."""
    if sigma_cells <= 0:
        return u.copy()
    n = u.shape[0]
    k = np.fft.fftfreq(n)
    g = np.exp(-2.0 * (np.pi * sigma_cells) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(u) * g))


# --------------------------------------------------------------------- OpenMM
def _periodic_derivative(values, axis):
    """d/dangle (per radian) at the nodes of a periodic cubic spline along axis."""
    n = values.shape[axis]
    x = np.arange(n + 1) * 2 * np.pi / n
    v = np.moveaxis(values, axis, 0)
    v = np.concatenate([v, v[:1]], axis=0)
    d = CubicSpline(x, v, axis=0, bc_type="periodic")(x[:-1], 1)
    return np.moveaxis(d, 0, axis)


# Inverse matrix of the bicubic Hermite problem (Numerical Recipes bcucof).
_WT = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
        [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
        [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
        [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
        [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
        [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
        [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
        [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
        [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1],
    ],
    dtype=float,
)


class BicubicMap:
    """
    A map evaluated like OpenMM's CMAPTorsionForce.

    grid : (n, n) rows = psi, cols = phi, nodes from -180 (our file layout).
    """

    def __init__(self, grid):
        g = np.asarray(grid, float)
        n = g.shape[0]
        self.n = n
        self.delta = 2 * np.pi / n
        # work in (phi index, psi index) order, as OpenMM: e[i_phi, j_psi]
        e = g.T
        d1 = _periodic_derivative(e, 0)  # d/dphi
        d2 = _periodic_derivative(e, 1)  # d/dpsi
        d12 = _periodic_derivative(d1, 1)  # d2/dphi dpsi
        coeff = np.empty((n, n, 4, 4))
        dl = self.delta
        for i in range(n):
            ip = (i + 1) % n
            for j in range(n):
                jp = (j + 1) % n
                # corners counter-clockwise: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
                y = [e[i, j], e[ip, j], e[ip, jp], e[i, jp]]
                y1 = [d1[i, j], d1[ip, j], d1[ip, jp], d1[i, jp]]
                y2 = [d2[i, j], d2[ip, j], d2[ip, jp], d2[i, jp]]
                y12 = [d12[i, j], d12[ip, j], d12[ip, jp], d12[i, jp]]
                x = np.concatenate([y, np.multiply(y1, dl), np.multiply(y2, dl), np.multiply(y12, dl * dl)])
                coeff[i, j] = (_WT @ x).reshape(4, 4)
        self.coeff = coeff

    def energy(self, phi_deg, psi_deg):
        """Energy (kJ/mol) at angles in degrees (any range)."""
        a = (np.radians(np.asarray(phi_deg, float)) + np.pi) % (2 * np.pi)
        b = (np.radians(np.asarray(psi_deg, float)) + np.pi) % (2 * np.pi)
        # nodes start at -180 deg in our layout
        fa, fb = a / self.delta, b / self.delta
        i = np.minimum(np.floor(fa).astype(int), self.n - 1)
        j = np.minimum(np.floor(fb).astype(int), self.n - 1)
        t, u = fa - i, fb - j
        c = self.coeff[i, j]  # (..., 4, 4)
        tp = np.stack([np.ones_like(t), t, t * t, t * t * t], -1)
        up = np.stack([np.ones_like(u), u, u * u, u * u * u], -1)
        return np.einsum("...k,...kl,...l->...", tp, c, up)


def backbone_positions(phi_deg, psi_deg):
    """
    Positions (nm) of C(-1), N, CA, C, N(+1) with dihedrals phi and psi.

    Ideal bond lengths and angles; only the two dihedrals matter here.
    """

    def place(a, b, c, bond, angle, torsion):
        bc = c - b
        bc /= np.linalg.norm(bc)
        nrm = np.cross(b - a, bc)
        nrm /= np.linalg.norm(nrm)
        m = np.array([bc, np.cross(nrm, bc), nrm]).T
        ang, tor = np.radians(angle), np.radians(torsion)
        d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
        return c + m @ d2

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.133, 0.0, 0.0])
    p2 = p1 + 0.146 * np.array([-np.cos(np.radians(121.7)), np.sin(np.radians(121.7)), 0.0])
    p3 = place(p0, p1, p2, 0.152, 111.0, phi_deg)
    p4 = place(p1, p2, p3, 0.133, 116.2, psi_deg)
    return np.array([p0, p1, p2, p3, p4])


def dihedral(p0, p1, p2, p3):
    """IUPAC dihedral in degrees."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


# ------------------------------------------------------- curvature-limited fit
"""
Curvature-constrained CMAP fit.

OpenMM's bicubic CMAP (derivatives from periodic cubic splines) is exactly the
tensor-product periodic cubic spline through the grid nodes, so the map on any
fine grid is ``S @ U @ S.T`` and its second derivatives are ``S @ U @ S2.T``
and ``S2 @ U @ S.T``, all linear in the node values ``U``.

The fit minimises

    -sum_C p_target(C) ln P_rw(C)  +  lam * sum (max(|d2U| - kappa, 0))^2

where P_rw is the sampled distribution reweighted with the map (frames binned
on a fine grid) and C are bins aligned with the target grid. Only curvature
above ``kappa`` is penalised; nothing else is smoothed, so basin populations
are fitted to the original target.
"""

RAD2 = (180.0 / np.pi) ** 2  # per degree^2 -> per rad^2


def spline_basis(n, points_deg, deriv=0):
    """Cardinal periodic cubic spline basis on n nodes from -180, evaluated at points (degrees)."""
    x = np.arange(n + 1) * 360.0 / n - 180.0
    out = np.empty((len(points_deg), n))
    for i in range(n):
        y = np.zeros(n + 1)
        y[i] = 1.0
        y[-1] = y[0]
        out[:, i] = CubicSpline(x, y, bc_type="periodic")(points_deg, deriv)
    return out


class ConstrainedFit:
    def __init__(self, phi, psi, target144, n=72, temperature=300.0, fine_deg=0.25, bin_deg=5.0, weights0=None):
        self.n, self.kt = n, KB * temperature
        m = int(round(360.0 / fine_deg))
        self.m = m
        centres = -180.0 + fine_deg * (np.arange(m) + 0.5)
        self.S = spline_basis(n, centres)
        self.S2 = spline_basis(n, centres, 2) * RAD2
        self.S1 = spline_basis(n, centres, 1) * np.sqrt(RAD2)
        # sampled frames on the fine grid (rows psi), optional prior weights (sampled map)
        i = np.floor((np.asarray(psi) + 180.0) / fine_deg).astype(int) % m
        j = np.floor((np.asarray(phi) + 180.0) / fine_deg).astype(int) % m
        cnt = np.zeros((m, m))
        np.add.at(cnt, (i, j), 1.0 if weights0 is None else weights0)
        self.counts = cnt
        # target aggregated to aligned bins of bin_deg
        f = int(round(bin_deg / 2.5))
        self.nb = 144 // f
        self.pt = target144.reshape(self.nb, f, self.nb, f).sum(axis=(1, 3))
        self.block = m // self.nb  # fine cells per bin along an axis
        self.eps = 1e-12

    def _bins(self, a):
        return a.reshape(self.nb, self.block, self.nb, self.block).sum(axis=(1, 3))

    def energy_fine(self, u):
        return self.S @ u @ self.S.T

    def loss(self, x, kappa, lam, mu=0.0):
        u = x.reshape(self.n, self.n)
        e = self.energy_fine(u)
        a = -(e - e.min()) / self.kt
        w = self.counts * np.exp(a)
        z = w.sum()
        pr = w / z
        prb = self._bins(pr)
        ce = -np.sum(self.pt * np.log(prb + self.eps))
        ratio = self.pt / (prb + self.eps)
        g_fine = (
            pr
            / self.kt
            * (np.kron(ratio, np.ones((self.block, self.block))) - np.sum(self.pt * prb / (prb + self.eps)))
        )
        grad = self.S.T @ g_fine @ self.S
        # curvature penalty
        k_phi = self.S @ u @ self.S2.T
        k_psi = self.S2 @ u @ self.S.T
        v_phi = np.maximum(np.abs(k_phi) - kappa, 0.0)
        v_psi = np.maximum(np.abs(k_psi) - kappa, 0.0)
        pen = lam * (np.sum(v_phi**2) + np.sum(v_psi**2))
        gp_phi = 2 * lam * v_phi * np.sign(k_phi)
        gp_psi = 2 * lam * v_psi * np.sign(k_psi)
        grad = grad + self.S.T @ gp_phi @ self.S2 + self.S2.T @ gp_psi @ self.S
        if mu > 0:
            # thin-plate smoothness: mean of U_pp^2 + U_ss^2 + 2 U_ps^2 over the fine grid
            k_x = self.S1 @ u @ self.S1.T
            norm = mu / k_phi.size
            pen += norm * (np.sum(k_phi**2) + np.sum(k_psi**2) + 2 * np.sum(k_x**2))
            grad = grad + 2 * norm * (
                self.S.T @ k_phi @ self.S2 + self.S2.T @ k_psi @ self.S + 2 * self.S1.T @ k_x @ self.S1
            )
        return ce + pen, grad.ravel()

    def fit(self, u0, kappa=500.0, lam=1e-7, maxiter=3000, mu=0.0):
        res = minimize(
            self.loss,
            u0.ravel(),
            args=(kappa, lam, mu),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-12, "gtol": 1e-9},
        )
        u = res.x.reshape(self.n, self.n)
        return u - u.min(), res
