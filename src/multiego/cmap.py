"""
Backbone CMAP corrections (phi/psi) for multi-eGO.

Purpose
-------
In multi-eGO the local backbone conformational preference is produced by the
GROMOS-style proper dihedrals and by the short-range 1-4/1-5 pairs.  Measured
on (AAQAA)3 this gives a Ramachandran distribution that is far too rich in the
right-handed alpha basin (~37% alpha, ~15% PPII at 300 K) compared with the
experimentally derived distributions of blocked amino acids (Ace-Xaa-Nme).
Too much intrinsic alpha propensity makes helix *nucleation* too easy, which
is what destroys the cooperativity of the helix-coil transition (a
Lifson-Roig analysis of our trajectories gives v ~ 0.6 instead of ~0.05).

This module adds a residue-specific correction on the (phi, psi) map -- a
CMAP, as in the CHARMM force fields -- fitted so that a blocked amino acid
reproduces a target distribution.  The CMAP is a *correction on top of* the
existing dihedrals: nothing is removed from the force field.

Target distributions
--------------------
The targets are the experimentally derived and computationally optimized
coil-library landscapes of Choi & Pappu, J. Chem. Theory Comput. 15, 1355
(2019), doi:10.1021/acs.jctc.8b00572.  They start from the Protein Coil
Library, extract conformational basins, and reweight the basins to reproduce
IR/Raman populations of the three major basins (PPII, beta, alphaR) measured
on blocked amino acids, validating against NMR 3J(HN,Halpha).

File format: one CSV per amino acid, a 144 x 144 grid of (unnormalised)
populations on 2.5 degree bins, rows = psi, columns = phi, both running from
-180 deg; the bin centres are therefore -178.75, -176.25, ...

Fitting: iterative Boltzmann inversion
--------------------------------------
A single Boltzmann inversion (``U = -kT ln P_target + kT ln P_background``)
would be exact only if the CMAP were the only (phi, psi)-dependent term of
the force field.  It is not: the dihedrals, the 1-4/1-5 pairs and the
non-bonded interactions of the blocked amino acid all change when the CMAP is
switched on.  We therefore iterate:

    U_{n+1}(phi, psi) = U_n(phi, psi) + kT ln( P_n(phi, psi) / P_target(phi, psi) )

starting from ``U_0 = 0``, where ``P_n`` is the distribution sampled with
``U_n``.  Each update is damped and clamped (see :func:`update`), and both
distributions are smoothed on the periodic grid, so that poorly sampled cells
cannot produce huge corrections.

Caveats
-------
* The target describes a *blocked amino acid*.  Using it as a per-residue
  CMAP in a chain assumes transferability; nearest-neighbour effects, which
  are known to be significant, are not represented.
* A CMAP is an energy, so the fitted free-energy landscape is reproduced at
  the temperature of the fit and is implicitly treated as purely enthalpic.
  Populations will be right at that temperature, their temperature dependence
  only approximately.
"""

import numpy as np

# Boltzmann constant in multi-eGO/OpenMM units (kJ/mol/K).
KB = 0.008314462618

# Resolution of the Choi & Pappu target grids (degrees).
TARGET_BIN = 2.5


def load_target(path, size):
    """
    Read a Choi & Pappu landscape and return it as a normalised ``size x size``
    probability map indexed as ``P[phi, psi]``.

    The 144 x 144 grid is block-summed down to ``size x size`` (``size`` must
    divide 144), which also averages away most of the sampling noise of the
    original coil-library histogram.
    """
    raw = np.loadtxt(path, delimiter=",")
    if raw.shape != (144, 144):
        raise ValueError(f"{path}: expected a 144x144 grid, got {raw.shape}")
    if 144 % size:
        raise ValueError(f"grid size {size} does not divide 144")
    raw = raw.T  # file is [psi, phi]; we work with [phi, psi]
    block = 144 // size
    # Group the fine bins into blocks *centred* on the coarse grid nodes
    # (grid_angles), not starting at them: hence the half-block roll.
    rolled = np.roll(raw, block // 2, axis=(0, 1))
    coarse = rolled.reshape(size, block, size, block).sum(axis=(1, 3))
    return coarse / coarse.sum()


def grid_angles(size):
    """
    Angles (degrees) of the grid nodes of a ``size x size`` periodic map,
    ``-180 + i * 360 / size``.  These are *nodes*, not bin centres: the energy
    of a map is defined at these angles and interpolated between them, so a
    histogram meant to be compared with a map must use bins *centred* on them
    (see :func:`histogram`).
    """
    return -180.0 + (360.0 / size) * np.arange(size)


def to_openmm_map(grid):
    """
    Convert a map indexed ``[phi, psi]`` with angles :func:`grid_angles` into
    the flat array expected by ``openmm.CMAPTorsionForce.addMap``.

    Two conventions differ and are both handled here (verified numerically
    against analytic maps, see tests/test_cmap.py):

    * the first index of the OpenMM map runs over the *second* torsion of
      ``addTorsion`` (psi for us), so the two axes are transposed;
    * the OpenMM angle origin is 0 degrees rather than -180, so both axes are
      rolled by half a period.
    """
    size = grid.shape[0]
    rolled = np.roll(np.asarray(grid, dtype=float), size // 2, axis=(0, 1))
    return rolled.ravel(order="F")


def histogram(phi, psi, size, weights=None):
    """
    Normalised 2D histogram of sampled angles (degrees), on the same grid
    convention as :func:`load_target`.
    """
    width = 360.0 / size

    def node_of(angles):
        """Index of the nearest grid node, with periodic wrapping."""
        shifted = (np.asarray(angles, dtype=float).ravel() + 180.0 + width / 2.0) % 360.0
        return np.floor(shifted / width).astype(int) % size

    counts = np.zeros((size, size))
    w = 1.0 if weights is None else np.asarray(weights, dtype=float).ravel()
    np.add.at(counts, (node_of(phi), node_of(psi)), w)
    return counts / counts.sum()


def smooth(grid, sigma_bins):
    """
    Periodic Gaussian smoothing of a map, done by multiplication in Fourier
    space (so the periodicity in phi and psi is exact).
    """
    if sigma_bins <= 0:
        return grid
    n = grid.shape[0]
    k = np.fft.fftfreq(n) * n
    kernel = np.exp(-0.5 * (2 * np.pi * sigma_bins / n) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(grid) * kernel))


def update(
    current,
    sampled,
    target,
    temperature,
    n_samples,
    damping=1.0,
    max_step=5.0,
    max_total=30.0,
    sigma_bins=1.0,
    pseudo_count=1.0,
):
    """
    One iterative-Boltzmann-inversion update of a CMAP.

    Parameters
    ----------
    current : np.ndarray
        Current CMAP (kJ/mol), ``size x size``, indexed ``[phi, psi]``.
    sampled, target : np.ndarray
        Sampled and target probability maps on the same grid.
    temperature : float
        Temperature of the sampling run (K).
    damping : float
        Fraction of the computed correction that is applied.  Values below 1
        slow convergence but make it more stable.
    max_step : float
        Largest change of a single cell in one iteration (kJ/mol).
    max_total : float
        Largest absolute value of the CMAP (kJ/mol).  Caps the correction in
        cells that the target leaves essentially empty, where the logarithm
        would otherwise diverge.
    sigma_bins : float
        Width (in bins) of the Gaussian smoothing applied to both maps before
        taking their ratio, and to the correction itself.
    n_samples : int
        Number of frames behind ``sampled``.  It sets the statistical
        resolution of the comparison (see ``pseudo_count``).
    pseudo_count : float
        Frames added to every cell of *both* maps before taking their ratio
        (the target being scaled to ``n_samples`` frames).  This is essential,
        not cosmetic.  A run of N frames cannot tell a population of 0 from
        one of ~1/N, so a cell where the target expects less than one frame
        and the simulation has none carries no information.  Without the
        pseudo-count the logarithm of the ratio there is large and negative,
        the cell is lowered at every iteration, and after a few iterations
        the map has dug a well deeper than the steric barrier that kept the
        region empty: the molecule then falls into it in bulk and the fit
        oscillates or diverges.  (Observed on threonine: a collapse into
        phi = +60 every third iteration.)

    Returns
    -------
    new_cmap : np.ndarray
    rms_change : float
        RMS of the applied correction (kJ/mol), a convergence indicator.
    """
    # Fourier smoothing of a sharply peaked map can produce slightly negative
    # cells: clip them.  The pseudo-count is expressed as a probability.
    floor = pseudo_count / float(n_samples)
    p_sim = np.maximum(smooth(sampled, sigma_bins), 0.0) + floor
    p_tgt = np.maximum(smooth(target, sigma_bins), 0.0) + floor
    p_sim /= p_sim.sum()
    p_tgt /= p_tgt.sum()

    correction = damping * KB * temperature * np.log(p_sim / p_tgt)
    correction = smooth(correction, sigma_bins)
    correction -= correction.mean()  # a constant shift is physically irrelevant
    correction = np.clip(correction, -max_step, max_step)

    new = np.clip(current + correction, -max_total, max_total)
    new -= new.min()  # keep the map non-negative: easier to read, same physics
    return new, float(np.sqrt((correction**2).mean()))


def basin_populations(grid):
    """
    Populations of the alphaR, PPII and beta basins of a (phi, psi) map, with
    the simple rectangular partition used throughout this work:

    ==========  =======================================================
    basin       definition
    ==========  =======================================================
    alphaR      -160 < phi < 0,      -120 < psi < 30
    PPII        -100 < phi < 0,      psi > 30 or psi < -120
    beta        phi < -100,          psi > 30 or psi < -120
    alphaL      phi > 0,             -120 < psi < 30
    ==========  =======================================================

    Everything else is reported as ``other``.  These boundaries reproduce the
    three-basin populations of the Choi & Pappu landscapes to better than a
    percent, and are applied identically to targets and to simulations.
    """
    size = grid.shape[0]
    angles = grid_angles(size)
    phi, psi = np.meshgrid(angles, angles, indexing="ij")
    extended = (psi > 30) | (psi < -120)
    masks = {
        "alphaR": (~extended) & (phi < 0) & (phi > -160),
        "PPII": extended & (phi > -100) & (phi < 0),
        "beta": extended & (phi <= -100),
        "alphaL": (~extended) & (phi > 0),
    }
    total = grid.sum()
    populations = {name: float(grid[mask].sum() / total) for name, mask in masks.items()}
    populations["other"] = float(1.0 - sum(populations.values()))
    return populations


def j_hnha(phi_degrees):
    """
    3J(HN,Halpha) coupling constant (Hz) from the Karplus relation with the
    parameters used by Choi & Pappu (Hu & Bax):
    ``7.09 cos^2(phi - 60) - 1.42 cos(phi - 60) + 1.55``.
    Averaging this over an ensemble gives the quantity compared with NMR.
    """
    theta = np.deg2rad(np.asarray(phi_degrees) - 60.0)
    return 7.09 * np.cos(theta) ** 2 - 1.42 * np.cos(theta) + 1.55


def mean_j(grid):
    """Ensemble-averaged 3J(HN,Halpha) of a (phi, psi) probability map."""
    angles = grid_angles(grid.shape[0])
    return float((grid.sum(axis=1) * j_hnha(angles)).sum() / grid.sum())


def flatten_unpopulated(grid, target, threshold=1e-4):
    """
    Raise the cells that the target leaves essentially empty to the maximum of
    the map.

    Iterative Boltzmann inversion never touches a cell that is sampled by
    neither the simulation nor the target: the ratio of the two floors is one,
    so the correction is zero there.  After the rest of the map has been
    pushed up, those untouched cells end up as the *lowest* points of the
    CMAP -- artificial wells in regions that are sterically forbidden.  They
    are harmless as long as the underlying force field keeps those regions
    inaccessible, but they are wrong, and they would become dangerous in any
    context where the steric barrier is softer.

    ``threshold`` is expressed as a fraction of a uniform distribution: cells
    whose (smoothed) target population is below ``threshold / size^2`` are
    flattened.
    """
    size = grid.shape[0]
    smoothed = np.maximum(smooth(target, 1.0), 0.0)
    empty = smoothed < threshold / size**2
    flattened = np.array(grid, dtype=float)
    flattened[empty] = flattened.max()
    return flattened


def flattening_bias(target, temperature, cap=12.0, sigma_bins=1.0, smooth_bias_bins=1.5):
    """
    Bias map that flattens the target landscape, for sampling rare regions.

    ``U_bias = kT ln(P_target / max P_target)``, clipped at ``-cap``: zero on
    the most populated cell, negative (favourable) where the target is rare.
    A simulation whose model already reproduces the target then samples all
    regions of the target with comparable probability (down to
    exp(-cap/kT) of the maximum), instead of visiting a basin of 0.3%
    population a few dozen times.  The unbiased distribution is recovered by
    weighting each frame with exp(+U_bias/kT) (see :func:`bias_weights`).

    Why it is needed: rare basins such as alphaL for non-glycine residues
    (target ~0.1-0.3%) are below the sampling resolution of unbiased runs,
    so their correction is never fitted.  In a helix the resulting error
    (3-6 kJ/mol per residue toward alphaL) is multiplied by the number of
    residues, and since the hydrogen-bond term is achiral it can make a
    left-handed helix of L residues as favourable as a right-handed one.
    """
    p = np.maximum(smooth(target, sigma_bins), 0.0)
    p = p / p.max()
    bias = np.clip(KB * temperature * np.log(np.maximum(p, 1e-30)), -cap, 0.0)
    # Smooth the clipped map as well: the clip creates edges where the bias
    # changes by the full cap over one grid cell, and such steep torques make
    # the integration unstable for deep caps (NaN at cap = 25 kJ/mol with a
    # 5 fs time step).  Re-zero the maximum afterwards.
    if smooth_bias_bins > 0:
        bias = smooth(bias, smooth_bias_bins)
        bias -= bias.max()
    return bias


def bias_weights(phi, psi, bias, temperature):
    """
    Reweighting factors exp(+U_bias/kT) of sampled frames, with U_bias
    interpolated bilinearly on the periodic node grid of :func:`grid_angles`.
    """
    size = bias.shape[0]
    width = 360.0 / size

    def coordinates(angles):
        position = ((np.asarray(angles, dtype=float).ravel() + 180.0) % 360.0) / width
        lower = np.floor(position).astype(int) % size
        return lower, (lower + 1) % size, position - np.floor(position)

    i0, i1, fi = coordinates(phi)
    j0, j1, fj = coordinates(psi)
    value = (
        bias[i0, j0] * (1 - fi) * (1 - fj)
        + bias[i1, j0] * fi * (1 - fj)
        + bias[i0, j1] * (1 - fi) * fj
        + bias[i1, j1] * fi * fj
    )
    return np.exp(value / (KB * temperature))
