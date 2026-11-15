"""
Tests of the backbone CMAP machinery (multiego/cmap.py + the CMAPTorsionForce
wiring of multiego.omm.builder).

The important one is test_openmm_map_matches_analytic_function: it pins down
the two index conventions of openmm.CMAPTorsionForce (which axis is phi, and
where the angle origin is) by comparing against an analytic, asymmetric
function of phi and psi.  A symmetric test function would not detect a swap
of the two axes.
"""

import os
import sys

import numpy as np
import openmm as mm
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from multiego import cmap  # noqa: E402


def build_two_torsion_system(grid):
    """A 5-particle chain whose only energy is one CMAP on its two torsions."""
    system = mm.System()
    for _ in range(5):
        system.addParticle(12.0)
    force = mm.CMAPTorsionForce()
    force.addMap(grid.shape[0], cmap.to_openmm_map(grid))
    force.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)  # phi = 0-1-2-3, psi = 1-2-3-4
    system.addForce(force)
    return system


def torsion(x, a, b, c, d):
    b0, b1, b2 = x[a] - x[b], x[c] - x[b], x[d] - x[c]
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return np.arctan2(np.cross(b1, v) @ w, v @ w)


@pytest.mark.parametrize("size", [36, 72])
def test_openmm_map_matches_analytic_function(size):
    """The map is evaluated as f(phi, psi) with our [phi, psi] indexing."""

    def f(phi, psi):  # asymmetric in phi <-> psi on purpose
        return 5.0 * np.cos(phi) + 3.0 * np.sin(2 * psi) + 2.0 * np.cos(phi - psi)

    angles = np.deg2rad(cmap.grid_angles(size))
    grid = f(angles[:, None], angles[None, :])
    context = mm.Context(
        build_two_torsion_system(grid), mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.normal(0.0, 0.15, (5, 3)) + np.array(
            [[0, 0, 0], [0.15, 0, 0], [0.3, 0.1, 0], [0.45, 0, 0], [0.6, 0.1, 0.1]]
        )
        context.setPositions(x)
        energy = context.getState(getEnergy=True).getPotentialEnergy()._value
        assert energy == pytest.approx(f(torsion(x, 0, 1, 2, 3), torsion(x, 1, 2, 3, 4)), abs=5e-3)


def test_target_loading_preserves_populations(tmp_path):
    """Down-sampling the 144x144 target keeps the basin populations."""
    fine = np.zeros((144, 144))
    # a blob in the PPII basin (phi -66, psi 146); file layout is [psi, phi]
    i_phi, i_psi = int((-66 + 180) / 2.5), int((146 + 180) / 2.5)
    fine[i_psi - 2 : i_psi + 3, i_phi - 2 : i_phi + 3] = 1.0  # noqa: E203
    path = tmp_path / "X.csv"
    np.savetxt(path, fine, delimiter=",")
    for size in (36, 72):
        grid = cmap.load_target(path, size)
        assert grid.sum() == pytest.approx(1.0)
        assert cmap.basin_populations(grid)["PPII"] == pytest.approx(1.0)


def test_histogram_bins_are_centred_on_nodes():
    """Angles exactly on a node fall in that node's bin."""
    size = 36
    angles = cmap.grid_angles(size)
    grid = cmap.histogram(angles, np.full(size, angles[3]), size)
    assert np.allclose(grid[:, 3], 1.0 / size)  # one count on each phi node
    assert grid[:, 3].sum() == pytest.approx(1.0)  # all in the psi bin of node 3
    # within half a bin of a node the angle belongs to that node, beyond it
    # to the next one
    width = 360.0 / size
    assert np.argmax(cmap.histogram([angles[5] + 0.49 * width], [angles[0]], size)) // size == 5
    assert np.argmax(cmap.histogram([angles[5] + 0.51 * width], [angles[0]], size)) // size == 6


def test_smoothing_is_periodic_and_conserves_mass():
    rng = np.random.default_rng(1)
    grid = rng.random((36, 36))
    smoothed = cmap.smooth(grid, 1.5)
    assert smoothed.sum() == pytest.approx(grid.sum())
    # shifting the input shifts the output (no edge effects)
    shifted = cmap.smooth(np.roll(grid, 7, axis=0), 1.5)
    assert np.allclose(shifted, np.roll(smoothed, 7, axis=0))


def test_update_moves_towards_target():
    """One IBI step reduces the discrepancy between sampled and target maps."""
    size = 36
    angles = np.deg2rad(cmap.grid_angles(size))
    target = np.exp(np.cos(angles)[:, None] + 0.0 * angles[None, :])
    target /= target.sum()
    sampled = np.exp(-np.cos(angles)[:, None] + 0.0 * angles[None, :])
    sampled /= sampled.sum()
    new, rms = cmap.update(np.zeros((size, size)), sampled, target, 300.0, n_samples=100000)
    assert rms > 0.0
    # the correction must be lower where the target wants more population
    assert new[np.argmax(target.sum(axis=1))].mean() < new[np.argmin(target.sum(axis=1))].mean()


def test_basin_populations_and_j_coupling_on_real_target():
    """Sanity values for the Choi & Pappu alanine landscape, if present."""
    path = "/mnt/user-data/uploads/ALA.csv"
    if not os.path.exists(path):
        pytest.skip("target landscape not available")
    grid = cmap.load_target(path, 72)
    populations = cmap.basin_populations(grid)
    assert populations["PPII"] > populations["beta"] > populations["alphaR"]
    assert 5.5 < cmap.mean_j(grid) < 6.5  # experimental Ala dipeptide ~6 Hz


def test_single_molecule_system_is_equivalent():
    """The fast single-molecule System used by tools/cmap_fit matches the builder."""
    params_path = "/tmp/dipep/out/ALA/base_1/mego_params.h5"
    conf = "/tmp/dipep/ALA_box.gro"
    if not (os.path.exists(params_path) and os.path.exists(conf)):
        pytest.skip("dipeptide model not available")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tools", "cmap_fit"))
    import openmm.app as app

    from multiego.omm import builder, energy, paramfile
    from single_molecule import create_single_molecule_system

    params = paramfile.read(params_path)
    gro = app.GromacsGroFile(conf)
    box = gro.getPeriodicBoxVectors()
    standard = builder.create_system(params, box_vectors=box)
    fast = create_single_molecule_system(params, box)
    rng = np.random.default_rng(3)
    x0 = np.array(gro.getPositions()._value)
    for _ in range(4):
        x = x0 + rng.normal(0.0, 0.05, x0.shape)
        assert not energy.compare(energy.decompose(standard, x), energy.decompose(fast, x))


def test_update_ignores_cells_below_the_sampling_resolution():
    """
    A cell where the target expects less than one frame and the simulation
    has none must not be lowered: doing so digs artificial wells that
    eventually capture the molecule (the failure seen on threonine).
    """
    size, frames = 36, 10000
    target = np.zeros((size, size))
    target[10, 30] = 0.99999  # the real basin
    target[25, 20] = 0.00001  # 0.1 expected frames: below the resolution
    sampled = np.zeros((size, size))
    sampled[10, 30] = 1.0
    new, _ = cmap.update(np.zeros((size, size)), sampled, target, 300.0, n_samples=frames, sigma_bins=0.0)
    assert abs(new[25, 20] - new[0, 0]) < 0.3  # same as a cell empty in both maps
    # ... while a cell the target clearly populates (100 expected frames) is lowered
    target[25, 20], target[10, 30] = 0.01, 0.99
    new, _ = cmap.update(np.zeros((size, size)), sampled, target, 300.0, n_samples=frames, sigma_bins=0.0)
    assert new[25, 20] < new[0, 0] - 4.0


def test_only_used_maps_are_stored():
    """
    The parameter file keeps only the CMAP maps of residue types present in
    the system: OpenMM's CPU/Reference platforms pay for every map at every
    step (all 20 maps made the CMAP of a 15-residue peptide 30x more expensive
    than the rest of the force field).
    """
    from multiego.omm import paramfile

    atoms = {
        "resname": np.array(["ACE", "ALA", "GLN", "NME"]),
        "resnum": np.array([1, 2, 3, 4]),
        "name": np.array(["C", "CA", "CA", "N"]),
    }
    molecules = {"X": paramfile.MoleculeParameters("X", atoms, {})}
    maps = {name: np.zeros((36, 36)) for name in ("ALA", "GLN", "GLY", "PRO", "TRP")}
    result = paramfile.cmap_from_maps(molecules, maps)
    assert sorted(result.maps) == ["ALA", "GLN"]


def test_flattening_bias_and_reweighting_recover_the_distribution():
    """
    Frames sampled from a biased distribution P*exp(-U_bias/kT) and weighted
    with bias_weights give back P, including its rare regions.
    """
    size, temperature = 36, 300.0
    angles = cmap.grid_angles(size)
    phi_grid, psi_grid = np.meshgrid(angles, angles, indexing="ij")
    target = np.exp(-((phi_grid + 70) ** 2 + (psi_grid - 140) ** 2) / 800.0) + 0.003 * np.exp(
        -((phi_grid - 60) ** 2 + (psi_grid - 40) ** 2) / 300.0
    )  # a main basin and a rare "alphaL" one
    target /= target.sum()
    bias = cmap.flattening_bias(target, temperature, cap=12.0, sigma_bins=0.0, smooth_bias_bins=0.0)
    assert bias.max() == pytest.approx(0.0) and bias.min() >= -12.0
    biased = target * np.exp(-bias / (cmap.KB * temperature))
    biased /= biased.sum()
    rng = np.random.default_rng(5)
    cells = rng.choice(size * size, size=200000, p=biased.ravel())
    i, j = np.unravel_index(cells, (size, size))
    phi, psi = angles[i], angles[j]
    recovered = cmap.histogram(phi, psi, size, cmap.bias_weights(phi, psi, bias, temperature))
    rare = (np.abs(phi_grid - 60) < 25) & (np.abs(psi_grid - 40) < 25)
    assert recovered[rare].sum() == pytest.approx(target[rare].sum(), rel=0.1)
    # and the rare region is sampled far more often than without the bias
    assert biased[rare].sum() > 20 * target[rare].sum()


def test_flattening_bias_has_no_steep_edges():
    """The default bias is smoothed: neighbouring cells differ by far less than the cap."""
    size = 36
    angles = np.deg2rad(cmap.grid_angles(size))
    target = np.exp(5 * np.cos(angles)[:, None] * np.cos(angles)[None, :])
    target /= target.sum()
    bias = cmap.flattening_bias(target, 300.0, cap=25.0)
    steps = max(np.abs(np.diff(bias, axis=0)).max(), np.abs(np.diff(bias, axis=1)).max())
    assert steps < 0.4 * 25.0
