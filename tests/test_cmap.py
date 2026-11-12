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
    new, rms = cmap.update(np.zeros((size, size)), sampled, target, 300.0)
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
