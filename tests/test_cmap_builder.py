"""
Tests for tools/cmap_builder: histograms, OpenMM-exact map evaluation, and
synthetic round trips of the map fit (known answer).
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "cmap_builder"))
import build_cmap as B  # noqa: E402
from cmap_math import KB, BicubicMap, backbone_positions, coarsen_target, node_histogram, nodes  # noqa: E402

KT = KB * 300.0


def test_node_histogram_is_centred_and_periodic():
    n = 36  # 10 degree bins centred on -180, -170, ...
    h = node_histogram(np.array([-180.0, 179.0, -175.1, -174.9, 4.9]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), n)
    row = n // 2  # psi = 0
    assert h[row, 0] == 3  # -180, 179 (wraps), -175.1
    assert h[row, 1] == 1  # -174.9 belongs to the node at -170
    assert h[row, 18] == 1  # 4.9 belongs to the node at 0
    assert h.sum() == 5


def test_coarsen_target_matches_node_histogram():
    """A target built from points must coarsen to the same bins the frames go to."""
    rng = np.random.default_rng(0)
    phi, psi = rng.uniform(-180, 180, (2, 5000))
    fine, _, _ = np.histogram2d(psi, phi, bins=[np.linspace(-180, 180, 145)] * 2)
    for n in (24, 36, 72):
        np.testing.assert_array_equal(coarsen_target(fine, n), node_histogram(phi, psi, n))
    with pytest.raises(ValueError):
        coarsen_target(fine, 48)  # 144/48 = 3 is odd: bins cannot be centred on nodes


def test_bicubic_map_matches_openmm():
    openmm = pytest.importorskip("openmm")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    from multiego.openmm_backend.cmap import to_openmm_map

    rng = np.random.default_rng(1)
    n = 24
    grid = rng.normal(0, 3, (n, n))
    system = openmm.System()
    for _ in range(5):
        system.addParticle(12.0)
    f = openmm.CMAPTorsionForce()
    f.addMap(n, to_openmm_map(grid))
    f.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)
    system.addForce(f)
    ctx = openmm.Context(system, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName("Reference"))
    bm = BicubicMap(grid)
    for phi, psi in rng.uniform(-180, 180, (50, 2)):
        ctx.setPositions(backbone_positions(phi, psi))
        e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
        assert e == pytest.approx(float(bm.energy(phi, psi)), abs=1e-10)


# ------------------------------------------------------------------ round trip
def _p0(phi, psi):
    """Broad 'unbiased' density (degrees in)."""
    p, s = np.radians(phi), np.radians(psi)
    return np.exp(0.8 * np.cos(p + np.radians(70)) + 0.6 * np.cos(s - np.radians(120)) + 0.3 * np.cos(p - s))


def _u_true(phi, psi):
    p, s = np.radians(phi), np.radians(psi)
    return 4.0 * np.cos(p) + 3.0 * np.sin(s + np.radians(30)) + 2.0 * np.sin(p - s)


def _sample(density, nframes, rng):
    """Frames from a density given as a function of (phi, psi), via a 0.5 degree grid."""
    a = -180 + 0.25 + 0.5 * np.arange(720)
    phi, psi = np.meshgrid(a, a)
    w = density(phi, psi).ravel()
    idx = rng.choice(w.size, size=nframes, p=w / w.sum())
    jitter = rng.uniform(-0.25, 0.25, (2, nframes))
    return phi.ravel()[idx] + jitter[0], psi.ravel()[idx] + jitter[1]


def _target(density):
    c = -180 + 1.25 + 2.5 * np.arange(144)
    phi, psi = np.meshgrid(c, c)  # rows psi, cols phi, as the Choi & Pappu grids
    t = density(phi, psi)
    return t / t.sum()


def _check(phi, psi, grid, target, sampled_grid=None):
    de = BicubicMap(grid).energy(phi, psi)
    if sampled_grid is not None:
        de = de - BicubicMap(sampled_grid).energy(phi, psi)
    w = np.exp(-(de - de.min()) / KT)
    r = B.basin_populations(phi, psi, w)
    t = B.target_basins(target)
    for k in ("aR", "PII", "beta", "aL"):
        assert r[k] == pytest.approx(t[k], abs=0.005), k
    # the map equals the true correction (up to a constant) where the target has weight
    n = grid.shape[0]
    th = nodes(n)
    p, s = np.meshgrid(th, th)
    mask = coarsen_target(target, n) > 2e-4
    diff = (grid - _u_true(p, s))[mask]
    assert np.std(diff) < 0.35


def test_round_trip_unbiased():
    rng = np.random.default_rng(7)
    phi, psi = _sample(_p0, 400_000, rng)
    target = _target(lambda a, b: _p0(a, b) * np.exp(-_u_true(a, b) / KT))
    grid, info = B.fit_map(phi, psi, target, n=36, temperature=300.0, log=lambda *a: None)
    assert info["total_variation_final"] < 0.02
    _check(phi, psi, grid, target)


def test_round_trip_with_sampled_map():
    """Second iteration: frames sampled with a map active; the fit must account for it."""
    rng = np.random.default_rng(8)
    n = 36
    th = nodes(n)
    p, s = np.meshgrid(th, th)
    sampled_grid = 2.5 * np.cos(np.radians(p - 40)) * np.sin(np.radians(s))  # some earlier map
    sampled_grid -= sampled_grid.min()
    bm_s = BicubicMap(sampled_grid)
    phi, psi = _sample(lambda a, b: _p0(a, b) * np.exp(-bm_s.energy(a, b) / KT), 400_000, rng)
    target = _target(lambda a, b: _p0(a, b) * np.exp(-_u_true(a, b) / KT))
    grid, _ = B.fit_map(phi, psi, target, n=n, temperature=300.0, sampled_map=sampled_grid, log=lambda *a: None)
    _check(phi, psi, grid, target, sampled_grid)


def test_cross_validation_reports_small_errors_on_clean_data():
    rng = np.random.default_rng(9)
    phi, psi = _sample(_p0, 400_000, rng)
    target = _target(lambda a, b: _p0(a, b) * np.exp(-_u_true(a, b) / KT))
    cv = B.cross_validate(phi, psi, target, 300.0, n=36, iterations=20)
    assert max(cv["basin"]) < 0.01
    assert max(abs(x) for x in cv["dg"]) < 0.5


def test_read_rama_xvg(tmp_path):
    p = tmp_path / "ALA.xvg"
    p.write_text(
        '# gmx rama\n@    title "Ramachandran Plot"\n@ s0 symbol 2\n-139.4  134.75  ALA-2\n-54.66  133.13  ALA-2\n'
    )
    phi, psi = B.read_rama(str(p))
    np.testing.assert_allclose(phi, [-139.4, -54.66])
    np.testing.assert_allclose(psi, [134.75, 133.13])


def test_first_inversion_step_alone():
    """Without refinement, the inversion alone must already be close (smooth data)."""
    rng = np.random.default_rng(10)
    phi, psi = _sample(_p0, 400_000, rng)
    target = _target(lambda a, b: _p0(a, b) * np.exp(-_u_true(a, b) / KT))
    grid, info = B.fit_map(phi, psi, target, n=36, temperature=300.0, iterations=0, log=lambda *a: None)
    assert info["total_variation_final"] < 0.05
    assert info["total_variation_final"] < 0.2 * info["total_variation_initial"]
