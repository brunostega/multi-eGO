"""
Tests of the geometric hydrogen-bond term (multiego/omm/hbond.py).

The tests use a minimal 4-particle system N-H ... O=C, with the H-O pair
having the LJ parameters of the multi-eGO mg prior (eps = 0.2 kJ/mol,
r_min = 0.1697 nm), and check the properties stated in the module docstring.
"""

import os
import sys

import numpy as np
import openmm as mm
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from multiego.omm import hbond  # noqa: E402

EPS, RMIN = 0.2, 0.1697
C6 = 2 * EPS * RMIN**6  # LJ with minimum -EPS at RMIN
C12 = EPS * RMIN**12
N_H, O_C = 0.101, 0.123  # N-H and C=O bond lengths (nm)
DEFAULTS = hbond.HBondSettings()


def angular(theta, theta0, width):
    """The Gaussian angular factor of the term (degrees)."""
    if theta <= 90.0:
        return 0.0
    return float(np.exp(-np.deg2rad(theta - theta0) ** 2 / (2 * np.deg2rad(width) ** 2)))


def expected(r, theta_nho, theta_hoc, settings=None, c6=C6, c12=C12):
    """Analytic energy of the term, independent of the OpenMM implementation."""
    settings = settings or hbond.HBondSettings()
    if c6 == 0.0:
        return c12 / r**12
    eps, rmin = c6**2 / (4 * c12), (2 * c12 / c6) ** (1 / 6)
    repulsion = c12 / r**12 - c6 / r**6 + eps if r < rmin else 0.0
    depth = eps if settings.epsilon is None else settings.epsilon
    radial = np.exp(-max(r - settings.r0, 0.0) ** 2 / (2 * settings.width**2))
    return repulsion - depth * radial * angular(theta_nho, settings.donor_theta0, settings.donor_width) * angular(
        theta_hoc, settings.acceptor_theta0, settings.acceptor_width
    )


def lj(r):
    return C12 / r**12 - C6 / r**6


def hbond_energy(r, theta_nho=180.0, theta_hoc=147.0, settings=None, c6=C6, c12=C12):
    """
    Energy of the term for H...O distance r and the two angles (degrees).
    H at the origin, O on +x; N and C placed to give the requested angles.
    """
    settings = settings or hbond.HBondSettings()
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(10, 0, 0), mm.Vec3(0, 10, 0), mm.Vec3(0, 0, 10))
    c6_table = np.array([[0.0, c6], [c6, 0.0]])
    c12_table = np.array([[0.0, c12], [c12, 0.0]])
    # particles: 0 = H (type 0), 1 = N, 2 = O (type 1), 3 = C
    hbond.add_hbond_force(system, settings, c6_table, c12_table, [(0, 1, 0)], [(2, 3, 1)], set())
    # the replaced LJ entries must be zeroed
    assert c6_table[0, 1] == c6_table[1, 0] == c12_table[0, 1] == 0.0

    t1, t2 = np.deg2rad(theta_nho), np.deg2rad(theta_hoc)
    h = np.zeros(3)
    o = np.array([r, 0.0, 0.0])
    n = h + N_H * np.array([np.cos(np.pi - t1) * -1, np.sin(np.pi - t1), 0.0])  # angle N-H-O = t1
    c = o + O_C * np.array([np.cos(np.pi - t2), np.sin(np.pi - t2), 0.0])  # angle H-O-C = t2
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions([h + 5, n + 5, o + 5, c + 5])
    return context.getState(getEnergy=True).getPotentialEnergy()._value


def test_geometry_helper():
    """Sanity check of the test geometry itself."""
    for t in (180.0, 135.0, 90.0):
        t1 = np.deg2rad(t)
        n = N_H * np.array([np.cos(np.pi - t1) * -1, np.sin(np.pi - t1), 0.0])
        cos_angle = np.dot(n, [1.0, 0, 0]) / np.linalg.norm(n)
        assert np.rad2deg(np.arccos(cos_angle)) == pytest.approx(t)


@pytest.mark.parametrize(
    "r,nho,hoc",
    [
        (0.21, 180.0, 147.0),  # the optimum of the term
        (0.22, 152.0, 142.0),  # measured alpha-helix geometry
        (0.20, 158.0, 152.0),  # measured beta-sheet geometry
        (0.24, 140.0, 101.0),  # measured 3-10 geometry
        (0.16, 170.0, 150.0),  # inside the repulsive wall
        (0.30, 150.0, 140.0),  # far out on the Gaussian tail
    ],
)
def test_matches_analytic_form(r, nho, hoc):
    """The OpenMM force reproduces the documented functional form exactly."""
    assert hbond_energy(r, nho, hoc) == pytest.approx(expected(r, nho, hoc), rel=1e-8, abs=1e-12)


def test_optimum_is_at_the_measured_geometry():
    """
    The well is deepest between r_min and r0 (the radial factor is flat
    inside r0) and at the two measured angular optima.
    """
    best = expected(DEFAULTS.r0, DEFAULTS.donor_theta0, DEFAULTS.acceptor_theta0)
    assert best == pytest.approx(-EPS)
    for r in (0.18, 0.19, 0.20):  # the plateau
        assert expected(r, DEFAULTS.donor_theta0, DEFAULTS.acceptor_theta0) == pytest.approx(best)
    for r in (0.23, 0.25, 0.30):  # the Gaussian tail
        assert expected(r, DEFAULTS.donor_theta0, DEFAULTS.acceptor_theta0) > best
    for theta in (120.0, 135.0, 160.0, 180.0):  # away from the acceptor optimum
        assert expected(DEFAULTS.r0, DEFAULTS.donor_theta0, theta) > best
    for theta in (120.0, 135.0, 160.0, 175.0):  # away from the donor optimum
        assert expected(DEFAULTS.r0, theta, DEFAULTS.acceptor_theta0) > best


def test_acceptor_optimum_is_not_linear():
    """A linear H...O=C approach is weaker than the lone-pair direction."""
    linear = expected(DEFAULTS.r0, 180.0, 180.0)
    lone_pair = expected(DEFAULTS.r0, 180.0, DEFAULTS.acceptor_theta0)
    assert lone_pair < linear
    assert lone_pair / linear > 1.5  # and by a sizeable factor


def test_alpha_beta_balance():
    """
    With the measured optima an alpha hydrogen bond is no longer strongly
    penalised with respect to a beta one, as it was with cos^2 on both angles.
    """
    alpha = angular(152.0, DEFAULTS.donor_theta0, DEFAULTS.donor_width) * angular(
        142.0, DEFAULTS.acceptor_theta0, DEFAULTS.acceptor_width
    )
    beta = angular(158.0, DEFAULTS.donor_theta0, DEFAULTS.donor_width) * angular(
        152.0, DEFAULTS.acceptor_theta0, DEFAULTS.acceptor_width
    )
    old_alpha = np.cos(np.deg2rad(152.0)) ** 2 * np.cos(np.deg2rad(142.0)) ** 2
    old_beta = np.cos(np.deg2rad(158.0)) ** 2 * np.cos(np.deg2rad(152.0)) ** 2
    assert beta / alpha < 1.15  # was about 1.4 with cos^2
    assert beta / alpha < old_beta / old_alpha
    assert alpha > 1.5 * old_alpha  # and much less of the well is wasted


@pytest.mark.parametrize("theta", [89.0, 60.0])
def test_bent_geometry_has_no_attraction(theta):
    """An angle on the wrong side of the atom switches the attraction off."""
    for r in (0.15, 0.21, 0.25):
        energy = hbond_energy(r, theta_nho=theta)
        rmin = (2 * C12 / C6) ** (1 / 6)
        assert energy == pytest.approx(C12 / r**12 - C6 / r**6 + EPS if r < rmin else 0.0, rel=1e-9, abs=1e-12)


def test_repulsive_pair_keeps_repulsion_only():
    """A purely repulsive H-O pair (c6 = 0) keeps c12/r^12 and gets no well."""
    c12 = 1e-10
    for r in (0.15, 0.2, 0.3):
        assert hbond_energy(r, c6=0.0, c12=c12) == pytest.approx(c12 / r**12, rel=1e-9)
        assert expected(r, 180.0, 147.0, c6=0.0, c12=c12) == pytest.approx(c12 / r**12, rel=1e-9)


def test_fixed_parameters():
    """Every parameter of the well can be set explicitly."""
    settings = hbond.HBondSettings(
        epsilon=5.0,
        r0=0.19,
        width=0.03,
        donor_theta0=170.0,
        donor_width=30.0,
        acceptor_theta0=140.0,
        acceptor_width=20.0,
    )
    assert hbond_energy(0.19, 170.0, 140.0, settings=settings) == pytest.approx(-5.0, rel=1e-9)
    for r, nho, hoc in ((0.19, 170.0, 140.0), (0.22, 150.0, 155.0), (0.26, 180.0, 120.0)):
        assert hbond_energy(r, nho, hoc, settings=settings) == pytest.approx(expected(r, nho, hoc, settings), rel=1e-8)


def test_settings_json_roundtrip():
    settings = hbond.HBondSettings(epsilon=1.0, r0=None, width=0.02, cutoff=0.6)
    assert hbond.HBondSettings.from_json(settings.to_json()) == settings


def test_find_donors_acceptors():
    """Backbone H/N and O/C of the same residue; proline-like residue has no donor."""
    atoms = {
        "name": np.array(["N", "H", "CA", "C", "O", "N", "CA", "C", "O"]),
        "resnum": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]),
    }
    donors, acceptors = hbond.find_donors_acceptors(atoms)
    np.testing.assert_array_equal(donors["h"], [1])
    np.testing.assert_array_equal(donors["n"], [0])
    np.testing.assert_array_equal(acceptors["o"], [4, 8])
    np.testing.assert_array_equal(acceptors["c"], [3, 7])


# ---------------------------------------------------------------------------
# Whole-system tests on GB1 (molten-globule model generated with --hbond)
# ---------------------------------------------------------------------------

import glob  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import openmm.app as app  # noqa: E402

from multiego.omm import builder, energy, paramfile, parameters  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
GRO = os.path.join(ROOT, "tests", "test_inputs", "gpref", "gb1_mego.gro")


@pytest.fixture(scope="module")
def gb1_hbond():
    """GB1 mg parameter file generated with --hbond, its structure and box."""
    for old in glob.glob(os.path.join(ROOT, "outputs", "gpref", "hbondtest_*")):
        shutil.rmtree(old)
    cmd = [sys.executable, os.path.join(ROOT, "multiego.py"), "--system", "gpref", "--egos", "mg", "--no_header"]
    cmd += ["--explicit_name", "hbondtest", "--backend", "openmm", "--hbond"]
    cmd += ["--inputs_dir", os.path.join(ROOT, "tests", "test_inputs")]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out = os.path.join(ROOT, "outputs", "gpref", "hbondtest_1")
    params = paramfile.read(os.path.join(out, paramfile.DEFAULT_FILENAME))
    shutil.rmtree(out)
    gro = app.GromacsGroFile(GRO)
    positions = np.array(gro.getPositions()._value)
    return params, positions, gro.getPeriodicBoxVectors()


def reference_hbond_energy(params, positions, settings):
    """
    Independent numpy implementation of the hydrogen-bond term over all
    donor-acceptor pairs of a single molecule (no PBC needed: GB1 is whole
    and far from the box edges), with the exclusions rebuilt from the
    parameter file.  Returns the total energy and the per-pair energies.
    """
    mol = next(iter(params.molecules.values()))
    donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
    c6_table, c12_table = params.nonbonded.dense_tables()
    types = mol.atoms["nb_type"]
    b = mol.bonded
    excluded = builder.bonded_exclusions(mol.n_atoms, b["bonds"]["ai"], b["bonds"]["aj"], params.nrexcl)
    for key in ("exclusions", "pairs"):
        excluded |= {(min(i, j), max(i, j)) for i, j in zip(b[key]["ai"], b[key]["aj"])}

    def angle(a, b_, c):
        u, v = positions[a] - positions[b_], positions[c] - positions[b_]
        return np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))

    pair_energies = {}
    for h, n in zip(donors["h"], donors["n"]):
        for o, c in zip(acceptors["o"], acceptors["c"]):
            if (min(h, o), max(h, o)) in excluded:
                continue
            r = np.linalg.norm(positions[h] - positions[o])
            if r >= settings.cutoff:
                continue
            c6, c12 = c6_table[types[h], types[o]], c12_table[types[h], types[o]]
            if settings.min_residue_separation > 0:
                resnum = mol.atoms["resnum"]
                if abs(int(resnum[h]) - int(resnum[o])) < settings.min_residue_separation:
                    continue
            pair_energies[(h, o)] = expected(
                r, np.degrees(angle(n, h, o)), np.degrees(angle(h, o, c)), settings, c6=c6, c12=c12
            )
    return sum(pair_energies.values()), pair_energies, len(donors["h"]), len(acceptors["o"])


def test_hbond_settings_stored(gb1_hbond):
    params, _, _ = gb1_hbond
    assert params.hbond == hbond.HBondSettings()


def test_hbond_energy_matches_reference(gb1_hbond):
    """The OpenMM force equals the independent implementation on the native structure."""
    params, positions, box = gb1_hbond
    system = builder.create_system(params, box_vectors=box)
    terms = energy.decompose(system, positions)
    total, pair_energies, n_donors, n_acceptors = reference_hbond_energy(params, positions, params.hbond)
    force = next(f for f in system.getForces() if f.getName() == "hbonds")
    assert (force.getNumDonors(), force.getNumAcceptors()) == (n_donors, n_acceptors)
    assert terms["hbonds"]["energy"] == pytest.approx(total, rel=1e-8)

    # Native GB1 (a 4-stranded beta-sheet packed on a helix): the term must
    # separate geometric hydrogen bonds (H...O < 0.25 nm, N-H...O > 120 deg,
    # H...O=C > 90 deg) from all other donor-acceptor pairs.
    mol = next(iter(params.molecules.values()))
    donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
    n_of_h, c_of_o = dict(zip(donors["h"], donors["n"])), dict(zip(acceptors["o"], acceptors["c"]))

    def angle(i, j, k):
        u, v = positions[i] - positions[j], positions[k] - positions[j]
        return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))

    eps = 0.2  # mg prior H-O epsilon
    geometric, other, far = [], [], []
    for (h, o), e in pair_energies.items():
        r = np.linalg.norm(positions[h] - positions[o])
        theta_nho, theta_hoc = angle(n_of_h[h], h, o), angle(h, o, c_of_o[o])
        if r < 0.25 and theta_nho > 120 and theta_hoc > 90:
            geometric.append(e / eps)
        else:
            other.append(e / eps)
            # clearly not a hydrogen bond: far away, or pointing the wrong way
            if r > 0.30 or theta_nho < 100 or theta_hoc < 100:
                far.append(e / eps)
    geometric, other, far = np.array(geometric), np.array(other), np.array(far)
    assert len(geometric) >= 30, f"only {len(geometric)} geometric hydrogen bonds in native GB1"
    assert np.all(geometric <= 0.0)
    assert np.sum(geometric < -0.1) >= 0.75 * len(geometric)
    # The potential is smooth while the criterion is a sharp cut, so pairs just
    # outside it keep some attraction; what must hold is that they are much
    # weaker on average, and that clearly non-bonded geometries get nothing.
    assert geometric.mean() < 3 * other.mean()
    assert np.all(far > -0.02), "attraction for a clearly non-hydrogen-bonded pair"


def test_hbond_only_changes_backbone_h_o(gb1_hbond):
    """
    With the term on: every other force is unchanged, and in the LJ tables only
    the backbone donor-acceptor entries change (to zero).
    """
    params, positions, box = gb1_hbond
    on = builder.create_system(params, box_vectors=box)
    off = builder.create_system(params, box_vectors=box, hbond=False)
    p_on, p_off = parameters.extract(on), parameters.extract(off)
    for term in ("bonds", "angles", "proper_dihedrals", "improper_dihedrals", "pairs"):
        assert p_on[term] == p_off[term]

    mol = next(iter(params.molecules.values()))
    donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
    changed = np.zeros(p_on["nonbonded"]["c6"].shape, dtype=bool)
    changed[np.ix_(donors["h"], acceptors["o"])] = True
    changed |= changed.T
    for name in ("c6", "c12"):
        assert np.all(p_on["nonbonded"][name][changed] == 0.0)
        np.testing.assert_array_equal(p_on["nonbonded"][name][~changed], p_off["nonbonded"][name][~changed])
    assert p_on["nonbonded"]["exclusions"] == p_off["nonbonded"]["exclusions"]


def test_hbond_off_by_default(gb1_hbond):
    """A parameter file generated without --hbond has no hydrogen-bond force."""
    params, _, box = gb1_hbond
    params.attrs.pop("hbond")
    try:
        system = builder.create_system(params, box_vectors=box)
        assert "hbonds" not in [f.getName() for f in system.getForces()]
    finally:
        params.attrs["hbond"] = hbond.HBondSettings().to_json()


def test_hbond_forces_are_energy_gradient(gb1_hbond):
    """
    Forces of the term equal minus the finite-difference gradient of its
    energy, on atoms involved in hydrogen bonds (checks the derivatives of the
    step/max/select pieces of the expression).
    """
    params, positions, box = gb1_hbond
    system = builder.create_system(params, box_vectors=box)
    for force in system.getForces():
        force.setForceGroup(1 if force.getName() == "hbonds" else 0)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))

    def state(x):
        context.setPositions(x)
        s = context.getState(getEnergy=True, getForces=True, groups={1})
        return s.getPotentialEnergy()._value, s.getForces(asNumpy=True)._value

    # a slightly displaced structure, so that no pair sits exactly at a kink
    x0 = positions + np.random.default_rng(1).normal(0.0, 0.005, positions.shape)
    _, forces = state(x0)
    mol = next(iter(params.molecules.values()))
    donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
    atoms = list(donors["h"][:6]) + list(donors["n"][:3]) + list(acceptors["o"][:6]) + list(acceptors["c"][:3])
    delta = 1e-6
    for atom in atoms:
        for dim in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[atom, dim] += delta
            xm[atom, dim] -= delta
            numerical = -(state(xp)[0] - state(xm)[0]) / (2 * delta)
            assert forces[atom, dim] == pytest.approx(numerical, rel=1e-4, abs=1e-5)
