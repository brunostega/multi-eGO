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


def lj(r):
    return C12 / r**12 - C6 / r**6


def hbond_energy(r, theta_nho=180.0, theta_hoc=180.0, settings=None, c6=C6, c12=C12):
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
    # at 180/180 the energy at r_min is -eps: checked below; here check angles via cos
    for t in (180.0, 135.0, 90.0):
        t1 = np.deg2rad(t)
        n = N_H * np.array([np.cos(np.pi - t1) * -1, np.sin(np.pi - t1), 0.0])
        cos_angle = np.dot(n, [1.0, 0, 0]) / np.linalg.norm(n)
        assert np.rad2deg(np.arccos(cos_angle)) == pytest.approx(t)


@pytest.mark.parametrize("r", [0.140, 0.150, 0.160, 0.1697])
def test_ideal_geometry_equals_lj_inside_minimum(r):
    """Linear H-bond, r <= r_min: identical to the original LJ."""
    assert hbond_energy(r) == pytest.approx(lj(r), rel=1e-9, abs=1e-12)


def test_ideal_geometry_well_and_tail():
    """At r_min the depth is eps; beyond, the well follows the LJ over the H-bond range."""
    assert hbond_energy(RMIN) == pytest.approx(-EPS, rel=1e-9)
    # zero slope at the minimum (continuous derivative with the flat inner part)
    h = 2e-4
    assert (hbond_energy(RMIN + h) - hbond_energy(RMIN)) / h == pytest.approx(0.0, abs=0.2)
    # half depth at the same distance as the LJ: 1.227 r_min
    r_half = RMIN * (1 - 2**-0.5) ** (-1 / 6)
    assert lj(r_half) == pytest.approx(-EPS / 2, rel=1e-9)
    assert hbond_energy(r_half) == pytest.approx(-EPS / 2, rel=1e-9)
    # close to the LJ over the hydrogen-bond range, tail cut beyond
    for x in (1.1, 1.2, 1.3):
        assert hbond_energy(x * RMIN) == pytest.approx(lj(x * RMIN), abs=0.1 * EPS)
    assert lj(1.6 * RMIN) < hbond_energy(1.6 * RMIN) <= 0.0
    assert abs(hbond_energy(0.45)) < 1e-10


@pytest.mark.parametrize("theta", [90.0, 60.0])
def test_bent_geometry_has_no_attraction(theta):
    """A non-hydrogen-bonding N-H...O angle switches the attraction off."""
    for r in (0.15, RMIN, 0.2, 0.25):
        energy = hbond_energy(r, theta_nho=theta)
        expected = lj(r) + EPS if r < RMIN else 0.0  # only the WCA wall remains
        assert energy == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_angular_dependence():
    """Attraction scales as cos^2 of each angle."""
    r = RMIN
    for t1, t2 in ((150.0, 180.0), (180.0, 140.0), (160.0, 150.0)):
        expected = -EPS * np.cos(np.deg2rad(t1)) ** 2 * np.cos(np.deg2rad(t2)) ** 2
        assert hbond_energy(r, t1, t2) == pytest.approx(expected, rel=1e-9)


def test_repulsive_pair_keeps_repulsion_only():
    """A purely repulsive H-O pair (c6 = 0) keeps c12/r^12 and gets no well."""
    c12 = 1e-10
    for r in (0.15, 0.2, 0.3):
        assert hbond_energy(r, c6=0.0, c12=c12) == pytest.approx(c12 / r**12, rel=1e-9)


def test_fixed_parameters():
    """eps, r0 and width can be fixed instead of derived from the LJ pair."""
    settings = hbond.HBondSettings(epsilon=5.0, r0=0.19, width=0.03)
    assert hbond_energy(0.19, settings=settings) == pytest.approx(-5.0, rel=1e-9)
    assert hbond_energy(0.22, settings=settings) == pytest.approx(-5.0 * np.exp(-(0.03**2) / (2 * 0.03**2)), rel=1e-9)


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
            if c6 == 0:
                pair_energies[(h, o)] = c12 / r**12
                continue
            eps, rmin = c6**2 / (4 * c12), (2 * c12 / c6) ** (1 / 6)
            erep = c12 / r**12 - c6 / r**6 + eps if r < rmin else 0.0
            w = rmin * hbond.HALF_WIDTH_RATIO
            cnho, choc = np.cos(angle(n, h, o)), np.cos(angle(h, o, c))
            ang = (cnho**2 if cnho < 0 else 0.0) * (choc**2 if choc < 0 else 0.0)
            pair_energies[(h, o)] = erep - eps * np.exp(-max(r - rmin, 0.0) ** 2 / (2 * w**2)) * ang
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
    geometric, other = [], []
    for (h, o), e in pair_energies.items():
        is_hbond = (
            np.linalg.norm(positions[h] - positions[o]) < 0.25
            and angle(n_of_h[h], h, o) > 120
            and angle(h, o, c_of_o[o]) > 90
        )
        (geometric if is_hbond else other).append(e / eps)
    geometric, other = np.array(geometric), np.array(other)
    assert len(geometric) >= 30, f"only {len(geometric)} geometric hydrogen bonds in native GB1"
    assert np.all(geometric <= 0.0)
    assert np.sum(geometric < -0.1) >= 0.75 * len(geometric)
    assert np.all(other > -0.1), "attraction for a non-hydrogen-bonded donor-acceptor pair"


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
