"""
Small unit tests of the OpenMM backend helpers (fast, no multi-eGO run).

The end-to-end check of the backend is tests/test_openmm_parity.py; these
tests pin down the individual building blocks so that a failure there is
easier to locate.
"""

import os
import sys

import numpy as np
import openmm as mm
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from multiego.nonbonded import TypePrior  # noqa: E402
from multiego.omm import builder, paramfile  # noqa: E402
from multiego.omm.simulate import parse_molecules  # noqa: E402


def test_bonded_exclusions_linear_chain():
    """On a chain 0-1-2-3-4-5, nrexcl = 3 excludes pairs up to 3 bonds apart."""
    excl = builder.bonded_exclusions(6, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], nrexcl=3)
    expected = {(i, j) for i in range(6) for j in range(i + 1, 6) if j - i <= 3}
    assert excl == expected


def test_bonded_exclusions_ring():
    """In a 6-ring every atom is at most 3 bonds from every other one."""
    excl = builder.bonded_exclusions(6, [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], nrexcl=3)
    assert len(excl) == 15


@pytest.mark.parametrize("shift", [True, False])
def test_lj_expression_two_particles(shift):
    """The table LJ force gives c12/r^12 - c6/r^6 (minus its value at rc)."""
    c6, c12, rc, r = 1e-3, 2e-6, 2.5, 0.45
    force = mm.CustomNonbondedForce(builder.lj_expression(rc, shift))
    force.addPerParticleParameter("type")
    force.addTabulatedFunction("c6tab", mm.Discrete2DFunction(1, 1, [c6]))
    force.addTabulatedFunction("c12tab", mm.Discrete2DFunction(1, 1, [c12]))
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(rc)
    system = mm.System()
    for _ in range(2):
        system.addParticle(1.0)
        force.addParticle([0])
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(10, 0, 0), mm.Vec3(0, 10, 0), mm.Vec3(0, 0, 10))
    context = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [r, 0, 0]])
    energy = context.getState(getEnergy=True).getPotentialEnergy()._value
    expected = c12 / r**12 - c6 / r**6
    if shift:
        expected -= c12 / rc**12 - c6 / rc**6
    assert energy == pytest.approx(expected, rel=1e-12)


def _prior(types, has_rule, c6, c12):
    """A small TypePrior for tests (epsilon/sigma are not used by the tables)."""
    z = np.zeros_like(np.asarray(c6, float))
    return TypePrior(np.array(types), np.array(has_rule), np.array(c6, float), np.array(c12, float), z, z)


def test_dense_tables_precedence():
    """Override > type prior (where a rule exists) > combination rule 1."""
    # three atoms: chemical types X, X, Y; the prior defines only X-Y
    prior = _prior(["X", "Y"], [[False, True], [True, False]], [[0, 0.3], [0.3, 0]], [[0, 5.0], [5.0, 0]])
    nb = paramfile.NonbondedParameters(
        sb_types=np.array(["A", "B", "C"]),
        chem_type=np.array([0, 0, 1]),
        default_c6=np.zeros(3),
        default_c12=np.array([1.0, 4.0, 9.0]),
        prior=prior,
        overrides={"i": np.array([1]), "j": np.array([2]), "c6": np.array([0.5]), "c12": np.array([7.0])},
    )
    c6, c12 = nb.dense_tables()
    assert c12[0, 1] == c12[1, 0] == pytest.approx(2.0)  # X-X: no rule -> sqrt(1*4)
    assert c12[0, 2] == c12[2, 0] == 5.0  # X-Y: prior
    assert c6[0, 2] == 0.3
    assert c12[1, 2] == c12[2, 1] == 7.0  # override, symmetric
    assert c6[1, 2] == 0.5


def test_parameter_file_roundtrip(tmp_path):
    """write() followed by read() returns identical data."""
    atoms = {
        "sb_type": np.array(["N_X_1", "CA_X_1"]),
        "type": np.array(["N", "CH1"]),
        "name": np.array(["N", "CA"]),
        "resname": np.array(["ALA", "ALA"]),
        "resnum": np.array([1, 1]),
        "atomic_number": np.array([7, 6]),
        "mass": np.array([14.0, 13.0]),
        "nb_type": np.array([0, 1]),
    }
    bonded = {
        term: {col: np.array([], dtype=dt) for col, dt in cols} for term, cols in paramfile.BONDED_COLUMNS.items()
    }
    bonded["bonds"] = {"ai": np.array([0]), "aj": np.array([1]), "r0": np.array([0.147]), "k": np.array([9e4])}
    prior = _prior(["CH1", "N"], [[True, False], [False, False]], [[1e-3, 0], [0, 0]], [[2e-6, 0], [0, 0]])
    nb = paramfile.NonbondedParameters(
        np.array(["N_X_1", "CA_X_1"]),
        np.array([1, 0]),
        np.zeros(2),
        np.array([1e-6, 2e-6]),
        prior,
        {"i": np.array([0]), "j": np.array([1]), "c6": np.array([1e-3]), "c12": np.array([3e-6])},
    )
    original = paramfile.MegoParameters(
        {"X": paramfile.MoleculeParameters("X", atoms, bonded)},
        nb,
        {"format_version": paramfile.FORMAT_VERSION, "nrexcl": 3},
    )
    path = tmp_path / "p.h5"
    paramfile.write(path, original)
    back = paramfile.read(path)
    assert list(back.molecules) == ["X"]
    for col, values in atoms.items():
        np.testing.assert_array_equal(back.molecules["X"].atoms[col], values)
    np.testing.assert_array_equal(back.molecules["X"].bonded["bonds"]["k"], [9e4])
    assert len(back.molecules["X"].bonded["pairs"]["ai"]) == 0
    np.testing.assert_array_equal(back.nonbonded.prior.types, ["CH1", "N"])
    np.testing.assert_array_equal(back.nonbonded.prior.has_rule, prior.has_rule)
    for table_back, table in zip(back.nonbonded.dense_tables(), nb.dense_tables()):
        np.testing.assert_array_equal(table_back, table)


def test_parse_molecules():
    assert parse_molecules("") is None
    assert parse_molecules("GB1:2,LIG") == [("GB1", 2), ("LIG", 1)]
