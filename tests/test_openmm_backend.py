"""
Tests for the OpenMM backend (``multiego.openmm_backend``).

1. Reader: the multi-eGO dialect is accepted, anything else is rejected.
2. Analytic checks of single terms (exclusions, improper wrapping, LJ table
   and potential shift).
3. Parameter level: every parameter in the System equals the file.
4. Independent parser: total forces equal those of OpenMM's own
   ``app.GromacsTopFile`` on the same files (exclusions are compared too).
5. GROMACS: per-term energies and forces equal ``gmx mdrun -rerun``
   (references in ``tests/openmm_reference``, regenerated with
   ``make_gromacs_reference.py``).
"""

import math
import os
import sys

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "openmm_reference"))
import cases  # noqa: E402

from multiego.openmm_backend import (  # noqa: E402
    FORCE_GROUPS,
    TopologyError,
    create_integrator,
    create_system,
    create_topology,
    load,
    molecule_exclusions,
    read_topology,
)

REFERENCE = openmm.Platform.getPlatformByName("Reference")
BOX = np.diag([8.0, 8.0, 8.0])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
HEADER = """
[ defaults ]
  1  1  no  1.0  1.0

[ atomtypes ]
 A  6 12.011 0.0 A 1.0e-03 1.0e-06
 B  8 15.999 0.0 A 4.0e-03 4.0e-06
 C  1  4.032 0.0 A 0.0     0.0
"""


def write_top(tmp_path, body, header=HEADER, name="topol.top"):
    path = tmp_path / name
    path.write_text(header + body)
    return str(path)


def chain_molecule(n, nrexcl=3, extra=""):
    """Linear chain of n atoms of type A, bonds i-(i+1)."""
    atoms = "\n".join(f" {i} A 1 RES A{i} {i}" for i in range(1, n + 1))
    bonds = "\n".join(f" {i} {i + 1} 1 0.15 1000.0" for i in range(1, n))
    return f"""
[ moleculetype ]
 CHAIN {nrexcl}

[ atoms ]
{atoms}

[ bonds ]
{bonds}
{extra}
"""


def context(system, positions):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPositions(positions)
    return ctx


def energy(ctx, group=None):
    kw = {"groups": {FORCE_GROUPS[group]}} if group else {}
    return ctx.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def forces(ctx):
    return ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def force_by_name(system, name):
    return next(f for f in system.getForces() if f.getName() == name)


@pytest.fixture(scope="module", params=list(cases.CASES))
def case(request, tmp_path_factory):
    name = request.param
    wd = tmp_path_factory.mktemp(name)
    top_path, gro_path, frames, box = cases.prepare(name, str(wd))
    top = read_topology(top_path)
    system = create_system(top, np.diag(box))
    return dict(name=name, top_path=top_path, gro_path=gro_path, frames=frames, box=box, top=top, system=system)


# ---------------------------------------------------------------------------
# 1. reader
# ---------------------------------------------------------------------------
def test_reader_basic(tmp_path):
    body = chain_molecule(3) + "\n[ system ]\ntest\n\n[ molecules ]\nCHAIN 2\n"
    top = read_topology(write_top(tmp_path, body))
    mt = top.moleculetypes["CHAIN"]
    assert len(mt.atoms) == 3 and mt.nrexcl == 3
    assert mt.atoms[0].mass == pytest.approx(12.011)
    assert mt.bonds[0] == (0, 1, 0.15, 1000.0)
    assert top.molecules == [("CHAIN", 2)]
    assert top.n_atoms == 6


def test_reader_include_and_ifdef(tmp_path):
    (tmp_path / "ff.itp").write_text(HEADER)
    body = '#include "ff.itp"\n' + chain_molecule(2) + '\n#ifdef POSRES\n#include "missing.itp"\n#endif\n'
    body += "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    top = read_topology(write_top(tmp_path, body, header=""))
    assert top.n_atoms == 2
    with pytest.raises(TopologyError, match="not found"):
        read_topology(str(tmp_path / "topol.top"), defines=("POSRES",))


@pytest.mark.parametrize(
    "extra, message",
    [
        ("#define FOO 1\n", "#define"),
        ("[ cmap ]\n 1 2 3 4 5 1\n", "unsupported section"),
        ("[ angles ]\n 1 2 3 5 109.5 400.0 0.1 0.1\n", "function type 5"),
        ("[ dihedrals ]\n 1 2 3 4 3 1 2 3 4 5 6\n", "function type 3"),
        ("[ bonds ]\n 1 9 1 0.15 1000.0\n", "atom 9 not in molecule"),
        ("[ pairs ]\n 1 4 1 0.0\n", "pairs need explicit"),
    ],
)
def test_reader_rejects(tmp_path, extra, message):
    body = chain_molecule(4) + extra + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    with pytest.raises(TopologyError, match=message):
        read_topology(write_top(tmp_path, body))


@pytest.mark.parametrize(
    "header, message",
    [
        (HEADER.replace("1  1  no", "1  2  no"), "comb-rule 1"),
        (HEADER.replace("1  1  no", "1  1  yes"), "gen-pairs"),
        (HEADER + "\n[ nonbond_params ]\n A D 1 0.0 1.0e-6\n", "unknown atom type D"),
    ],
)
def test_reader_rejects_header(tmp_path, header, message):
    body = chain_molecule(2) + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    with pytest.raises(TopologyError, match=message):
        read_topology(write_top(tmp_path, body, header=header))


def test_charges_rejected(tmp_path):
    body = (
        chain_molecule(2).replace(" 1 A 1 RES A1 1", " 1 A 1 RES A1 1 0.5")
        + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    )
    with pytest.raises(ValueError, match="charges"):
        create_system(read_topology(write_top(tmp_path, body)), BOX)


def test_cutoff_longer_than_half_box(tmp_path):
    body = chain_molecule(2) + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    with pytest.raises(ValueError, match="half the box"):
        create_system(read_topology(write_top(tmp_path, body)), np.diag([4.0, 8.0, 8.0]))


# ---------------------------------------------------------------------------
# 2. analytic checks
# ---------------------------------------------------------------------------
def test_exclusions_nrexcl_and_explicit(tmp_path):
    extra = "\n[ pairs ]\n 1 5 1 0.0 1.0e-6\n\n[ exclusions ]\n 1 6\n"
    body = chain_molecule(7, extra=extra) + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    mt = read_topology(write_top(tmp_path, body)).moleculetypes["CHAIN"]
    excl = molecule_exclusions(mt)
    expected = {(i, j) for i in range(7) for j in range(i + 1, 7) if j - i <= 3} | {(0, 5)}
    assert excl == expected  # (0, 4) is a pair but is not excluded automatically
    mt.nrexcl = 1
    assert molecule_exclusions(mt) == {(i, i + 1) for i in range(6)} | {(0, 5)}


def test_exclusions_per_copy(tmp_path):
    body = chain_molecule(3) + "\n[ system ]\nx\n[ molecules ]\nCHAIN 2\n"
    system = create_system(read_topology(write_top(tmp_path, body)), BOX)
    nb = force_by_name(system, "LJ (SR)")
    got = {tuple(sorted(nb.getExclusionParticles(k))) for k in range(nb.getNumExclusions())}
    assert got == {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}


def _four_atoms(tmp_path, dihedral_line):
    atoms = "\n".join(f" {i} C 1 RES C{i} {i}" for i in range(1, 5))
    body = f"""
[ moleculetype ]
 M 3
[ atoms ]
{atoms}
[ dihedrals ]
{dihedral_line}
[ system ]
x
[ molecules ]
M 1
"""
    return create_system(read_topology(write_top(tmp_path, body)), BOX, com_interval=0)


def _positions_with_dihedral(phi):
    """Atoms 1-2-3-4 around the z axis; the dihedral is computed by _dihedral."""
    b = 0.15
    p0 = np.array([b, 0.0, 0.0])
    p1 = np.zeros(3)
    p2 = np.array([0.0, 0.0, b])
    p3 = p2 + b * np.array([math.cos(phi), math.sin(phi), 0.0])
    return np.array([p0, p1, p2, p3]) + 4.0


def _dihedral(x):
    """IUPAC dihedral (degrees) of four positions, independent of OpenMM."""
    b0, b1, b2 = x[0] - x[1], x[2] - x[1], x[3] - x[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


@pytest.mark.parametrize("phi_deg", [-175.0, -60.0, 0.0, 95.0, 178.0])
def test_improper_wrapping(tmp_path, phi_deg):
    xi0, k = 170.0, 300.0
    system = _four_atoms(tmp_path, f" 1 2 3 4 2 {xi0} {k}")
    x = _positions_with_dihedral(math.radians(phi_deg))
    e = energy(context(system, x), "Improper Dih.")
    d = (_dihedral(x) - xi0 + 180.0) % 360.0 - 180.0
    assert e == pytest.approx(0.5 * k * math.radians(d) ** 2, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("phi_deg", [-120.0, 30.0, 150.0])
def test_proper_periodic_multiple_terms(tmp_path, phi_deg):
    lines = " 1 2 3 4 1 0.0 5.92 3\n 1 2 3 4 1 120.0 2.50 2\n 1 2 3 4 9 180.0 2.00 1"
    system = _four_atoms(tmp_path, lines)
    x = _positions_with_dihedral(math.radians(phi_deg))
    e = energy(context(system, x), "Proper Dih.")
    p = math.radians(_dihedral(x))
    expected = sum(
        k * (1 + math.cos(n * p - math.radians(p0))) for p0, k, n in [(0, 5.92, 3), (120, 2.5, 2), (180, 2.0, 1)]
    )
    assert e == pytest.approx(expected, rel=1e-9)


def _two_atoms(tmp_path, nonbond_params=""):
    header = HEADER + nonbond_params
    body = """
[ moleculetype ]
 X 3
[ atoms ]
 1 A 1 RES A 1
[ moleculetype ]
 Y 3
[ atoms ]
 1 B 1 RES B 1
[ system ]
x
[ molecules ]
X 1
Y 1
"""
    return create_system(read_topology(write_top(tmp_path, body, header=header)), BOX, com_interval=0)


@pytest.mark.parametrize("r", [0.3, 0.5, 1.2, 2.49, 2.51])
def test_lj_combination_rule_and_shift(tmp_path, r):
    rc = 2.5
    c6, c12 = math.sqrt(1.0e-3 * 4.0e-3), math.sqrt(1.0e-6 * 4.0e-6)
    e = energy(context(_two_atoms(tmp_path), np.array([[2.0, 2.0, 2.0], [2.0 + r, 2.0, 2.0]])), "LJ (SR)")
    expected = (c12 * (r**-12 - rc**-12) - c6 * (r**-6 - rc**-6)) if r < rc else 0.0
    assert e == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_lj_nonbond_params_override(tmp_path):
    r, rc = 0.4, 2.5
    system = _two_atoms(tmp_path, "\n[ nonbond_params ]\n B A 1 2.0e-3 3.0e-6\n")
    e = energy(context(system, np.array([[2.0, 2.0, 2.0], [2.0 + r, 2.0, 2.0]])), "LJ (SR)")
    assert e == pytest.approx(3.0e-6 * (r**-12 - rc**-12) - 2.0e-3 * (r**-6 - rc**-6), rel=1e-9)


def test_pairs_have_no_cutoff_and_no_shift(tmp_path):
    atoms = "\n".join(f" {i} A 1 RES A{i} {i}" for i in range(1, 3))
    body = f"""
[ moleculetype ]
 M 0
[ atoms ]
{atoms}
[ pairs ]
 1 2 1 1.0e-3 1.0e-6
[ exclusions ]
 1 2
[ system ]
x
[ molecules ]
M 1
"""
    system = create_system(read_topology(write_top(tmp_path, body)), BOX, com_interval=0)
    for r in (0.5, 3.0):
        ctx = context(system, np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]]))
        assert energy(ctx, "LJ-14") == pytest.approx(1.0e-6 * r**-12 - 1.0e-3 * r**-6, rel=1e-9)
        assert energy(ctx, "LJ (SR)") == 0.0


def test_integrator_and_com_remover(tmp_path):
    integ = create_integrator(300.0)
    assert integ.getStepSize().value_in_unit(unit.picosecond) == pytest.approx(0.005)
    assert integ.getFriction().value_in_unit(unit.picosecond**-1) == pytest.approx(0.04)
    assert integ.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(300.0)
    body = chain_molecule(2) + "\n[ system ]\nx\n[ molecules ]\nCHAIN 1\n"
    system = create_system(read_topology(write_top(tmp_path, body)), BOX)
    cmm = [f for f in system.getForces() if isinstance(f, openmm.CMMotionRemover)]
    assert len(cmm) == 1 and cmm[0].getFrequency() == 20


# ---------------------------------------------------------------------------
# 3. parameter level, on the regression-test models
# ---------------------------------------------------------------------------
def _expected_terms(top):
    """Per-term parameter lists built straight from the parsed file."""
    deg = math.pi / 180.0
    exp = {k: [] for k in ("bonds", "angles", "propers", "impropers", "pairs")}
    masses = []
    o = 0
    for name, count in top.molecules:
        mt = top.moleculetypes[name]
        for _ in range(count):
            masses += [a.mass for a in mt.atoms]
            exp["bonds"] += [(o + i, o + j, b0, k) for i, j, b0, k in mt.bonds]
            exp["angles"] += [(o + i, o + j, o + k, t * deg, kt) for i, j, k, t, kt in mt.angles]
            exp["propers"] += [(o + i, o + j, o + k, o + l, n, p * deg, kp) for i, j, k, l, p, kp, n in mt.propers]
            exp["impropers"] += [(o + i, o + j, o + k, o + l, x * deg, kx) for i, j, k, l, x, kx in mt.impropers]
            exp["pairs"] += [(o + i, o + j, c6, c12) for i, j, c6, c12 in mt.pairs]
            o += len(mt.atoms)
    return exp, masses


def _system_terms(system):
    def q(x):
        return x._value if unit.is_quantity(x) else x

    b = force_by_name(system, "Bond")
    a = force_by_name(system, "Angle")
    p = force_by_name(system, "Proper Dih.")
    im = force_by_name(system, "Improper Dih.")
    pr = force_by_name(system, "LJ-14")
    return {
        "bonds": [tuple(q(x) for x in b.getBondParameters(k)) for k in range(b.getNumBonds())],
        "angles": [tuple(q(x) for x in a.getAngleParameters(k)) for k in range(a.getNumAngles())],
        "propers": [tuple(q(x) for x in p.getTorsionParameters(k)) for k in range(p.getNumTorsions())],
        "impropers": [
            tuple(im.getTorsionParameters(k)[:4]) + tuple(im.getTorsionParameters(k)[4])
            for k in range(im.getNumTorsions())
        ],
        "pairs": [
            tuple(pr.getBondParameters(k)[:2]) + tuple(pr.getBondParameters(k)[2]) for k in range(pr.getNumBonds())
        ],
    }


def test_parameters_match_file(case):
    exp, masses = _expected_terms(case["top"])
    got = _system_terms(case["system"])
    for term in exp:
        assert len(got[term]) == len(exp[term]), term
        np.testing.assert_allclose(np.array(got[term], float), np.array(exp[term], float), rtol=1e-12, err_msg=term)
    sys_masses = [case["system"].getParticleMass(i)._value for i in range(case["system"].getNumParticles())]
    np.testing.assert_allclose(sys_masses, masses, rtol=1e-12)


def test_lj_table_matches_file(case):
    """Every type pair seen by the nonbonded force has the file's c6/c12."""
    top, system = case["top"], case["system"]
    nb = force_by_name(system, "LJ (SR)")
    n6, _, c6 = nb.getTabulatedFunction(0).getFunctionParameters()
    n12, _, c12 = nb.getTabulatedFunction(1).getFunctionParameters()
    c6 = np.array(c6).reshape(n6, n6, order="F")
    c12 = np.array(c12).reshape(n12, n12, order="F")
    index = {}
    for name, count in top.molecules:
        for atom in top.moleculetypes[name].atoms:
            index[atom.type] = None
    atom_types = [a.type for name, count in top.molecules for _ in range(count) for a in top.moleculetypes[name].atoms]
    for p, t in enumerate(atom_types):
        index[t] = int(nb.getParticleParameters(p)[0])
    assert c6.shape == (len(index), len(index))
    for (a, b), (p6, p12) in top.nonbond_params.items():
        if a in index and b in index:
            assert c6[index[a], index[b]] == pytest.approx(p6, rel=1e-12, abs=0)
            assert c12[index[a], index[b]] == pytest.approx(p12, rel=1e-12, abs=0)
    # a type pair absent from nonbond_params uses the geometric combination
    missing = next(((a, b) for a in index for b in index if (min(a, b), max(a, b)) not in top.nonbond_params), None)
    if missing:
        a, b = missing
        ta, tb = top.atomtypes[a], top.atomtypes[b]
        assert c6[index[a], index[b]] == pytest.approx(math.sqrt(ta.c6 * tb.c6), rel=1e-12, abs=0)
        assert c12[index[a], index[b]] == pytest.approx(math.sqrt(ta.c12 * tb.c12), rel=1e-12, abs=0)


def test_topology_object(case):
    omm_top = create_topology(case["top"], np.diag(case["box"]))
    assert omm_top.getNumAtoms() == case["system"].getNumParticles()
    copies = sum(count for _, count in case["top"].molecules)
    assert omm_top.getNumChains() == copies
    assert omm_top.getNumBonds() == force_by_name(case["system"], "Bond").getNumBonds()


# ---------------------------------------------------------------------------
# 4. independent parser: OpenMM's GromacsTopFile
# ---------------------------------------------------------------------------
def _exclusion_set(force):
    return {tuple(sorted(force.getExclusionParticles(k))) for k in range(force.getNumExclusions())}


def test_against_openmm_gromacstopfile(case):
    box = case["box"]
    gtf = app.GromacsTopFile(case["top_path"], periodicBoxVectors=np.diag(box) * unit.nanometer)
    ref_system = gtf.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=2.5 * unit.nanometer,
        removeCMMotion=False,
    )
    # exclusions: GromacsTopFile stores them on its CustomNonbondedForce
    ref_nb = next(f for f in ref_system.getForces() if isinstance(f, openmm.CustomNonbondedForce))
    # The dispersion correction does not change forces and is very slow with one type per atom.
    ref_nb.setUseLongRangeCorrection(False)
    for f in ref_system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setUseDispersionCorrection(False)
    assert _exclusion_set(force_by_name(case["system"], "LJ (SR)")) == _exclusion_set(ref_nb)
    # forces (energies differ by the potential shift, which GromacsTopFile does not apply)
    c1 = context(case["system"], case["frames"][0])
    c2 = context(ref_system, case["frames"][0])
    for x in case["frames"]:
        c1.setPositions(x)
        c2.setPositions(x)
        f1, f2 = forces(c1), forces(c2)
        np.testing.assert_allclose(f1, f2, rtol=0, atol=1e-8 * np.abs(f2).max())


# ---------------------------------------------------------------------------
# 5. GROMACS
# ---------------------------------------------------------------------------
def test_against_gromacs(case):
    path = cases.reference_path(case["name"])
    if not os.path.exists(path):
        pytest.fail(f"missing GROMACS reference {path}; run tests/openmm_reference/make_gromacs_reference.py")
    ref = np.load(path)
    if str(ref["input_hash"]) != cases.input_hash(case["name"]):
        pytest.fail(
            f"GROMACS reference for {case['name']} is stale (model files changed); "
            "run tests/openmm_reference/make_gromacs_reference.py"
        )
    np.testing.assert_array_equal(ref["frames"], case["frames"])
    terms = list(ref["terms"])
    ctx = context(case["system"], case["frames"][0])
    for k, x in enumerate(case["frames"]):
        ctx.setPositions(x)
        for term in (t for t in FORCE_GROUPS if t in terms):  # CMAP has no GROMACS counterpart
            e_ref = ref["energies"][terms.index(term)][k]
            # GROMACS runs in mixed precision
            assert energy(ctx, term) == pytest.approx(e_ref, rel=1e-5, abs=0.05), f"frame {k} {term}"
        assert energy(ctx) == pytest.approx(ref["energies"][terms.index("Potential")][k], rel=1e-5, abs=0.1)
        f, f_ref = forces(ctx), ref["forces"][k]
        scale = np.abs(f_ref).max()
        assert np.abs(f - f_ref).max() < 1e-3 * scale, f"frame {k}: max force error"
        assert np.sqrt(((f - f_ref) ** 2).mean()) < 1e-3 * np.sqrt((f_ref**2).mean()), f"frame {k}: rms force error"


def test_load(case):
    if cases.CASES[case["name"]][2] != 1:
        pytest.skip("load() is exercised on single-copy inputs")
    system, topology, positions = load(case["top_path"], case["gro_path"])
    assert system.getNumParticles() == topology.getNumAtoms() == len(positions)


def test_load_atom_count_mismatch(tmp_path):
    name = "ttr_two_copies"
    top_path, _, _, _ = cases.prepare(name, str(tmp_path))
    with pytest.raises(ValueError, match="atoms"):
        load(top_path, os.path.join(cases.INP, cases.CASES[name][1]))
