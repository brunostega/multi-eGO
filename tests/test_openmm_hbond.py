"""
Geometric backbone hydrogen bond (``--hbond``).

* generation: the polar backbone prior (H-O, N-O, C-N, N-N, O-O) becomes
  c12-only in every output, and nothing else changes;
* the OpenMM term: with zero depth it reproduces the H-O Lennard-Jones of
  the GROMACS files; its geometric part equals an independent evaluation of
  the formula, within and between chains;
* learned H-O contacts written as ``[ pairs ]`` (handoff 4.6).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
pytest.importorskip("h5py")

from multiego import hbond as hb  # noqa: E402
from multiego.openmm_backend import FORCE_GROUPS, builder, paramfile  # noqa: E402
from multiego.openmm_backend import hbond as omm_hb  # noqa: E402
from multiego.openmm_backend.gromacs_reader import read_topology  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "tests", "test_inputs")
GRO = os.path.join(INPUTS, "gpref", "gb1_mego.gro")
REFERENCE = openmm.Platform.getPlatformByName("Reference")
BOX = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]


def generate(tmp, name, *extra):
    cmd = [sys.executable, os.path.join(ROOT, "multiego.py"), "--system", "gpref", "--egos", "mg", "--no_header"]
    cmd += ["--explicit_name", name, "--inputs_dir", INPUTS, "--outputs_dir", str(tmp), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return os.path.join(str(tmp), "gpref", f"{name}_1"), res.stdout


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hb")
    plain, _ = generate(tmp, "plain")
    with_hb, log = generate(tmp, "hb", "--hbond", "--hbond_epsilon_local", "10", "--hbond_epsilon_nonlocal", "5")
    return plain, with_hb, log


def positions():
    lines = open(GRO).read().splitlines()
    n = int(lines[1])
    return np.array([[float(x[20:28]), float(x[28:36]), float(x[36:44])] for x in lines[2:][:n]])


def nonbond_params(path):
    out = {}
    for line in open(path):
        tok = line.split(";")[0].split()
        if len(tok) == 5 and tok[2] == "1":
            try:
                out[(tok[0], tok[1])] = (float(tok[3]), float(tok[4]))
            except ValueError:
                pass
    return out


def energies(system, x, groups=None):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPositions(x)
    return {
        f.getName(): ctx.getState(getEnergy=True, groups={f.getForceGroup()}).getPotentialEnergy()._value
        for f in system.getForces()
        if f.getName() in FORCE_GROUPS
    }


def with_settings(params, local, nonlocal_):
    params.attrs["hbond"] = hb.HBondSettings(local, nonlocal_).to_json()
    return params


def reference_geometric(x, names, resnums, chains, eps_local, eps_nonlocal, settings):
    """The branch formula evaluated pair by pair with numpy."""
    idx = {(c, r, n): k for k, (c, r, n) in enumerate(zip(chains, resnums, names))}
    donors = [(k, idx[(c, r, "N")], r, c) for k, (c, r, n) in enumerate(zip(chains, resnums, names)) if n == "H"]
    acceptors = [(k, idx[(c, r, "C")], r, c) for k, (c, r, n) in enumerate(zip(chains, resnums, names)) if n == "O"]

    def angle(a, b, c):
        u, v = a - b, c - b
        return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))

    def gauss(t, t0, w):
        return np.exp(-((t - np.deg2rad(t0)) ** 2) / (2 * np.deg2rad(w) ** 2)) * (t > np.pi / 2)

    energy = 0.0
    for h, n, rd, cd in donors:
        for o, c, ra, ca in acceptors:
            same = cd == ca
            if same and abs(rd - ra) < settings.min_residue_separation:
                continue
            r = np.linalg.norm(x[h] - x[o])
            if r >= settings.cutoff:
                continue
            eps = eps_local if same and abs(rd - ra) < settings.nonlocal_separation else eps_nonlocal
            radial = np.exp(-max(r - settings.r0, 0) ** 2 / (2 * settings.width**2))
            energy -= (
                eps
                * radial
                * gauss(angle(x[n], x[h], x[o]), settings.donor_theta0, settings.donor_width)
                * gauss(angle(x[h], x[o], x[c]), settings.acceptor_theta0, settings.acceptor_width)
            )
    return energy


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def test_backbone_roles():
    names = ["N", "H", "CA", "C", "O", "N", "H1", "O1", "OD1", "ND2", "C"]
    types = ["N", "H", "CH1", "C", "O", "NL", "H", "OM", "O", "NT", "C"]
    assert hb.backbone_roles(names, types).tolist() == [1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 3]


def test_requires_epsilon(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "multiego.py"), "--system", "gpref", "--egos", "mg"]
    cmd += ["--inputs_dir", INPUTS, "--outputs_dir", str(tmp_path), "--hbond"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode != 0
    assert "--hbond_epsilon_local" in res.stdout + res.stderr


def test_polar_backbone_prior_is_repulsive(models):
    plain, with_hb, log = models
    assert "no hydrogen-bond term" in log
    with open(os.path.join(plain, "topol_mego.top")) as a, open(os.path.join(with_hb, "topol_mego.top")) as b:
        assert a.read() == b.read()
    old = nonbond_params(os.path.join(plain, "ffnonbonded.itp"))
    new = nonbond_params(os.path.join(with_hb, "ffnonbonded.itp"))
    assert set(old) == set(new)
    nb = paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME)).nonbonded
    table = nb.prior.backbone_c12
    role = dict(zip(nb.sb_types, nb.bb_role.tolist()))
    assert role["N_GB1_2"] == hb.ROLE_INDEX["N"] and role["N_GB1_1"] == 0  # the charged N-terminus is not
    expected_pairs = {tuple(sorted(hb.ROLE_INDEX[x] for x in p)) for p in hb.REPULSIVE_BACKBONE_PAIRS}
    changed = 0
    for key in new:
        ra, rb = role[key[0]], role[key[1]]
        if tuple(sorted((ra, rb))) in expected_pairs:
            assert new[key] == (0.0, pytest.approx(table[ra, rb]))
            changed += new[key] != old[key]
        else:
            assert new[key] == old[key], key
    assert changed > 10000
    assert np.isnan(table[hb.ROLE_INDEX["C"], hb.ROLE_INDEX["O"]])  # C-O keeps its prior


def test_parameter_file(models):
    _, with_hb, _ = models
    params = paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME))
    assert params.hbond == hb.HBondSettings(10.0, 5.0)
    assert json.loads(params.attrs["arguments"])["hbond"] is True
    # dense tables of the h5 = tables of the GROMACS files
    top = read_topology(os.path.join(with_hb, "topol_mego.top"))
    type_index, c6, c12 = builder.lj_tables(top)
    d6, d12 = params.nonbonded.dense_tables()
    order = [type_index[t] for t in params.nonbonded.sb_types]
    np.testing.assert_allclose(d6, c6[np.ix_(order, order)], rtol=1e-6, atol=1e-30)
    np.testing.assert_allclose(d12, c12[np.ix_(order, order)], rtol=1e-6, atol=1e-30)


# ---------------------------------------------------------------------------
# OpenMM term
# ---------------------------------------------------------------------------
def test_zero_depth_reproduces_gromacs(models):
    _, with_hb, _ = models
    x = positions()
    gmx = energies(builder.create_system(read_topology(os.path.join(with_hb, "topol_mego.top")), BOX), x)
    params = with_settings(paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME)), 0.0, 0.0)
    omm = energies(builder.create_system_from_params(params, BOX), x)
    for term in ("Bond", "Angle", "Proper Dih.", "Improper Dih.", "LJ-14"):
        assert omm[term] == pytest.approx(gmx[term], rel=1e-6)
    # only the potential shift / cutoff of the H-O pairs differ (c12-only, negligible)
    assert omm["LJ (SR)"] + omm["H-bond"] == pytest.approx(gmx["LJ (SR)"], abs=1e-3)
    assert omm["H-bond"] != 0.0


def test_exclusions_per_donor(models):
    _, with_hb, _ = models
    system = builder.create_system_from_params(paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME)), BOX)
    force = next(f for f in system.getForces() if f.getName() == "H-bond")
    per_donor = np.bincount([force.getExclusionParticles(k)[0] for k in range(force.getNumExclusions())])
    assert force.getNumDonors() == 55 and force.getNumAcceptors() == 55
    assert 0 < per_donor.max() <= 2


@pytest.mark.parametrize("copies", [1, 2])
def test_geometric_part_matches_formula(models, copies):
    _, with_hb, _ = models
    params = paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME))
    x1 = positions()
    x1 = x1 - x1.mean(0)
    # second copy rotated by 180 degrees and touching the first: inter-chain hydrogen bonds
    x = np.vstack([x1, x1 @ np.diag([-1.0, -1.0, 1.0]).T + [0.0, 0.0, 1.6]][:copies]) + 5.0
    box = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    group = FORCE_GROUPS["H-bond"]

    def total(local, nonlocal_):
        s = builder.create_system_from_params(with_settings(params, local, nonlocal_), box, molecules=[("GB1", copies)])
        return energies(s, x)["H-bond"]

    assert group == 7
    geometric = total(10.0, 5.0) - total(0.0, 0.0)
    at = params.molecules["GB1"].atoms
    names = [str(n) for n in at["name"]] * copies
    resnums = np.tile(np.asarray(at["resnum"], int), copies)
    chains = np.repeat(np.arange(copies), len(at["name"]))
    expected = reference_geometric(x, names, resnums, chains, 10.0, 5.0, params.hbond)
    assert geometric == pytest.approx(expected, rel=1e-9)
    assert expected < -100.0


def test_classify_pairs():
    settings = hb.HBondSettings(10.0, 10.0)
    names = ["N", "H", "C", "O"] * 6
    resnums = np.repeat(np.arange(1, 7), 4)
    donors, acceptors = omm_hb.find_donors_acceptors(names, resnums)
    assert len(donors) == 6 and len(acceptors) == 6
    within5 = {(1, 3)}  # H1-O1 topological
    pairs = [(1, 3, 0.0, 1e-9), (1, 7, 1e-3, 1e-7), (1, 15, 1e-3, 1e-7), (0, 8, 0.0, 1e-6)]
    kept, local, dropped = omm_hb.classify_pairs(pairs, donors, acceptors, resnums, within5, settings)
    assert kept == [(1, 3, 0.0, 1e-9), (0, 8, 0.0, 1e-6)]  # topological H-O, and a non H-O pair
    assert local == [(1, 7, 1e-3, 1e-7)]  # H(1)-O(2): learned, local
    assert dropped == [(1, 15)]  # H(1)-O(4): learned, handed to the hydrogen bond


def test_learned_pairs(models):
    """Learned H-O [ pairs ] beyond 5 bonds: local ones keep their value, the others go to the hydrogen bond."""
    _, with_hb, _ = models
    params = paramfile.read(os.path.join(with_hb, paramfile.DEFAULT_FILENAME))
    mol = params.molecules["GB1"]
    names = [str(n) for n in mol.atoms["name"]]
    res = np.asarray(mol.atoms["resnum"], int)
    h = {r: k for k, (n, r) in enumerate(zip(names, res)) if n == "H"}
    o = {r: k for k, (n, r) in enumerate(zip(names, res)) if n == "O"}
    local_pair = (h[10], o[11], 2e-3, 3e-7)  # |dres| = 1, beyond 5 bonds
    far_pair = (h[10], o[30], 4e-3, 6e-7)  # |dres| = 20
    p = mol.bonded["pairs"]
    for col, values in zip(("ai", "aj", "c6", "c12"), zip(local_pair, far_pair)):
        p[col] = np.concatenate([p[col], np.array(values, dtype=p[col].dtype)])
    e = mol.bonded["exclusions"]
    e["ai"] = np.concatenate([e["ai"], [h[10], h[10]]])
    e["aj"] = np.concatenate([e["aj"], [o[11], o[30]]])
    x = positions()

    # without the term: plain LJ, the pairs as written
    plain = paramfile.MegoParameters(params.molecules, params.nonbonded, dict(params.attrs), params.cmap)
    del plain.attrs["hbond"]
    e_plain = energies(builder.create_system_from_params(plain, BOX), x)
    e_zero = energies(builder.create_system_from_params(with_settings(params, 0.0, 0.0), BOX), x)
    p6, p12 = params.nonbonded.prior_tables()
    nb = np.asarray(mol.atoms["nb_type"], int)
    i, j, c6, c12 = far_pair
    r = np.linalg.norm(x[i] - x[j])
    handed_over = (p12[nb[i], nb[j]] / r**12 - p6[nb[i], nb[j]] / r**6) - (c12 / r**12 - c6 / r**6)
    total_plain = e_plain["LJ (SR)"] + e_plain["LJ-14"]
    total_zero = e_zero["LJ (SR)"] + e_zero["LJ-14"] + e_zero["H-bond"]
    assert total_zero - total_plain == pytest.approx(handed_over, abs=1e-3)


# ---------------------------------------------------------------------------
# Production: H-O and N-O are not learned beyond the local range (option b)
# ---------------------------------------------------------------------------
def _toy_prior():
    from multiego.nonbonded import TypePrior

    names = ["H", "O"]
    z = np.zeros((2, 2))
    prior = TypePrior(names, np.zeros((2, 2), bool), z, z, z, z)
    table = np.full((5, 5), np.nan)
    for a, b in hb.REPULSIVE_BACKBONE_PAIRS:
        ia, ib = hb.ROLE_INDEX[a], hb.ROLE_INDEX[b]
        table[ia, ib] = table[ib, ia] = 1e-7 * (ia + ib)
    prior.backbone_c12 = table
    # residue r: N_r, H_r, C_r, O_r, CA_r (CA has no role); N_1 is a charged terminus (role 0)
    prior.sb_role = {}
    for r in range(1, 11):
        for name in ("N", "H", "C", "O"):
            prior.sb_role[f"{name}_X_{r}"] = hb.ROLE_INDEX[name]
        prior.sb_role[f"CA_X_{r}"] = 0
    prior.sb_role["N_X_1"] = 0
    return prior


def test_frozen_mask_and_companions():
    import pandas as pd

    from multiego import lj

    prior = _toy_prior()
    resnum = {k: int(k.split("_")[-1]) for k in prior.sb_role}
    rows = [
        ("H_X_2", "O_X_6", True),  # H-O, 4 apart: frozen
        ("H_X_2", "O_X_3", True),  # H-O, local: learned
        ("N_X_5", "O_X_2", True),  # N-O, 3 apart: frozen
        ("N_X_5", "O_X_4", True),  # N-O, local: learned
        ("H_X_2", "O_X_3", False),  # H-O, other chain: frozen
        ("N_X_1", "O_X_9", True),  # charged N-terminus: not a backbone role, learned
        ("C_X_2", "O_X_7", True),  # C-O: learned
        ("C_X_2", "N_X_8", True),  # C-N: learned
        ("CA_X_2", "O_X_9", False),  # not polar backbone: learned
    ]
    df = pd.DataFrame(rows, columns=["ai", "aj", "same_chain"])
    mask = lj._hbond_frozen_mask(df, prior, resnum, 3)
    assert mask.tolist() == [True, False, True, False, True, False, False, False, False]

    kept = df.loc[~mask].assign(
        learned=1, source="intramat", probability=0.5, rc_probability=0.1, epsilon=0.3, sigma=0.3
    )
    comp = lj._hbond_prior_companions(kept, prior)
    # one intermolecular prior row per learned local H-O / N-O contact
    assert sorted(zip(comp["ai"], comp["aj"])) == [("H_X_2", "O_X_3"), ("N_X_5", "O_X_4")]
    assert not comp["same_chain"].any() and (comp["learned"] == 0).all()
    ho = comp.set_index("ai").loc["H_X_2"]
    c12 = prior.backbone_c12[hb.ROLE_INDEX["H"], hb.ROLE_INDEX["O"]]
    assert ho["epsilon"] == pytest.approx(-c12)
    assert ho["sigma"] == pytest.approx(c12 ** (1 / 12) / 2 ** (1 / 6))
