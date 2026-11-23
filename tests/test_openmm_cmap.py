"""
Backbone CMAP support of the OpenMM backend (``multiego.openmm_backend.cmap``).
"""

import os
import sys

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from multiego.openmm_backend import FORCE_GROUPS, create_system, read_topology  # noqa: E402
from multiego.openmm_backend import cmap as C  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cmap_builder"))
from cmap_math import BicubicMap, backbone_positions, dihedral  # noqa: E402

REFERENCE = openmm.Platform.getPlatformByName("Reference")
BOX = np.diag([8.0, 8.0, 8.0])

HEADER = """
[ defaults ]
  1  1  no  1.0  1.0

[ atomtypes ]
 X  6 12.011 0.0 A 0.0 0.0
"""


def peptide(residues, caps=True, dihedrals=""):
    """
    Topology text of a backbone-only chain: (ACE) res... (NME). Each amino acid
    has N, CA, C; ACE has CA, C; NME has N, CA. No nonbonded interactions.
    """
    atoms, bonds = [], []

    def add(name, resnum, resname):
        atoms.append((len(atoms) + 1, resnum, resname, name))
        return len(atoms)

    prev_c = None
    resnum = 1
    if caps:
        ca = add("CA", resnum, "ACE")
        prev_c = add("C", resnum, "ACE")
        bonds.append((ca, prev_c))
        resnum += 1
    for name in residues:
        n = add("N", resnum, name)
        ca = add("CA", resnum, name)
        c = add("C", resnum, name)
        bonds += [(n, ca), (ca, c)] + ([(prev_c, n)] if prev_c else [])
        prev_c = c
        resnum += 1
    if caps:
        n = add("N", resnum, "NME")
        ca = add("CA", resnum, "NME")
        bonds += [(prev_c, n), (n, ca)]
    lines = ["[ moleculetype ]", " PEP 3", "[ atoms ]"]
    lines += [f" {i} X {r} {rn} {an} {i}" for i, r, rn, an in atoms]
    lines += ["[ bonds ]"] + [f" {i} {j} 1 0.15 1000.0" for i, j in bonds]
    return "\n".join(lines) + "\n" + dihedrals


def build(tmp_path, residues, cmap_dir, copies=1, caps=True, dihedrals=""):
    text = HEADER + peptide(residues, caps, dihedrals) + f"\n[ system ]\nx\n[ molecules ]\nPEP {copies}\n"
    path = tmp_path / "topol.top"
    path.write_text(text)
    top = read_topology(str(path))
    return top, create_system(top, BOX, com_interval=0, cmap_dir=cmap_dir)


def random_map(n=24, seed=0):
    rng = np.random.default_rng(seed)
    th = np.radians(C.grid_nodes(n))
    phi, psi = np.meshgrid(th, th)
    return 3 * np.cos(phi) + 2 * np.sin(psi) + 1.5 * np.sin(phi - 2 * psi) + rng.normal(0, 0.7, (n, n))


def cmap_energy(system, positions):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True, groups={FORCE_GROUPS["CMAP"]})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def cmap_force(system):
    return next(f for f in system.getForces() if isinstance(f, openmm.CMAPTorsionForce))


# ----------------------------------------------------------------- file format
@pytest.mark.parametrize(
    "name, expected",
    [
        ("ALA", ("ALA", False)),
        ("DALA", ("ALA", True)),
        ("CYS2", ("CYS", False)),
        ("DCYS2", ("CYS", True)),
        ("HISH", ("HIS", False)),
        ("GLY", ("GLY", False)),
        ("DPRO", ("PRO", True)),
    ],
)
def test_map_residue(name, expected):
    assert C.map_residue(name) == expected


def test_write_read_roundtrip(tmp_path):
    grid = random_map(36)
    C.write_cmap(str(tmp_path / "ALA.cmap"), "ALA", grid, {"temperature": "300 K"})
    meta, back = C.read_cmap(str(tmp_path / "ALA.cmap"))
    assert meta["residue"] == "ALA" and meta["grid"] == "36" and meta["temperature"] == "300 K"
    np.testing.assert_allclose(back, grid, atol=1e-6)


def test_read_rejects_bad_files(tmp_path):
    p = tmp_path / "ALA.cmap"
    p.write_text("# residue: ALA\n# grid: 4\n1 2 3 4\n")
    with pytest.raises(ValueError, match="4x4"):
        C.read_cmap(str(p))
    p.write_text("# grid: 2\n1 2\n3 4\n")
    with pytest.raises(ValueError, match="residue"):
        C.read_cmap(str(p))


def test_mirror():
    grid = random_map(24)
    m = C.mirror(grid)
    np.testing.assert_array_equal(C.mirror(m), grid)
    nodes = C.grid_nodes(24)
    for i, j in [(0, 0), (3, 7), (12, 5), (23, 1)]:
        ii = int(np.argmin(np.abs(((nodes + nodes[i]) + 180) % 360 - 180)))
        jj = int(np.argmin(np.abs(((nodes + nodes[j]) + 180) % 360 - 180)))
        assert m[i, j] == grid[ii, jj]  # U_D(phi, psi) = U_L(-phi, -psi)


# --------------------------------------------------------------------- builder
def test_backbone_quintuplets(tmp_path):
    top = read_topology(str(_write(tmp_path, ["ALA", "GLY", "VAL"], caps=False)))
    quints = C.backbone_quintuplets(top.moleculetypes["PEP"])
    assert [(r, n) for r, n, _ in quints] == [("GLY", 2)]  # termini have no full pair
    top = read_topology(str(_write(tmp_path, ["ALA"], caps=True)))
    ((res, _, atoms),) = C.backbone_quintuplets(top.moleculetypes["PEP"])
    names = [top.moleculetypes["PEP"].atoms[a].name for a in atoms]
    assert res == "ALA" and names == ["C", "N", "CA", "C", "N"]


def _write(tmp_path, residues, caps):
    path = tmp_path / "q.top"
    path.write_text(HEADER + peptide(residues, caps) + "\n[ system ]\nx\n[ molecules ]\nPEP 1\n")
    return path


@pytest.mark.parametrize("n", [24, 72])
def test_energy_convention_against_grid(tmp_path, n):
    """Asymmetric map: at the nodes OpenMM must return the file value; elsewhere the bicubic value."""
    C.write_cmap(str(tmp_path / "ALA.cmap"), "ALA", random_map(n, seed=n))
    _, grid = C.read_cmap(str(tmp_path / "ALA.cmap"))  # as stored (6 decimals)
    _, system = build(tmp_path, ["ALA"], str(tmp_path))
    bm = BicubicMap(grid)
    nodes = C.grid_nodes(n)
    rng = np.random.default_rng(3)
    for i, j in rng.integers(0, n, (12, 2)):
        x = _chain_positions(nodes[j], nodes[i])
        assert cmap_energy(system, x) == pytest.approx(grid[i, j], abs=1e-9)
    for phi, psi in rng.uniform(-180, 180, (12, 2)):
        assert cmap_energy(system, _chain_positions(phi, psi)) == pytest.approx(float(bm.energy(phi, psi)), abs=1e-9)


def _chain_positions(phi, psi):
    """ACE CA, ACE C, ALA N, CA, C, NME N, NME CA with the requested phi/psi."""
    core = backbone_positions(phi, psi)  # C(-1), N, CA, C, N(+1)
    assert abs((dihedral(*core[:4]) - phi + 180) % 360 - 180) < 1e-8
    ace_ca = core[0] + np.array([-0.1, 0.1, 0.05])
    nme_ca = core[4] + np.array([0.1, -0.05, 0.1])
    return np.vstack([ace_ca, core, nme_ca]) + 4.0


def test_d_residue_uses_mirrored_map(tmp_path):
    grid = random_map(24, seed=5)
    C.write_cmap(str(tmp_path / "ALA.cmap"), "ALA", grid)
    _, sys_l = build(tmp_path, ["ALA"], str(tmp_path))
    _, sys_d = build(tmp_path, ["DALA"], str(tmp_path))
    for phi, psi in [(-63.0, -41.0), (-75.0, 145.0), (57.0, 47.0)]:
        e_d = cmap_energy(sys_d, _chain_positions(phi, psi))
        e_l = cmap_energy(sys_l, _chain_positions(-phi, -psi))
        assert e_d == pytest.approx(e_l, abs=1e-9)


def test_only_used_maps_and_copies(tmp_path):
    for name in ("ALA", "GLY", "VAL"):
        C.write_cmap(str(tmp_path / f"{name}.cmap"), name, random_map(24, seed=len(name)))
    _, system = build(tmp_path, ["ALA", "GLY", "ALA"], str(tmp_path), copies=3)
    f = cmap_force(system)
    assert f.getNumMaps() == 2  # ALA and GLY; VAL is in the directory but unused
    assert f.getNumTorsions() == 9


def test_missing_map(tmp_path):
    C.write_cmap(str(tmp_path / "ALA.cmap"), "ALA", random_map(24))
    with pytest.raises(ValueError, match="GLY"):
        build(tmp_path, ["ALA", "GLY"], str(tmp_path))


def test_phi_psi_proper_with_cmap_is_rejected(tmp_path):
    C.write_cmap(str(tmp_path / "ALA.cmap"), "ALA", random_map(24))
    # atoms: 1 ACE CA, 2 ACE C, 3 N, 4 CA, 5 C, 6 NME N, 7 NME CA; phi = 2-3-4-5
    with pytest.raises(ValueError, match="--backbone cmap"):
        build(tmp_path, ["ALA"], str(tmp_path), dihedrals="[ dihedrals ]\n 2 3 4 5 1 0.0 1.0 1\n")


def test_no_cmap_without_dir(tmp_path):
    _, system = build(tmp_path, ["ALA"], None)
    assert not any(isinstance(f, openmm.CMAPTorsionForce) for f in system.getForces())
