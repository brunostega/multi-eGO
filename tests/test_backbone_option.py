"""
Tests for ``--backbone``: with ``cmap`` every backbone phi/psi proper dihedral
is removed and nothing else changes.
"""

import os
import types
import warnings

import pytest

from multiego import topology as _topology

TESTS = os.path.dirname(os.path.abspath(__file__))


def _dih(*atoms):
    """Mock parmed dihedral from (name, residue index) pairs."""
    a = [types.SimpleNamespace(name=n, residue=types.SimpleNamespace(idx=r)) for n, r in atoms]
    return types.SimpleNamespace(atom1=a[0], atom2=a[1], atom3=a[2], atom4=a[3])


@pytest.mark.parametrize(
    "atoms, expected",
    [
        ((("C", 0), ("N", 1), ("CA", 1), ("C", 1)), True),  # phi
        ((("N", 1), ("CA", 1), ("C", 1), ("N", 2)), True),  # psi
        ((("C", 1), ("CA", 1), ("N", 1), ("C", 0)), True),  # phi, reversed
        ((("N", 2), ("C", 1), ("CA", 1), ("N", 1)), True),  # psi, reversed
        ((("CA", 0), ("C", 0), ("N", 1), ("CA", 1)), False),  # omega
        ((("N", 1), ("CA", 1), ("CB", 1), ("CG", 1)), False),  # chi1
        ((("C", 1), ("N", 1), ("CA", 1), ("C", 1)), False),  # phi names, no residue boundary
        ((("N", 1), ("CA", 1), ("C", 1), ("N", 1)), False),  # psi names, no residue boundary
    ],
)
def test_is_backbone_phi_psi(atoms, expected):
    assert _topology.is_backbone_phi_psi(_dih(*atoms)) is expected


def _load(system):
    parmed = pytest.importorskip("parmed")
    from multiego._parmed_compat import gmxlib_in_topo_dir

    path = os.path.join(TESTS, "test_inputs", system, "topol.top")
    with gmxlib_in_topo_dir(path), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parmed.load_file(path, {"DISULFIDE": 1})


def _key(df):
    return sorted(map(tuple, df[["ai", "aj", "ak", "al", "phase", "phi_k", "per"]].astype(float).values.tolist()))


# GB1 has 56 residues: phi on residues 2-56, psi on residues 1-55; the termini lose theirs too.
@pytest.mark.parametrize("system, n_phi, n_psi", [("aladip", 1, 1), ("gpref", 55, 55)])
def test_backbone_cmap_removes_only_phi_psi(system, n_phi, n_psi):
    from multiego.ensemble_data import _build_bonded_interactions

    top = _load(system)
    gromos, _, _ = _build_bonded_interactions(top, backbone="gromos")
    cmap, _, _ = _build_bonded_interactions(top, backbone="cmap")
    for molecule in gromos:
        for term in ("bonds", "angles", "impropers", "pairs"):
            assert gromos[molecule][term].equals(cmap[molecule][term]), term

        mol = top.molecules[molecule][0]
        atoms = mol.atoms
        removed = set(_key(gromos[molecule]["dihedrals"])) - set(_key(cmap[molecule]["dihedrals"]))
        kept = _key(cmap[molecule]["dihedrals"])
        assert set(kept) <= set(_key(gromos[molecule]["dihedrals"]))
        assert len(kept) + len(removed) == len(gromos[molecule]["dihedrals"])

        def names(row):
            return tuple(atoms[int(i) - 1].name for i in row[:4])

        assert all(names(r) in (_topology.PHI_NAMES, _topology.PSI_NAMES) for r in removed)
        assert not any(
            names(r) in (_topology.PHI_NAMES, _topology.PSI_NAMES)
            and len({atoms[int(i) - 1].residue.idx for i in r[:4]}) == 2
            for r in kept
        )
        # every phi and every psi is gone, termini included
        phi_res = {atoms[int(r[2]) - 1].residue.idx for r in removed if names(r) == _topology.PHI_NAMES}
        psi_res = {atoms[int(r[1]) - 1].residue.idx for r in removed if names(r) == _topology.PSI_NAMES}
        assert (len(phi_res), len(psi_res)) == (n_phi, n_psi)


def test_aladip_keeps_only_omega():
    from multiego.ensemble_data import _build_bonded_interactions

    top = _load("aladip")
    cmap, _, _ = _build_bonded_interactions(top, backbone="cmap")
    (dihedrals,) = [v["dihedrals"] for v in cmap.values()]
    atoms = list(top.molecules.values())[0][0].atoms
    got = [tuple(atoms[int(i) - 1].name for i in row) for row in dihedrals[["ai", "aj", "ak", "al"]].values]
    assert got == [("CA", "C", "N", "CA"), ("CA", "C", "N", "CA")]


def test_backbone_unknown_value():
    from multiego.ensemble_data import _build_bonded_interactions

    with pytest.raises(ValueError, match="backbone"):
        _build_bonded_interactions(_load("aladip"), backbone="charmm")
