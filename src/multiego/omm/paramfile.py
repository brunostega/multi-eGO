"""
The multi-eGO parameter file (``mego_params.h5``).

This file is the engine-independent output of the force-field *generation*
step (``multiego.py --egos mg|production``).  It replaces the GROMACS
``ffnonbonded.itp`` + ``topol_mego.top`` pair as the input of OpenMM
simulations; the System is assembled from it at run time by
:mod:`multiego.omm.builder`.  The separation between generation and simulation
is therefore the same as with GROMACS.

All values are stored in OpenMM units: nm, rad, kJ/mol, amu.

Layout (format version 1)
-------------------------
::

    /                         attrs: format_version, mego_version, egos, system,
                                     created, arguments (JSON of the CLI args),
                                     nrexcl
    /molecules/<name>/        one group per molecule type, attrs: order (int)
        atoms/                one entry per atom, in topology order
            sb_type           str    structure-based type, e.g. "CA_GB1_12"
            type              str    chemical atom type, e.g. "CH1"
            name, resname     str
            resnum            int
            atomic_number     int
            mass              float  amu
            nb_type           int    index into /nonbonded/sb_types
        bonds                 ai, aj, r0, k                 harmonic, E = k/2 (r-r0)^2
        angles                ai, aj, ak, theta0, k         harmonic, E = k/2 (t-t0)^2
        proper_dihedrals      ai, aj, ak, al, periodicity, phase, k
                                                            E = k (1 + cos(n phi - phase))
        improper_dihedrals    ai, aj, ak, al, theta0, k     harmonic, E = k/2 (xi-xi0)^2
        pairs                 ai, aj, c6, c12               E = c12/r^12 - c6/r^6 (no cut-off)
        exclusions            ai, aj                        explicit exclusions
    /nonbonded/
        sb_types              str   all sb_types of all molecule types; the
                                    position in this list is the table index
        default_c6            float per-type c6  (combination rule, see below)
        default_c12           float per-type c12
        pair_types            int   (M, 2) explicit type pairs (i <= j)
        pair_c6, pair_c12     float (M,) parameters of the explicit pairs

Within a molecule, atom indices (ai, aj, ...) are 0-based positions in the
``atoms`` table of that molecule.

Nonbonded semantics
-------------------
These follow the GROMACS ``[ defaults ]`` written by multi-eGO
(``nbfunc 1, comb-rule 1, gen-pairs no``), so the two backends are
equivalent:

* the LJ parameters of a pair of types (i, j) are given by the explicit list
  ``pair_types / pair_c6 / pair_c12`` (the GROMACS ``[ nonbond_params ]``);
* every type pair *not* in the list uses the combination rule 1:
  ``c6 = sqrt(c6_i c6_j)``, ``c12 = sqrt(c12_i c12_j)`` with the per-type
  ``default_c6/default_c12`` (the GROMACS ``[ atomtypes ]``);
* atoms within ``nrexcl`` bonds of each other, the explicit ``exclusions``
  and the ``pairs`` of each molecule are excluded from the nonbonded term.
"""

import json
import time
from dataclasses import dataclass, field

import h5py
import numpy as np

FORMAT_VERSION = 1
DEFAULT_FILENAME = "mego_params.h5"

# Columns of each bonded table: (name, dtype).  Atom indices come first.
BONDED_COLUMNS = {
    "bonds": [("ai", int), ("aj", int), ("r0", float), ("k", float)],
    "angles": [("ai", int), ("aj", int), ("ak", int), ("theta0", float), ("k", float)],
    "proper_dihedrals": [
        ("ai", int),
        ("aj", int),
        ("ak", int),
        ("al", int),
        ("periodicity", int),
        ("phase", float),
        ("k", float),
    ],
    "improper_dihedrals": [("ai", int), ("aj", int), ("ak", int), ("al", int), ("theta0", float), ("k", float)],
    "pairs": [("ai", int), ("aj", int), ("c6", float), ("c12", float)],
    "exclusions": [("ai", int), ("aj", int)],
}

ATOM_STR_COLUMNS = ("sb_type", "type", "name", "resname")
ATOM_NUM_COLUMNS = (("resnum", int), ("atomic_number", int), ("mass", float), ("nb_type", int))


# ---------------------------------------------------------------------------
# In-memory representation
# ---------------------------------------------------------------------------


@dataclass
class MoleculeParameters:
    """Parameters of one molecule type.  Every table is a dict of numpy arrays."""

    name: str
    atoms: dict
    bonded: dict  # term name -> {column: array}, keys as in BONDED_COLUMNS

    @property
    def n_atoms(self):
        return len(self.atoms["sb_type"])


@dataclass
class NonbondedParameters:
    """Type-based LJ parameters (see 'Nonbonded semantics' in the module docstring)."""

    sb_types: np.ndarray
    default_c6: np.ndarray
    default_c12: np.ndarray
    pair_types: np.ndarray  # (M, 2) int
    pair_c6: np.ndarray
    pair_c12: np.ndarray

    @property
    def n_types(self):
        return len(self.sb_types)

    def dense_tables(self):
        """
        Return the full (n_types x n_types) c6 and c12 tables.

        Unlisted pairs follow combination rule 1, listed pairs overwrite them
        (symmetrically).  This is the only place where an N x N object is
        created: it is what the simulation engine needs.
        """
        c6 = np.sqrt(np.outer(self.default_c6, self.default_c6))
        c12 = np.sqrt(np.outer(self.default_c12, self.default_c12))
        i, j = self.pair_types[:, 0], self.pair_types[:, 1]
        c6[i, j] = c6[j, i] = self.pair_c6
        c12[i, j] = c12[j, i] = self.pair_c12
        return c6, c12


@dataclass
class MegoParameters:
    """Content of a multi-eGO parameter file."""

    molecules: dict  # name -> MoleculeParameters, in topology order
    nonbonded: NonbondedParameters
    attrs: dict = field(default_factory=dict)

    @property
    def nrexcl(self):
        return int(self.attrs.get("nrexcl", 3))


# ---------------------------------------------------------------------------
# Construction from the multi-eGO generation data structures
# ---------------------------------------------------------------------------


def _bonded_tables(bonded_interactions, pairs_14):
    """
    Convert the GROMACS-style bonded DataFrames of one molecule into the
    tables of this format (0-based indices, OpenMM units).

    Parameters
    ----------
    bonded_interactions : dict of pd.DataFrame
        ``meGO_ensemble.meGO_bonded_interactions[molecule]``: GROMACS columns
        and units (atom numbers 1-based, angles in degrees).
    pairs_14 : pd.DataFrame
        ``meGO_LJ_14[molecule]``: the [ pairs ] with their c6/c12.  As in the
        GROMACS writer, these pairs also become the explicit exclusions.
    """

    def idx(df, col):
        return df[col].astype(int).to_numpy() - 1  # GROMACS numbers are 1-based

    tables = {}
    b = bonded_interactions["bonds"]
    tables["bonds"] = {
        "ai": idx(b, "ai"),
        "aj": idx(b, "aj"),
        "r0": b["req"].astype(float).to_numpy(),
        "k": b["k"].astype(float).to_numpy(),
    }

    a = bonded_interactions["angles"]
    tables["angles"] = {
        "ai": idx(a, "ai"),
        "aj": idx(a, "aj"),
        "ak": idx(a, "ak"),
        "theta0": np.deg2rad(a["theteq"].astype(float).to_numpy()),
        "k": a["k"].astype(float).to_numpy(),
    }

    d = bonded_interactions["dihedrals"]
    tables["proper_dihedrals"] = {
        "ai": idx(d, "ai"),
        "aj": idx(d, "aj"),
        "ak": idx(d, "ak"),
        "al": idx(d, "al"),
        "periodicity": d["per"].astype(int).to_numpy(),
        "phase": np.deg2rad(d["phase"].astype(float).to_numpy()),
        "k": d["phi_k"].astype(float).to_numpy(),
    }

    im = bonded_interactions["impropers"]
    tables["improper_dihedrals"] = {
        "ai": idx(im, "ai"),
        "aj": idx(im, "aj"),
        "ak": idx(im, "ak"),
        "al": idx(im, "al"),
        "theta0": np.deg2rad(im["psi_eq"].astype(float).to_numpy()),
        "k": im["psi_k"].astype(float).to_numpy(),
    }

    p = pairs_14
    if p is None or p.empty:
        # No parametrised pairs: like the GROMACS writer, no pairs and no
        # explicit exclusions are written for this molecule.
        tables["pairs"] = {"ai": [], "aj": [], "c6": [], "c12": []}
    else:
        tables["pairs"] = {
            "ai": idx(p, "ai"),
            "aj": idx(p, "aj"),
            "c6": p["c6"].astype(float).to_numpy(),
            "c12": p["c12"].astype(float).to_numpy(),
        }
    tables["exclusions"] = {"ai": tables["pairs"]["ai"].copy(), "aj": tables["pairs"]["aj"].copy()}

    # Every table must have every column, also when empty.
    for term, columns in BONDED_COLUMNS.items():
        for col, dtype in columns:
            tables[term][col] = np.asarray(tables[term][col], dtype=dtype)
    return tables


def from_generation(meGO_ensemble, meGO_LJ, meGO_LJ_14, parameters, attrs=None):
    """
    Build :class:`MegoParameters` from the data structures of the generation
    pipeline, i.e. from exactly what the GROMACS writer receives.

    Parameters
    ----------
    meGO_ensemble : MeGOEnsemble
    meGO_LJ : pd.DataFrame
        Nonbonded LJ pairs after ``fileio.sort_LJ`` (columns ai, aj = sb_types,
        c6, c12): the GROMACS ``[ nonbond_params ]``.
    meGO_LJ_14 : dict of pd.DataFrame
        Pairs per molecule: the GROMACS ``[ pairs ]``.
    parameters : argparse.Namespace
        CLI arguments; ``egos`` selects the per-type default c6/c12 (the mg
        prior uses ``mg_c6/mg_c12``, as in ``fileio.write_nonbonded``).
    attrs : dict, optional
        Extra file attributes (provenance).
    """
    topology = meGO_ensemble.topology_dataframe
    c6_col, c12_col = ("mg_c6", "mg_c12") if parameters.egos == "mg" else ("c6", "c12")

    # Global type table: one type per sb_type, in topology order.
    sb_types = topology["sb_type"].astype(str).to_numpy()
    type_index = {t: i for i, t in enumerate(sb_types)}

    molecules = {}
    for molecule, bonded in meGO_ensemble.meGO_bonded_interactions.items():
        rows = topology.loc[topology["molecule_name"] == molecule].sort_values("number", key=lambda s: s.astype(int))
        atoms = {col: rows[col].astype(str).to_numpy() for col in ATOM_STR_COLUMNS}
        atoms["resnum"] = rows["resnum"].astype(int).to_numpy()
        atoms["atomic_number"] = rows["atomic_number"].astype(int).to_numpy()
        atoms["mass"] = rows["mass"].astype(float).to_numpy()
        atoms["nb_type"] = np.array([type_index[t] for t in atoms["sb_type"]], dtype=int)
        molecules[molecule] = MoleculeParameters(molecule, atoms, _bonded_tables(bonded, meGO_LJ_14.get(molecule)))

    pair_types = np.column_stack(
        [meGO_LJ["ai"].astype(str).map(type_index).to_numpy(), meGO_LJ["aj"].astype(str).map(type_index).to_numpy()]
    ).astype(int)
    pair_types.sort(axis=1)  # i <= j
    # A type pair listed twice would be ambiguous (GROMACS silently keeps the
    # last one): refuse it instead.
    if len(np.unique(pair_types, axis=0)) != len(pair_types):
        raise ValueError("duplicate type pairs in the nonbonded parameters")
    nonbonded = NonbondedParameters(
        sb_types=sb_types,
        default_c6=topology[c6_col].astype(float).to_numpy(),
        default_c12=topology[c12_col].astype(float).to_numpy(),
        pair_types=pair_types,
        pair_c6=meGO_LJ["c6"].astype(float).to_numpy(),
        pair_c12=meGO_LJ["c12"].astype(float).to_numpy(),
    )

    file_attrs = {
        "format_version": FORMAT_VERSION,
        "egos": parameters.egos,
        "system": parameters.system,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": json.dumps({k: v for k, v in vars(parameters).items()}, default=str),
        "nrexcl": 3,  # same as the [ moleculetype ] written by the GROMACS backend
    }
    file_attrs.update(attrs or {})
    return MegoParameters(molecules, nonbonded, file_attrs)


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

_STR = h5py.string_dtype(encoding="utf-8")


def _write_table(group, name, columns, compression="gzip"):
    """Write a dict of equal-length arrays as datasets of a sub-group."""
    sub = group.create_group(name)
    for col, values in columns.items():
        values = np.asarray(values)
        if values.dtype.kind in "OUS":
            sub.create_dataset(col, data=values.astype(object), dtype=_STR)
        else:
            # Compression only pays off (and is only allowed) for non-empty data.
            sub.create_dataset(col, data=values, compression=compression if values.size else None)


def _read_table(group):
    return {col: (ds.asstr()[()] if h5py.check_string_dtype(ds.dtype) else ds[()]) for col, ds in group.items()}


def write(path, params):
    """Write :class:`MegoParameters` to ``path`` (HDF5)."""
    with h5py.File(path, "w") as f:
        for key, value in params.attrs.items():
            f.attrs[key] = value
        mols = f.create_group("molecules")
        for order, (name, mol) in enumerate(params.molecules.items()):
            g = mols.create_group(name)
            g.attrs["order"] = order
            _write_table(g, "atoms", mol.atoms)
            for term, table in mol.bonded.items():
                _write_table(g, term, table)
        nb = params.nonbonded
        _write_table(
            f,
            "nonbonded",
            {
                "sb_types": nb.sb_types,
                "default_c6": nb.default_c6,
                "default_c12": nb.default_c12,
                "pair_c6": nb.pair_c6,
                "pair_c12": nb.pair_c12,
            },
        )
        f["nonbonded"].create_dataset(
            "pair_types", data=nb.pair_types, compression="gzip" if nb.pair_types.size else None
        )


def read(path):
    """Read a parameter file written by :func:`write`."""
    with h5py.File(path, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        if attrs.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: unsupported format version {attrs.get('format_version')} (expected {FORMAT_VERSION})"
            )
        groups = sorted(f["molecules"].items(), key=lambda item: item[1].attrs["order"])
        molecules = {}
        for name, g in groups:
            bonded = {term: _read_table(g[term]) for term in BONDED_COLUMNS}
            molecules[name] = MoleculeParameters(name, _read_table(g["atoms"]), bonded)
        t = _read_table(f["nonbonded"])
        nonbonded = NonbondedParameters(
            sb_types=t["sb_types"],
            default_c6=t["default_c6"],
            default_c12=t["default_c12"],
            pair_types=t["pair_types"].reshape(-1, 2),
            pair_c6=t["pair_c6"],
            pair_c12=t["pair_c12"],
        )
    return MegoParameters(molecules, nonbonded, attrs)
