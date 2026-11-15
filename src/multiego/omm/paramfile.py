"""
The multi-eGO parameter file (``mego_params.h5``).

This file is the engine-independent output of the force-field *generation*
step (``multiego.py --egos mg|production``).  It replaces the GROMACS
``ffnonbonded.itp`` + ``topol_mego.top`` pair as the input of OpenMM
simulations; the System is assembled from it at run time by
:mod:`multiego.omm.builder`.  The separation between generation and simulation
is therefore the same as with GROMACS.

All values are stored in OpenMM units: nm, rad, kJ/mol, amu.

Layout (format version 3)
-------------------------
::

    /                         attrs: format_version, mego_version, egos, system,
                                     created, arguments (JSON of the CLI args),
                                     nrexcl, and -- only when the geometric
                                     hydrogen-bond term is enabled -- hbond
                                     (JSON of omm.hbond.HBondSettings)
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
    /cmap/                    optional backbone CMAP corrections
        maps/<resname>        float (S, S) energy in kJ/mol, [phi, psi],
                              periodic grid of 360/S degrees (see multiego.cmap)
        residues/molecule     str   one entry per CMAP-corrected residue:
        residues/index        int   molecule name and the atom indices of
        residues/c0,n,ca,c,n1 int   phi = c0-n-ca-c, psi = n-ca-c-n1
    /nonbonded/
        sb_types              str   all sb_types of all molecule types; the
                                    position in this list is the table index
        chem_type             int   chemical type of each sb_type (index into
                                    prior/types)
        default_c6            float per-type c6  (combination rule, see below)
        default_c12           float per-type c12
        prior/types           str   chemical atom types
        prior/has_rule        bool  (T, T) whether the mg prior defines the pair
        prior/c6, prior/c12   float (T, T) mg prior parameters
        overrides/i, j        int   (M,) atom-pair overrides (learned contacts)
        overrides/c6, c12     float (M,)

Within a molecule, atom indices (ai, aj, ...) are 0-based positions in the
``atoms`` table of that molecule.

Nonbonded semantics
-------------------
These follow the GROMACS ``[ defaults ]`` written by multi-eGO
(``nbfunc 1, comb-rule 1, gen-pairs no``), so the two backends are
equivalent:

* a pair of sb_types (i, j) takes, in order of precedence, the parameters of
  an override, or of the mg prior of their chemical types if
  ``prior/has_rule`` is set, or else the combination rule 1:
  ``c6 = sqrt(c6_i c6_j)``, ``c12 = sqrt(c12_i c12_j)`` with the per-type
  ``default_c6/default_c12`` (the GROMACS ``[ atomtypes ]``).  The first two
  cases are the pairs listed in the GROMACS ``[ nonbond_params ]``; the
  representation is compact (see ``multiego.nonbonded``): the prior is stored
  once per pair of *chemical* types and only the learned pairs per atom pair;
* atoms within ``nrexcl`` bonds of each other, the explicit ``exclusions``
  and the ``pairs`` of each molecule are excluded from the nonbonded term.
"""

import json
import time
from dataclasses import dataclass, field

import h5py
import numpy as np

from multiego.nonbonded import TypePrior, dense_tables

FORMAT_VERSION = 3
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
    chem_type: np.ndarray
    default_c6: np.ndarray
    default_c12: np.ndarray
    prior: TypePrior
    overrides: dict  # i, j, c6, c12 arrays

    @property
    def n_types(self):
        return len(self.sb_types)

    def dense_tables(self):
        """
        Return the full (n_types x n_types) c6 and c12 tables.

        This is the only place where an N x N object is created: it is what
        the simulation engine needs.  The expansion is shared with the
        generation code (``multiego.nonbonded.dense_tables``).
        """
        return dense_tables(self.chem_type, self.default_c6, self.default_c12, self.prior, self.overrides)


@dataclass
class CmapParameters:
    """
    Backbone CMAP corrections: the maps themselves, and the list of residues
    they apply to (with the five atoms defining phi and psi).
    """

    maps: dict  # residue name -> (S, S) array, kJ/mol, [phi, psi]
    residues: dict  # arrays: molecule, index (into maps order), c0, n, ca, c, n1

    def map_order(self):
        """Residue names in a fixed order; the index stored per residue."""
        return sorted(self.maps)


@dataclass
class MegoParameters:
    """Content of a multi-eGO parameter file."""

    molecules: dict  # name -> MoleculeParameters, in topology order
    nonbonded: NonbondedParameters
    attrs: dict = field(default_factory=dict)
    cmap: object = None  # CmapParameters, or None when no CMAP is used

    @property
    def nrexcl(self):
        return int(self.attrs.get("nrexcl", 3))

    @property
    def hbond(self):
        """HBondSettings if the hydrogen-bond term is enabled, else None."""
        if "hbond" not in self.attrs:
            return None
        from multiego.omm.hbond import HBondSettings

        return HBondSettings.from_json(self.attrs["hbond"])


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


def cmap_from_maps(molecules, maps):
    """
    Build :class:`CmapParameters` by locating, in every molecule, the residues
    whose name has a CMAP and that have a complete phi/psi backbone
    (C of the previous residue and N of the next one must exist, so the first
    and last residues of a chain are skipped -- as are the Ace/Nme caps).
    """
    # Keep only the maps of residue types present in the system: OpenMM's CPU
    # and Reference platforms pay for every map they are given at every step
    # (with all 20 maps, the CMAP of a 15-residue peptide cost 30x the rest of
    # the force field).
    present = {str(n) for mol in molecules.values() for n in mol.atoms["resname"]}
    maps = {name: grid for name, grid in maps.items() if name in present}
    order = {name: i for i, name in enumerate(sorted(maps))}
    columns = {k: [] for k in ("molecule", "index", "c0", "n", "ca", "c", "n1")}
    for name, mol in molecules.items():
        atoms = {(int(r), str(a)): k for k, (r, a) in enumerate(zip(mol.atoms["resnum"], mol.atoms["name"]))}
        for resnum, resname in sorted({(int(r), str(n)) for r, n in zip(mol.atoms["resnum"], mol.atoms["resname"])}):
            if resname not in order:
                continue
            needed = [(resnum - 1, "C"), (resnum, "N"), (resnum, "CA"), (resnum, "C"), (resnum + 1, "N")]
            if any(key not in atoms for key in needed):
                continue  # chain terminus: phi or psi undefined
            columns["molecule"].append(name)
            columns["index"].append(order[resname])
            for column, key in zip(("c0", "n", "ca", "c", "n1"), needed):
                columns[column].append(atoms[key])
    arrays = {k: np.array(v, dtype=object if k == "molecule" else int) for k, v in columns.items()}
    return CmapParameters(maps, arrays)


def from_generation(meGO_ensemble, nonbonded_model, meGO_LJ_14, parameters, attrs=None, cmap_maps=None):
    """
    Build :class:`MegoParameters` from the data structures of the generation
    pipeline, i.e. from exactly what the GROMACS writer receives.

    Parameters
    ----------
    meGO_ensemble : MeGOEnsemble
    nonbonded_model : multiego.nonbonded.NonbondedModel
        Nonbonded parameters: per-atom defaults, mg prior, learned overrides.
    meGO_LJ_14 : dict of pd.DataFrame
        Pairs per molecule: the GROMACS ``[ pairs ]``.
    parameters : argparse.Namespace
        CLI arguments (stored as provenance).
    attrs : dict, optional
        Extra file attributes (provenance).
    """
    topology = meGO_ensemble.topology_dataframe

    # Global type table: one type per sb_type, in topology order (the same
    # order as nonbonded_model.sb_type).
    sb_types = nonbonded_model.sb_type
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

    ov = nonbonded_model.overrides
    nonbonded = NonbondedParameters(
        sb_types=sb_types,
        chem_type=nonbonded_model.chem,
        default_c6=nonbonded_model.default_c6,
        default_c12=nonbonded_model.default_c12,
        prior=nonbonded_model.prior,
        overrides={k: ov[k].to_numpy() for k in ("i", "j", "c6", "c12")},
    )

    file_attrs = {
        "format_version": FORMAT_VERSION,
        "egos": parameters.egos,
        "system": parameters.system,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": json.dumps({k: v for k, v in vars(parameters).items()}, default=str),
        "nrexcl": 3,  # same as the [ moleculetype ] written by the GROMACS backend
    }
    if getattr(parameters, "hbond", False):
        from multiego.omm.hbond import HBondSettings

        # Only the options the user actually set override the defaults of
        # HBondSettings (which come from the measured hydrogen-bond geometry).
        overrides = {
            "epsilon": parameters.hbond_epsilon,  # None keeps "take it from the LJ pair"
            "r0": parameters.hbond_r0,
            "width": parameters.hbond_width,
            "donor_theta0": parameters.hbond_donor_theta0,
            "donor_width": parameters.hbond_donor_width,
            "acceptor_theta0": parameters.hbond_acceptor_theta0,
            "acceptor_width": parameters.hbond_acceptor_width,
            "cutoff": parameters.hbond_cutoff,
            "epsilon_nonlocal": getattr(parameters, "hbond_epsilon_nonlocal", None),
            "nonlocal_separation": getattr(parameters, "hbond_nonlocal_separation", None),
        }
        settings = {k: v for k, v in overrides.items() if v is not None or k == "epsilon"}
        file_attrs["hbond"] = HBondSettings(**settings).to_json()
    file_attrs.update(attrs or {})
    cmap = cmap_from_maps(molecules, cmap_maps) if cmap_maps else None
    return MegoParameters(molecules, nonbonded, file_attrs, cmap)


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
                "chem_type": nb.chem_type,
                "default_c6": nb.default_c6,
                "default_c12": nb.default_c12,
            },
        )
        p = nb.prior
        _write_table(f["nonbonded"], "prior", {"types": p.types})
        for name in ("has_rule", "c6", "c12", "epsilon", "sigma"):
            f["nonbonded/prior"].create_dataset(name, data=getattr(p, name))
        _write_table(f["nonbonded"], "overrides", nb.overrides)
        if params.cmap is not None:
            g = f.create_group("cmap")
            maps = g.create_group("maps")
            for name in params.cmap.map_order():
                maps.create_dataset(name, data=params.cmap.maps[name])
            _write_table(g, "residues", params.cmap.residues)


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
        g = f["nonbonded"]
        prior_group = g["prior"]
        prior = TypePrior(
            types=prior_group["types"].asstr()[()],
            **{name: prior_group[name][()] for name in ("has_rule", "c6", "c12", "epsilon", "sigma")},
        )
        nonbonded = NonbondedParameters(
            sb_types=g["sb_types"].asstr()[()],
            chem_type=g["chem_type"][()],
            default_c6=g["default_c6"][()],
            default_c12=g["default_c12"][()],
            prior=prior,
            overrides=_read_table(g["overrides"]),
        )
        cmap = None
        if "cmap" in f:
            maps = {name: ds[()] for name, ds in f["cmap/maps"].items()}
            cmap = CmapParameters(maps, _read_table(f["cmap/residues"]))
    return MegoParameters(molecules, nonbonded, attrs, cmap)
