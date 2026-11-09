"""
Compact representation of the multi-eGO nonbonded (LJ) parameters.

Why this module exists
----------------------
multi-eGO gives every atom its own type (its *sb_type*), so the nonbonded
parameters are formally an N x N table (N = number of atoms per molecule
type).  The generation pipeline used to build that table explicitly as a
pandas DataFrame with ~20 columns and one row per *ordered* atom pair, which
needs gigabytes of RAM for a 1,500-atom protein.

The table, however, has a lot of structure:

* the molten-globule (mg) prior depends only on the *chemical* atom types of
  the two atoms (about 20-40 types), not on the individual atoms;
* the learned (production) interactions are sparse: they exist only for the
  atom pairs seen in the training data;
* every other pair follows the GROMACS combination rule 1 from per-atom
  defaults (``c6 = sqrt(c6_i c6_j)``, ``c12 = sqrt(c12_i c12_j)``).

This module stores exactly that -- a :class:`TypePrior` matrix over chemical
types plus a sparse table of per-atom-pair *overrides* -- and expands it into
explicit atom pairs only while writing, one row of the table at a time
(:meth:`NonbondedModel.iter_rows`).  Memory then scales with N (one row)
rather than with N^2.

The explicit pairs produced are exactly the GROMACS ``[ nonbond_params ]``
written before this change (same pairs, same values, same order); the
regression tests check this.

Precedence (which parameters a pair of atoms gets)
--------------------------------------------------
1. an override, if the pair has one (overrides are resolved by the
   generation step, see ``lj.resolve_overrides``);
2. otherwise the mg prior of the two chemical types, if a prior rule exists;
3. otherwise the combination rule (the pair is *not* listed explicitly).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns of the explicit-pair rows (the GROMACS [ nonbond_params ] columns,
# the last four being written as a comment).
ROW_COLUMNS = ["ai", "aj", "type", "c6", "c12", "same_chain", "source", "number_ai", "number_aj"]


@dataclass
class TypePrior:
    """
    The mg prior as a matrix over chemical atom types.

    All matrices are (n_types x n_types) and symmetric.  ``has_rule[a, b]``
    tells whether a prior interaction is defined for types a and b; where it
    is False the other matrices are meaningless (the combination rule applies).

    Attributes
    ----------
    types : np.ndarray of str
        Chemical atom type names; the matrix index of a type is its position.
    has_rule, c6, c12, epsilon, sigma : np.ndarray
        ``epsilon < 0`` marks a purely repulsive interaction (c6 = 0,
        c12 = -epsilon), as everywhere in multi-eGO.
    """

    types: np.ndarray
    has_rule: np.ndarray
    c6: np.ndarray
    c12: np.ndarray
    epsilon: np.ndarray
    sigma: np.ndarray
    # sb_type -> chemical type of the system the prior was built for
    # (filled by mg.generate_MG_prior; not stored in the parameter file).
    sb_chem: dict = None

    @property
    def n_types(self):
        return len(self.types)

    def index(self, type_names):
        """Matrix index of each chemical type name."""
        lookup = {t: i for i, t in enumerate(self.types)}
        return np.array([lookup[t] for t in type_names], dtype=int)


class NonbondedModel:
    """
    Nonbonded parameters of a multi-eGO model: per-atom defaults, the
    type-level prior and sparse per-atom-pair overrides.

    Atoms are identified by their position in the topology DataFrame
    (``meGO_ensemble.topology_dataframe``), i.e. one entry per sb_type.

    Parameters
    ----------
    meGO_ensemble : MeGOEnsemble
        Provides the topology (sb_type, chemical type, molecule, atom number).
    prior : TypePrior
        The mg prior (see ``mg.generate_MG_prior``).
    default_c6, default_c12 : array-like
        Per-atom combination-rule parameters (the GROMACS ``[ atomtypes ]``).
    overrides : pd.DataFrame, optional
        One row per atom pair with columns ``ai``, ``aj`` (sb_type names),
        ``c6``, ``c12``, ``epsilon``, ``sigma``, ``same_chain``, ``source``.
        Each unordered pair may appear only once.
    """

    def __init__(self, meGO_ensemble, prior, default_c6, default_c12, overrides=None):
        top = meGO_ensemble.topology_dataframe
        self.prior = prior
        self.sb_type = top["sb_type"].astype(str).to_numpy()
        self.chem = prior.index(top["type"].astype(str))
        self.default_c6 = np.asarray(default_c6, dtype=float)
        self.default_c12 = np.asarray(default_c12, dtype=float)
        self.number = np.array([meGO_ensemble.sbtype_number_dict[t] for t in self.sb_type], dtype=int)

        # Output order of the explicit pairs (same as the former fileio.sort_LJ):
        # by molecule name as written in the molecule dictionary ("1_GB1", ...,
        # compared as strings), then by atom number.
        mol_key = {name.split("_", 1)[1]: name for name in meGO_ensemble.molecules_idx_sbtype_dictionary}
        self.molecule = np.array([mol_key[m] for m in top["molecule_name"].astype(str)])
        self.blocks = {m: np.flatnonzero(self.molecule == m) for m in sorted(set(self.molecule))}
        for m, atoms in self.blocks.items():
            self.blocks[m] = atoms[np.argsort(self.number[atoms], kind="stable")]
        # rank: position of each atom in the global output order
        order = np.concatenate([self.blocks[m] for m in sorted(self.blocks)])
        self.rank = np.empty(len(order), dtype=int)
        self.rank[order] = np.arange(len(order))

        self.overrides = self._index_overrides(overrides)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _index_overrides(self, overrides):
        """
        Convert the overrides to integer atom indices, canonically ordered
        (i before j in the output order), sorted, and checked for duplicates.
        """
        columns = ["i", "j", "c6", "c12", "epsilon", "sigma", "same_chain", "source"]
        if overrides is None or len(overrides) == 0:
            return pd.DataFrame({c: pd.Series(dtype=object if c == "source" else float) for c in columns}).astype(
                {"i": int, "j": int, "same_chain": bool}
            )
        index = {t: k for k, t in enumerate(self.sb_type)}
        i = overrides["ai"].astype(str).map(index).to_numpy()
        j = overrides["aj"].astype(str).map(index).to_numpy()
        swap = self.rank[i] > self.rank[j]
        i, j = np.where(swap, j, i), np.where(swap, i, j)
        df = pd.DataFrame(
            {
                "i": i,
                "j": j,
                "c6": overrides["c6"].to_numpy(float),
                "c12": overrides["c12"].to_numpy(float),
                "epsilon": overrides["epsilon"].to_numpy(float),
                "sigma": overrides["sigma"].to_numpy(float),
                "same_chain": overrides["same_chain"].to_numpy(bool),
                "source": overrides["source"].astype(str).to_numpy(),
            }
        )
        if df.duplicated(["i", "j"]).any():
            raise ValueError("an atom pair has more than one nonbonded override")
        return df.sort_values(["i", "j"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Explicit pairs
    # ------------------------------------------------------------------

    def iter_rows(self):
        """
        Yield the explicit pairs (GROMACS ``[ nonbond_params ]``) in output
        order, as one DataFrame (columns ``ROW_COLUMNS``) per atom i and
        molecule block.  Only one such row of the N x N table is in memory at
        any time.
        """
        prior = self.prior
        by_i = {i: g for i, g in self.overrides.groupby("i", sort=False)}
        molecules = sorted(self.blocks)
        for a, mol_a in enumerate(molecules):
            for mol_b in molecules[a:]:
                atoms_b = self.blocks[mol_b]
                for i in self.blocks[mol_a]:
                    # Within a molecule only j >= i (each unordered pair once, self included).
                    js = atoms_b if mol_a != mol_b else atoms_b[self.rank[atoms_b] >= self.rank[i]]
                    if len(js) == 0:
                        continue
                    ci, cj = self.chem[i], self.chem[js]
                    listed = prior.has_rule[ci, cj].copy()
                    c6 = prior.c6[ci, cj].copy()
                    c12 = prior.c12[ci, cj].copy()
                    same_chain = np.zeros(len(js), dtype=bool)
                    source = np.full(len(js), "mg", dtype=object)

                    ov = by_i.get(i)
                    if ov is not None:
                        pos = {j: k for k, j in enumerate(js)}
                        sel = ov[ov["j"].isin(pos)]
                        k = np.array([pos[j] for j in sel["j"]], dtype=int)
                        listed[k] = True
                        c6[k] = sel["c6"].to_numpy()
                        c12[k] = sel["c12"].to_numpy()
                        same_chain[k] = sel["same_chain"].to_numpy()
                        source[k] = sel["source"].to_numpy()

                    if not listed.any():
                        continue
                    js, c6, c12 = js[listed], c6[listed], c12[listed]
                    yield pd.DataFrame(
                        {
                            "ai": self.sb_type[i],
                            "aj": self.sb_type[js],
                            "type": 1,
                            "c6": c6,
                            "c12": c12,
                            "same_chain": same_chain[listed],
                            "source": source[listed],
                            "number_ai": self.number[i],
                            "number_aj": self.number[js],
                        }
                    )

    def n_explicit_pairs(self):
        """Number of explicit pairs (rows of [ nonbond_params ]), without building them."""
        return sum(len(rows) for rows in self.iter_rows())

    # ------------------------------------------------------------------
    # Dense tables (for the simulation engine)
    # ------------------------------------------------------------------

    def dense_tables(self):
        """
        Full (N x N) c6 and c12 tables, i.e. what the engine evaluates:
        combination rule, overwritten by the prior where a rule exists,
        overwritten by the overrides.
        """
        return dense_tables(self.chem, self.default_c6, self.default_c12, self.prior, self.overrides)


def dense_tables(chem, default_c6, default_c12, prior, overrides):
    """
    Build the (N x N) c6/c12 tables from the compact representation.

    Shared by :class:`NonbondedModel` and the OpenMM parameter-file reader, so
    that the two cannot disagree.  ``overrides`` needs columns i, j, c6, c12.
    """
    c6 = np.sqrt(np.outer(default_c6, default_c6))
    c12 = np.sqrt(np.outer(default_c12, default_c12))
    rule = prior.has_rule[np.ix_(chem, chem)]
    c6 = np.where(rule, prior.c6[np.ix_(chem, chem)], c6)
    c12 = np.where(rule, prior.c12[np.ix_(chem, chem)], c12)
    i, j = np.asarray(overrides["i"], int), np.asarray(overrides["j"], int)
    c6[i, j] = c6[j, i] = overrides["c6"]
    c12[i, j] = c12[j, i] = overrides["c12"]
    return c6, c12
