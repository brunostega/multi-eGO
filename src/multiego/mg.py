from . import type_definitions

import numpy as np
import pandas as pd
import itertools


def _make_combinations(sbtype1, sbtype2):
    """Return the full ordered list of (ai, aj) pairs for the given sb_type sets."""
    if sbtype1 == sbtype2:
        return list(itertools.product(sbtype1, repeat=2))
    return list(set(itertools.product(sbtype1, sbtype2)) | set(itertools.product(sbtype2, sbtype1)))


def generate_MG_LJ_pairs_rep(sbtype1, sbtype2, sbtype_c12_dict, c12_rep=None, factor=1.0):
    """
    Generates repulsive LJ pairs for the MG prior force field.

    Parameters
    ----------
    sbtype1, sbtype2 : list of str
        Lists of sb_type identifiers for the two interacting groups.
    sbtype_c12_dict : dict
        Mapping from sb_type to rc_c12 value, used when c12_rep is not specified.
    c12_rep : float or None
        Fixed repulsive c12 value. If None, computed as geometric mean from dictionary.
    factor : float
        Divisor applied to c12_rep (default 1.0).

    Returns
    -------
    pd.DataFrame
        Pairs DataFrame with columns ai, aj, c6, c12, epsilon, sigma, mg_sigma, mg_epsilon.
    """
    combinations = _make_combinations(sbtype1, sbtype2)

    if c12_rep is None:
        c12_rep = np.array([np.sqrt(sbtype_c12_dict[ai] * sbtype_c12_dict[aj]) for ai, aj in combinations])

    pairs_LJ = pd.DataFrame(combinations, columns=["ai", "aj"])
    pairs_LJ["c12"] = c12_rep / factor
    pairs_LJ["c6"] = 0.0
    pairs_LJ["epsilon"] = -c12_rep / factor
    pairs_LJ["sigma"] = (c12_rep / factor) ** (1.0 / 12.0) / 2.0 ** (1.0 / 6.0)
    pairs_LJ["mg_sigma"] = pairs_LJ["sigma"]
    pairs_LJ["mg_epsilon"] = -c12_rep / factor

    return pairs_LJ.reset_index(drop=True)


def generate_MG_LJ_pairs_attr(sbtype1, sbtype2, sbtype_mg_c12_dict, sbtype_mg_c6_dict, epsilon=None, sigma=None):
    """
    Generates attractive LJ pairs for the MG prior force field.

    Parameters
    ----------
    sbtype1, sbtype2 : list of str
        Lists of sb_type identifiers for the two interacting groups.
    sbtype_mg_c12_dict : dict
        Mapping from sb_type to mg_c12, used to derive sigma when not specified.
    sbtype_mg_c6_dict : dict
        Mapping from sb_type to mg_c6, used to derive sigma when not specified.
    epsilon : float or None
        Interaction well depth.
    sigma : float or None
        Interaction distance parameter. Cannot be provided without epsilon.

    Returns
    -------
    pd.DataFrame
        Pairs DataFrame with columns ai, aj, c6, c12, epsilon, sigma, mg_sigma, mg_epsilon.
    """
    combinations = _make_combinations(sbtype1, sbtype2)

    if epsilon is not None and sigma is None:
        c6 = np.array([np.sqrt(sbtype_mg_c6_dict[ai] * sbtype_mg_c6_dict[aj]) for ai, aj in combinations])
        c12_rep = np.array([np.sqrt(sbtype_mg_c12_dict[ai] * sbtype_mg_c12_dict[aj]) for ai, aj in combinations])
        sigma = (c12_rep / c6) ** (1.0 / 6.0)
    elif epsilon is None and sigma is not None:
        raise ValueError("You can provide a custom value for epsilon but not for sigma alone")

    pairs_LJ = pd.DataFrame(combinations, columns=["ai", "aj"])
    pairs_LJ["c12"] = 4.0 * epsilon * sigma**12.0
    pairs_LJ["c6"] = 4.0 * epsilon * sigma**6.0
    pairs_LJ["epsilon"] = epsilon
    pairs_LJ["sigma"] = sigma
    pairs_LJ["mg_sigma"] = sigma
    pairs_LJ["mg_epsilon"] = epsilon

    return pairs_LJ.reset_index(drop=True)


def _per_type_values(meGO_ensemble, sbtype_dict, label):
    """
    Collapse a per-sb_type dictionary to a per-chemical-type dictionary.

    The mg prior is defined on chemical types; this is only valid if all atoms
    of a chemical type carry the same value, which is checked here.
    """
    values = {}
    for sbtype, value in sbtype_dict.items():
        atomtype = meGO_ensemble.sbtype_type_dict[sbtype]
        if atomtype in values and values[atomtype] != value:
            raise ValueError(
                f"{label} differs between atoms of chemical type {atomtype} "
                f"({values[atomtype]} vs {value}): the mg prior cannot be defined per type"
            )
        values[atomtype] = value
    return values


def generate_MG_prior(meGO_ensemble):
    """
    Generates the molten-globule prior as a matrix over chemical atom types.

    The prior rules in ``type_definitions.special_non_local`` are defined on
    chemical atom types, so instead of expanding every rule to all pairs of
    atoms (an N x N DataFrame, as the former ``generate_MG_LJ`` did) the
    parameters are stored once per pair of types.  The atom
    pairs are generated later, only when writing (see
    ``nonbonded.NonbondedModel``).

    The values are computed by the same functions that built the former
    per-atom-pair prior (``generate_MG_LJ_pairs_rep/attr``), called with
    chemical types in place of sb_types, so the numbers are identical.

    Parameters
    ----------
    meGO_ensemble : MeGOEnsemble
        The initialized meGO ensemble.

    Returns
    -------
    nonbonded.TypePrior
        Prior over all chemical types present in the topology.  Type pairs
        without a rule fall back to the combination rule.
    """
    from .nonbonded import TypePrior

    types = np.array(sorted(set(meGO_ensemble.sbtype_type_dict.values())))
    index = {t: k for k, t in enumerate(types)}
    n = len(types)
    prior = TypePrior(
        types=types,
        has_rule=np.zeros((n, n), dtype=bool),
        c6=np.zeros((n, n)),
        c12=np.zeros((n, n)),
        epsilon=np.zeros((n, n)),
        sigma=np.zeros((n, n)),
    )

    # Per-type versions of the per-sb_type dictionaries used by the rules.
    c12_by_type = _per_type_values(meGO_ensemble, meGO_ensemble.sbtype_c12_dict, "c12")
    mg_c12_by_type = _per_type_values(meGO_ensemble, meGO_ensemble.sbtype_mg_c12_dict, "mg_c12")
    mg_c6_by_type = _per_type_values(meGO_ensemble, meGO_ensemble.sbtype_mg_c6_dict, "mg_c6")

    for special in type_definitions.special_non_local:
        types_a = [t for t in special["atomtypes"][0] if t in index]
        types_b = [t for t in special["atomtypes"][1] if t in index]
        if not types_a or not types_b:
            continue  # rule for types absent from this system
        if special["interaction"] == "rep":
            rows = generate_MG_LJ_pairs_rep(
                types_a, types_b, c12_by_type, c12_rep=special["epsilon"], factor=special.get("factor", 1.0)
            )
        elif special["interaction"] == "att":
            rows = generate_MG_LJ_pairs_attr(
                types_a, types_b, mg_c12_by_type, mg_c6_by_type, epsilon=special["epsilon"], sigma=special["sigma"]
            )
        else:
            raise ValueError(f"Unknown interaction type {special['interaction']} in special_non_local definition")

        # rows contains both orientations (a, b) and (b, a): fill symmetrically.
        a = rows["ai"].map(index).to_numpy()
        b = rows["aj"].map(index).to_numpy()
        if prior.has_rule[a, b].any():
            # Two rules for the same pair would be ambiguous (duplicated atom pairs).
            raise ValueError(f"two special_non_local rules define the same type pair: {special['atomtypes']}")
        prior.has_rule[a, b] = True
        prior.c6[a, b] = rows["c6"].to_numpy()
        prior.c12[a, b] = rows["c12"].to_numpy()
        prior.epsilon[a, b] = rows["epsilon"].to_numpy()
        prior.sigma[a, b] = rows["sigma"].to_numpy()

    # sb_type -> chemical type, used to look up the prior of an atom pair.
    prior.sb_chem = dict(meGO_ensemble.sbtype_type_dict)
    return prior


def prior_stats_frame(meGO_ensemble, prior):
    """
    The prior as one row per ordered pair of chemical types with a rule, with
    the number of atom pairs it applies to (``count``), for
    ``fileio.print_stats``.  The counts reproduce the former statistics, which
    were computed on all ordered pairs of atoms.
    """
    n_atoms = pd.Series(list(meGO_ensemble.sbtype_type_dict.values())).value_counts()
    a, b = np.nonzero(prior.has_rule)
    return pd.DataFrame(
        {
            "epsilon": prior.epsilon[a, b],
            "sigma": prior.sigma[a, b],
            "same_chain": False,
            "count": n_atoms[prior.types[a]].to_numpy() * n_atoms[prior.types[b]].to_numpy(),
        }
    )
