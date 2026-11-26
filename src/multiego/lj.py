from . import type_definitions
from . import bonded
from . import _term
from .model_config import config
from . import fileio

import gc
import numpy as np
import pandas as pd
import itertools
import time


def create_linearized_mask(
    set1,
    set2,
    types,
    symmetrize=False,
    inner_op=lambda x, y: x == y,
    outer_op=lambda x, y: x * y,
):
    """
    Creates a linearized boolean mask based on comparison operations between two sets.

    For each ``(type1, type2)`` pair in ``types``, ``inner_op`` is applied
    element-wise to each set and the results are combined with ``outer_op``.
    The final mask is the logical OR across all type pairs.  When
    ``symmetrize=True`` each ``(type1, type2)`` entry is automatically
    complemented by ``(type2, type1)``.

    Parameters
    ----------
    set1 : numpy.ndarray
        First 1-D array of values (e.g. atom-type labels).
    set2 : numpy.ndarray
        Second 1-D array of values, same length as ``set1``.
    types : list of tuple
        Pairs ``(type1, type2)`` used for element-wise comparisons.
    symmetrize : bool, optional
        If ``True``, automatically add the reversed pair ``(type2, type1)``
        for every entry in ``types`` (default ``False``).
    inner_op : callable, optional
        Element-wise comparison applied to each array against a type value
        (default ``lambda x, y: x == y``).
    outer_op : callable, optional
        Combines the two per-set results into a single boolean array
        (default ``lambda x, y: x * y``, i.e. logical AND).

    Returns
    -------
    numpy.ndarray
        1-D boolean mask of length ``set1.shape[0]``.
    """
    if symmetrize:
        types = list(set(types + [(t[1], t[0]) for t in types]))

    mask = np.full(set1.shape[0], False)
    for type1, type2 in types:
        mask |= outer_op(inner_op(set1, type1), inner_op(set2, type2))

    return mask


def set_sig_epsilon(meGO_LJ, parameters):
    """
    Set sigma and epsilon for all LJ interactions based on probability and distance.

    Attractive interactions are assigned when the training probability exceeds the
    adaptive threshold relative to the RC distribution. Repulsive interactions are
    assigned when training probability is above MD threshold but below the attractive
    threshold. 1-4 interactions are always kept repulsive.

    With x = ln(p_MD / max(p_RC, p_RC^th)) / ln(1 / p_RC^th) and eps_p the
    prior energy (0 for repulsive priors), the attractive epsilon is
    eps_p + (eps_0 - eps_p) * x (``--reweighting interpolate``, between the
    prior and eps_0) or eps_p + eps_0 * x (``--reweighting additive``, eps_0
    is the largest energy added to the prior). It is attractive when it
    exceeds epsilon_min; otherwise (training above its threshold, RC above the
    MD threshold) the contact is repulsive with c12 = rep (d_MD / d_RC)^12,
    clipped to [0.05, 2] rep, for both rules.

    Parameters
    ----------
    meGO_LJ : pd.DataFrame
        DataFrame containing LJ parameters including probability thresholds.
    parameters : argparse.Namespace
        Parsed arguments (``learn_tolerance`` is now read from ``config``).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with epsilon and sigma columns set.
    """
    # first: all contacts are set as for the prior model
    # these contacts are not considered as learned so can be overriden
    meGO_LJ["learned"] = 0
    meGO_LJ["epsilon"] = meGO_LJ["epsilon_prior"]
    meGO_LJ["sigma"] = meGO_LJ["sigma_prior"]

    # Precompute shared sub-expression (used in both attractive and repulsive conditions)
    eps_prior_pos = np.maximum(0.0, meGO_LJ["epsilon_prior"])

    # Attractive interactions
    # These are defined only if the training probability is greater than MD_threshold and
    # by comparing them with RC_probabilities so that the resulting epsilon is between eps_min and eps_0
    condition = (
        meGO_LJ["probability"]
        > meGO_LJ["limit_rc_att"] * np.maximum(meGO_LJ["rc_probability"], meGO_LJ["rc_threshold"])
    ) & (meGO_LJ["probability"] > meGO_LJ["md_threshold"])
    # slope: 'interpolate' -> eps between eps_prior and eps_0; 'additive' -> eps_0 is the largest
    # energy added to the prior (the thresholds of contacts.initialize_molecular_contacts match the rule)
    if getattr(parameters, "reweighting", "interpolate") == "additive":
        scale = meGO_LJ["epsilon_0"]
    else:
        scale = meGO_LJ["epsilon_0"] - eps_prior_pos
    with np.errstate(divide="ignore"):
        meGO_LJ.loc[condition, "epsilon"] = eps_prior_pos - (scale / np.log(meGO_LJ["rc_threshold"])) * (
            np.log(meGO_LJ["probability"] / (np.maximum(meGO_LJ["rc_probability"], meGO_LJ["rc_threshold"])))
        )
    meGO_LJ.loc[condition, "learned"] = 1
    meGO_LJ.loc[condition, "sigma"] = meGO_LJ["distance"] / 2.0 ** (1.0 / 6.0)

    # Repulsive interactions (probability above MD threshold but below attractive threshold)
    # this is used only when MD_th < MD_p < limit_rc_att*RC_p
    condition = (
        (
            meGO_LJ["probability"]
            <= meGO_LJ["limit_rc_att"] * np.maximum(meGO_LJ["rc_probability"], meGO_LJ["rc_threshold"])
        )
        & (meGO_LJ["probability"] > meGO_LJ["md_threshold"])
        & (meGO_LJ["rc_probability"] > meGO_LJ["md_threshold"])
    )
    sub = meGO_LJ.loc[condition]
    meGO_LJ.loc[condition, "epsilon"] = (-sub["rep"] * (sub["distance"] / sub["rc_distance"]) ** 12).clip(
        lower=-2 * sub["rep"], upper=-0.05 * sub["rep"]
    )
    meGO_LJ.loc[condition, "learned"] = 1

    # for repulsive interactions reset sigma to its effective value for consistent merging
    meGO_LJ.loc[(meGO_LJ["epsilon"] < 0.0), "sigma"] = (-meGO_LJ["epsilon"]) ** (1.0 / 12.0) / (2.0 ** (1.0 / 6.0))

    meGO_LJ.dropna(subset=["epsilon"], inplace=True)
    meGO_LJ = meGO_LJ[meGO_LJ.epsilon != 0]

    return meGO_LJ


def apply_symmetries(meGO_ensemble, meGO_input, symmetry):
    """
    Apply symmetries to the molecular ensemble.

    Parameters
    ----------
    meGO_ensemble : dict
        A dictionary containing relevant meGO data.
    meGO_input : pd.DataFrame
        Input DataFrame containing molecular ensemble data.
    symmetry : list
        List of symmetry definitions parsed from arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame with applied symmetries.
    """
    if meGO_input.empty:
        return pd.DataFrame(columns=meGO_input.columns)

    if not symmetry:
        return pd.DataFrame(columns=meGO_input.columns)

    topology_df = meGO_ensemble.topology_dataframe
    dict_sbtype_to_resname = topology_df.set_index("sb_type")["resname"].to_dict()

    # Pre-compute sb_types that belong to C-terminal (CTER) and N-terminal (NTER)
    # residues — the last and first residue of each molecule respectively.
    cter_sbtypes: set = set()
    nter_sbtypes: set = set()
    topology_df["_resnum_int"] = topology_df["resnum"].astype(int)
    for _, mol_group in topology_df.groupby("molecule_name"):
        resnums = mol_group["_resnum_int"]
        cter_sbtypes |= set(mol_group.loc[resnums == resnums.max(), "sb_type"])
        nter_sbtypes |= set(mol_group.loc[resnums == resnums.min(), "sb_type"])
    topology_df.drop(columns=["_resnum_int"], inplace=True)

    def _filter_and_masks(df, ai_col, aj_col, sym0):
        """Return (filtered_df, ai_mask, aj_mask) for a given residue specifier."""
        if sym0 == "CTER":
            terminal = cter_sbtypes
            ai_mask = df[ai_col].isin(terminal)
            aj_mask = df[aj_col].isin(terminal)
        elif sym0 == "NTER":
            terminal = nter_sbtypes
            ai_mask = df[ai_col].isin(terminal)
            aj_mask = df[aj_col].isin(terminal)
        else:
            resn_ai = df[ai_col].map(dict_sbtype_to_resname)
            resn_aj = df[aj_col].map(dict_sbtype_to_resname)
            ai_mask = resn_ai == sym0
            aj_mask = resn_aj == sym0
        return df[ai_mask | aj_mask], ai_mask, aj_mask

    df_list = []

    for sym in symmetry:
        if not sym:
            continue
        meGO_filtered, mgf_mask_ai, mgf_mask_aj = _filter_and_masks(meGO_input, "ai", "aj", sym[0])
        # Reindex masks to match the filtered frame
        mgf_mask_ai = mgf_mask_ai.reindex(meGO_filtered.index)
        mgf_mask_aj = mgf_mask_aj.reindex(meGO_filtered.index)
        for atypes in itertools.permutations(sym[1:]):
            t_df_ai = meGO_filtered[meGO_filtered["ai"].str.startswith(f"{atypes[0]}_") & mgf_mask_ai]
            t_df_aj = meGO_filtered[meGO_filtered["aj"].str.startswith(f"{atypes[0]}_") & mgf_mask_aj]
            t_df_ai.loc[:, "ai"] = t_df_ai["ai"].str.replace(r"^(.*?)_", atypes[1] + "_", regex=True)
            t_df_aj.loc[:, "aj"] = t_df_aj["aj"].str.replace(r"^(.*?)_", atypes[1] + "_", regex=True)
            df_list.extend([t_df_ai, t_df_aj])

    df_tmp = pd.concat(df_list, ignore_index=True)
    df_list = []
    df_tmp.drop_duplicates(inplace=True)

    for sym in symmetry:
        if not sym:
            continue
        df_tmp_filt, df_mask_ai, df_mask_aj = _filter_and_masks(df_tmp, "ai", "aj", sym[0])
        df_mask_ai = df_mask_ai.reindex(df_tmp_filt.index)
        df_mask_aj = df_mask_aj.reindex(df_tmp_filt.index)
        for atypes in itertools.permutations(sym[1:]):
            t_df_ai = df_tmp_filt[df_tmp_filt["ai"].str.startswith(f"{atypes[0]}_") & df_mask_ai]
            t_df_aj = df_tmp_filt[df_tmp_filt["aj"].str.startswith(f"{atypes[0]}_") & df_mask_aj]
            t_df_ai.loc[:, "ai"] = t_df_ai["ai"].str.replace(r"^(.*?)_", atypes[1] + "_", regex=True)
            t_df_aj.loc[:, "aj"] = t_df_aj["aj"].str.replace(r"^(.*?)_", atypes[1] + "_", regex=True)
            df_list.extend([t_df_ai, t_df_aj])

    tmp_df = pd.concat(df_list + [df_tmp], ignore_index=True)
    tmp_df.drop_duplicates(inplace=True)

    return tmp_df


# Number of rows of a contact matrix annotated at once by init_LJ_datasets.
# Only affects memory use (and very slightly speed), never the result.
ANNOTATION_ROWS = 250_000


def init_LJ_datasets(meGO_ensemble, matrices, args):
    """
    Assembles the full training dataset by merging train/reference contact matrices
    with 1-4 pair data and computing default repulsive and MG sigma/epsilon values.

    Parameters
    ----------
    meGO_ensemble : dict
        The initialized meGO ensemble.
    matrices : dict
        Contains 'reference_matrices' and 'train_matrices'.  The matrices are
        removed from these dicts as soon as they have been used, to release
        their memory.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    train_dataset : pd.DataFrame
        Fully assembled and annotated training dataset ready for generate_LJ.
        Only the rows with ``learned == 1`` are returned: they are the only
        ones generate_LJ uses, and keeping the others would need memory
        proportional to (number of matrices) x N^2.
    """
    td_fields = [
        "molecule_name_ai",
        "ai",
        "molecule_name_aj",
        "aj",
        "distance",
        "probability",
        "cutoff",
        "same_chain",
        "source",
        "reference",
        "epsilon_0",
        "epsilon_prior",
        "sigma_prior",
        "md_threshold",
        "rc_threshold",
        "limit_rc_att",
        "rc_distance",
        "rc_probability",
        "learned",
    ]

    # Inputs of the row-wise annotation that do not depend on the chunk.
    all_bd = bonded.generate_bond_distance_data(meGO_ensemble)
    rule_tables = _special_rule_tables(meGO_ensemble)
    c12_table = _nthbond_c12_table(meGO_ensemble)
    backbone_rule = None
    if getattr(args, "hbond", False):
        from . import hbond

        backbone_rule = (hbond.sbtype_roles(meGO_ensemble), hbond.backbone_repulsive_c12(meGO_ensemble))

    # A reference matrix can serve several training matrices: count its uses
    # so that it is freed after the last one.
    remaining_uses = {}
    for _, ref_name in meGO_ensemble.train_matrix_tuples:
        remaining_uses[ref_name] = remaining_uses.get(ref_name, 0) + 1

    chunks = []
    for name, ref_name in meGO_ensemble.train_matrix_tuples:
        if ref_name not in matrices["reference_matrices"]:
            raise RuntimeError(
                f'Encountered error while trying to find {ref_name} in reference matrices {matrices["reference_matrices"].keys()}'
            )

        temp_merged = pd.merge(
            matrices["train_matrices"][name],
            matrices["reference_matrices"][ref_name],
            left_index=True,
            right_index=True,
            how="outer",
        )

        if (np.abs(temp_merged["rc_cutoff"] - temp_merged["cutoff"])).max() > 0:
            detail = (
                temp_merged[["ai", "aj", "rc_ai", "rc_aj", "source", "rc_source", "cutoff", "rc_cutoff", "same_chain"]]
                .loc[(np.abs(temp_merged["rc_cutoff"] - temp_merged["cutoff"]) > 0)]
                .to_string()
            )
            raise RuntimeError(
                f"Inconsistent cutoff values between the TRAINING and corresponding REFERENCE input data\n{detail}"
            )

        # Restrict the check to contacts that appear in both matrices: the outer
        # merge leaves rc_same_chain = NaN for training contacts that have no
        # counterpart in the reference, and those rows must not trigger the error.
        # Cast to bool before comparing: after an outer join, boolean columns that
        # receive any NaN are upcast to object dtype, and Series.equals() is
        # dtype-sensitive, so object([True]) != bool([True]) even though the
        # values are identical. A plain != on bool-cast values avoids this.
        both_present = temp_merged["same_chain"].notna() & temp_merged["rc_same_chain"].notna()
        mismatched = both_present & (
            temp_merged["same_chain"].astype(bool) != temp_merged["rc_same_chain"].astype(bool)
        )
        if mismatched.any():
            diff_indices = temp_merged.index[mismatched].tolist()
            raise RuntimeError(
                f"You are pairing intra and inter molecular training and reference data\n"
                f"Mismatched indices: {diff_indices}"
            )

        # All further processing is row by row: do it on slices of
        # ANNOTATION_ROWS rows and keep only the rows generate_LJ uses
        # (learned == 1).  Every row is still validated (see
        # _annotate_training_chunk), but neither the annotated copy of the
        # whole matrix nor the concatenation of all matrices -- several times
        # N^2 rows for a protein of N atoms -- is ever built.
        for start in range(0, len(temp_merged), ANNOTATION_ROWS):
            stop = start + ANNOTATION_ROWS
            piece = temp_merged.iloc[start:stop][td_fields]
            piece = _annotate_training_chunk(meGO_ensemble, piece, all_bd, rule_tables, c12_table, backbone_rule)
            chunks.append(piece.loc[piece["learned"] == 1])
        del temp_merged

        # Free the input matrices after their last use.
        del matrices["train_matrices"][name]
        remaining_uses[ref_name] -= 1
        if remaining_uses[ref_name] == 0:
            del matrices["reference_matrices"][ref_name]
        gc.collect()

    train_dataset = (
        pd.concat(chunks, axis=0, sort=False, ignore_index=True) if chunks else pd.DataFrame(columns=td_fields)
    )
    for col in ("molecule_name_ai", "molecule_name_aj", "source", "ai", "aj"):
        train_dataset[col] = train_dataset[col].astype("category")
    return train_dataset


def _special_rule_tables(meGO_ensemble):
    """
    Lookup tables of the special non-local (mg prior) rules, indexed by an
    integer code of the chemical atom type.

    The naive approach iterates over every rule, computes a boolean mask by
    scanning type_ai/type_aj for each (type_i, type_j) pair (186 scans total),
    then writes results with .loc -- slow for large datasets.  Instead we build
    tiny 2D lookup tables indexed by integer atom-type codes, so that a single
    vectorised fancy-index gives the per-row rule outcome.

    Returns
    -------
    dict
        ``type_to_code`` (types absent from the rules get code 0, "no rule
        applies") and the ``outcome``, ``epsilon`` and ``sigma`` tables.
    """
    # --- Step 1: assign a small integer code to every atom type in the rules ---
    all_rule_types = sorted(
        {atomtype for rule in type_definitions.special_non_local for side in rule["atomtypes"] for atomtype in side}
    )
    type_to_code = {atomtype: code for code, atomtype in enumerate(all_rule_types, start=1)}
    n_codes = len(all_rule_types) + 1  # +1 for the 0 sentinel

    # --- Step 2: fill the lookup tables (one entry per ordered type pair) ---
    # Rule-outcome codes (int8 to keep the tables tiny): see RULE_* below.
    lut_outcome = np.zeros((n_codes, n_codes), dtype=np.int8)
    lut_epsilon = np.full((n_codes, n_codes), np.nan)
    lut_sigma = np.full((n_codes, n_codes), np.nan)

    for rule in type_definitions.special_non_local:
        types_i, types_j = rule["atomtypes"]
        epsilon = rule["epsilon"]
        sigma = rule.get("sigma")
        if rule["interaction"] == "rep":
            outcome = RULE_REP_FIXED if epsilon is not None else RULE_REP_NONE
        else:
            outcome = RULE_ATT
        for ti in types_i:
            for tj in types_j:
                for ci, cj in ((type_to_code[ti], type_to_code[tj]), (type_to_code[tj], type_to_code[ti])):
                    lut_outcome[ci, cj] = outcome
                    lut_epsilon[ci, cj] = epsilon if epsilon is not None else np.nan
                    lut_sigma[ci, cj] = sigma if sigma is not None else np.nan
    return {"type_to_code": type_to_code, "outcome": lut_outcome, "epsilon": lut_epsilon, "sigma": lut_sigma}


# Rule-outcome codes of _special_rule_tables:
#   1 = repulsive with a fixed epsilon value
#   2 = repulsive, derive mg parameters from the existing rep (epsilon=None)
#   3 = attractive
RULE_REP_FIXED, RULE_REP_NONE, RULE_ATT = np.int8(1), np.int8(2), np.int8(3)


def _nthbond_c12_table(meGO_ensemble):
    """
    Default repulsive c12 of every pair of chemical types, from
    ``type_definitions._c12_df`` (0 where the pair is not listed).

    Replaces the former per-row ``type_pair`` strings ("CH2_O"): the value is
    looked up in a small (types x types) table indexed by integer type codes.
    Where the file lists a pair in both orders with different values, the
    first occurrence wins, as before (both "atp1_atp2" and then "atp2_atp1"
    keys were collected and the first duplicate was kept).

    Returns
    -------
    dict
        ``type_to_code`` and the ``c12`` table.
    """
    types = sorted(set(meGO_ensemble.sbtype_type_dict.values()))
    type_to_code = {t: k for k, t in enumerate(types)}
    # One extra row/column of zeros (code len(types)) for unknown types.
    table = np.zeros((len(types) + 1, len(types) + 1))
    seen = np.zeros(table.shape, dtype=bool)
    df = type_definitions._c12_df
    # Same precedence as the former lookup: all (atp1, atp2) keys first, then
    # all (atp2, atp1) keys, the first occurrence of a key winning.
    for first, second in (("atp1", "atp2"), ("atp2", "atp1")):
        for t1, t2, c12 in zip(df[first], df[second], df["c12"]):
            if t1 in type_to_code and t2 in type_to_code:
                i, j = type_to_code[t1], type_to_code[t2]
                if not seen[i, j]:
                    table[i, j], seen[i, j] = c12, True
    return {"type_to_code": type_to_code, "c12": table}


def _annotate_training_chunk(meGO_ensemble, train_dataset, all_bd, rule_tables, c12_table, backbone_rule=None):
    """
    Row-wise annotation of one merged training/reference contact matrix:
    bond distances, default repulsion, mg sigma/epsilon and the special
    non-local rules, followed by the consistency check of the input cutoffs.

    ``backbone_rule`` (``--hbond``): (sb_type -> backbone role, 5 x 5 c12
    table) of the repulsive polar-backbone prior (``multiego.hbond``); it
    takes precedence over the special rules for the pairs it covers.

    Every operation is row-wise, so annotating each matrix separately gives
    the same rows as annotating their concatenation.
    """
    train_dataset = train_dataset.copy()
    train_dataset["molecule_name_ai"] = train_dataset["molecule_name_ai"].astype("category")
    train_dataset["molecule_name_aj"] = train_dataset["molecule_name_aj"].astype("category")
    train_dataset["source"] = train_dataset["source"].astype("category")

    train_dataset = pd.merge(
        train_dataset,
        all_bd[["ai", "aj", "bond_distance"]],
        how="left",
        on=["ai", "aj"],
    )
    train_dataset["bond_distance"] = train_dataset["bond_distance"].fillna(config.max_bond_separation + 1).astype(int)

    train_dataset["ai"] = train_dataset["ai"].astype("category")
    train_dataset["aj"] = train_dataset["aj"].astype("category")

    # Remove 0_1_2_3 intramolecular interactions
    train_dataset = train_dataset[
        ~((train_dataset["same_chain"]) & (train_dataset["bond_distance"] <= config.bond14_separation))
    ]
    train_dataset = train_dataset.reset_index(drop=True)

    type_ai = train_dataset["ai"].map(meGO_ensemble.sbtype_type_dict)
    type_aj = train_dataset["aj"].map(meGO_ensemble.sbtype_type_dict)

    # default repulsive C12: data-driven value of the pair of chemical types,
    # 0 where the pair is not listed (such rows are removed below).
    code = c12_table["type_to_code"]
    unknown = len(code)  # zero row/column of the table
    train_dataset["rep"] = c12_table["c12"][
        type_ai.map(code).fillna(unknown).to_numpy(dtype=np.intp),
        type_aj.map(code).fillna(unknown).to_numpy(dtype=np.intp),
    ]

    # default (mg) sigma
    pairwise_mg_sigma = (
        train_dataset["ai"].map(meGO_ensemble.sbtype_mg_c12_dict)
        * train_dataset["aj"].map(meGO_ensemble.sbtype_mg_c12_dict)
        / (
            train_dataset["ai"].map(meGO_ensemble.sbtype_mg_c6_dict)
            * train_dataset["aj"].map(meGO_ensemble.sbtype_mg_c6_dict)
        )
    ) ** (1 / 12)
    train_dataset["mg_sigma"] = pairwise_mg_sigma

    # default (mg) epsilon
    pairwise_mg_epsilon = (
        train_dataset["ai"].map(meGO_ensemble.sbtype_mg_c6_dict)
        * train_dataset["aj"].map(meGO_ensemble.sbtype_mg_c6_dict)
    ) / (
        4
        * np.sqrt(
            train_dataset["ai"].map(meGO_ensemble.sbtype_mg_c12_dict)
            * train_dataset["aj"].map(meGO_ensemble.sbtype_mg_c12_dict)
        )
    )
    train_dataset["mg_epsilon"] = pairwise_mg_epsilon

    # Apply the special non-local interaction rules (see _special_rule_tables).
    # --- encode every row's ai/aj type as an integer code ---
    type_to_code = rule_tables["type_to_code"]
    ai_codes = type_ai.map(type_to_code).fillna(0).to_numpy(dtype=np.intp)
    aj_codes = type_aj.map(type_to_code).fillna(0).to_numpy(dtype=np.intp)

    # --- single vectorised lookup: one numpy fancy-index per table ---
    row_outcome = rule_tables["outcome"][ai_codes, aj_codes]
    row_epsilon = rule_tables["epsilon"][ai_codes, aj_codes]
    row_sigma = rule_tables["sigma"][ai_codes, aj_codes]

    # --- apply all rule outcomes in one pass with np.where on numpy arrays ---
    # Working on raw arrays avoids the per-call pandas alignment overhead of .loc.
    rep = train_dataset["rep"].to_numpy(dtype=float)
    mg_epsilon = train_dataset["mg_epsilon"].to_numpy(dtype=float)
    mg_sigma = train_dataset["mg_sigma"].to_numpy(dtype=float)

    is_rep_fixed = row_outcome == RULE_REP_FIXED
    is_rep_none = row_outcome == RULE_REP_NONE
    is_att = row_outcome == RULE_ATT

    # repulsive, fixed epsilon: override rep, mg_epsilon, and mg_sigma
    rep = np.where(is_rep_fixed, row_epsilon, rep)
    mg_epsilon = np.where(is_rep_fixed, -row_epsilon, mg_epsilon)
    mg_sigma = np.where(is_rep_fixed, row_epsilon ** (1 / 12) / 2 ** (1 / 6), mg_sigma)

    # repulsive, epsilon=None: derive mg_epsilon and mg_sigma from the existing rep
    mg_epsilon = np.where(is_rep_none, -rep, mg_epsilon)
    mg_sigma = np.where(is_rep_none, rep ** (1 / 12) / 2 ** (1 / 6), mg_sigma)

    # attractive: override mg_epsilon and/or mg_sigma where the rule specifies them
    mg_epsilon = np.where(is_att & ~np.isnan(row_epsilon), row_epsilon, mg_epsilon)
    mg_sigma = np.where(is_att & ~np.isnan(row_sigma), row_sigma, mg_sigma)

    # --hbond: repulsive polar-backbone prior
    if backbone_rule is not None:
        roles, bb_table = backbone_rule
        role_i = train_dataset["ai"].astype(str).map(roles).fillna(0).to_numpy(dtype=np.intp)
        role_j = train_dataset["aj"].astype(str).map(roles).fillna(0).to_numpy(dtype=np.intp)
        bb_c12 = bb_table[role_i, role_j]
        is_bb = ~np.isnan(bb_c12)
        mg_epsilon = np.where(is_bb, -bb_c12, mg_epsilon)
        mg_sigma = np.where(is_bb, bb_c12 ** (1 / 12) / 2 ** (1 / 6), mg_sigma)

    train_dataset["rep"] = rep
    train_dataset["mg_epsilon"] = mg_epsilon
    train_dataset["mg_sigma"] = mg_sigma

    train_dataset.dropna(subset=["mg_sigma"], inplace=True)
    train_dataset = train_dataset.loc[train_dataset["rep"] > 0.0]

    if (np.abs(train_dataset["cutoff"] - 1.45 * train_dataset["rep"] ** (1 / 12))).max() > 1e-5:
        detail = (
            train_dataset[["ai", "aj", "source", "same_chain", "cutoff", "rep"]]
            .loc[(np.abs(train_dataset["cutoff"] - 1.45 * train_dataset["rep"] ** (1 / 12)) > 1e-5)]
            .to_string()
        )
        raise RuntimeError(f"Inconsistent cutoff and C12 repulsive values\n{detail}")

    return train_dataset


def _add_c6_c12(df):
    """Compute c6/c12 columns in-place from epsilon and sigma."""
    df["c6"] = np.where(df["epsilon"] < 0.0, 0.0, 4 * df["epsilon"] * df["sigma"] ** 6)
    df["c12"] = np.where(df["epsilon"] < 0.0, -df["epsilon"], 4 * df["epsilon"] * df["sigma"] ** 12)


def resolve_overrides(meGO_LJ, prior):
    """
    Reduce the learned interactions to one row per unordered atom pair, i.e.
    the overrides of the mg prior.

    This reproduces the former merge of the learned interactions with the
    fully expanded mg prior (concatenate, symmetrize, sort by
    ``learned`` (descending) then ``sigma``, keep the first row per pair, then
    keep one orientation per pair) without expanding the prior:

    * among the learned rows of a pair (both orientations) the same row wins;
    * a row with ``learned = 1`` always beats the prior;
    * a row with ``learned = 0`` beats the prior only if its sigma is not
      larger than the prior sigma (on a tie the learned row is kept, as it
      came first in the former concatenation); otherwise it is dropped so
      that the prior applies;
    * of the two orientations (ai, aj)/(aj, ai) the one with ai <= aj
      (string order) is kept, as the former ``sort_LJ`` did.

    Parameters
    ----------
    meGO_LJ : pd.DataFrame
        Learned interactions (columns ai, aj, learned, sigma, epsilon, ...).
    prior : nonbonded.TypePrior
        The mg prior.  Its ``chem_of`` lookup is provided through the
        ``sb_chem`` mapping attached by ``mg.generate_MG_prior``.

    Returns
    -------
    pd.DataFrame
        One row per unordered atom pair.
    """
    meGO_LJ = meGO_LJ.copy()
    for col in ("ai", "aj", "molecule_name_ai", "molecule_name_aj"):
        meGO_LJ[col] = meGO_LJ[col].astype(str)

    # Symmetrize
    inverse_meGO_LJ = meGO_LJ.rename(
        columns={
            "ai": "aj",
            "aj": "ai",
            "molecule_name_ai": "molecule_name_aj",
            "molecule_name_aj": "molecule_name_ai",
        }
    )
    meGO_LJ = pd.concat([meGO_LJ, inverse_meGO_LJ], axis=0, sort=False, ignore_index=True)
    meGO_LJ.sort_values(by=["ai", "aj", "learned", "sigma"], ascending=[True, True, False, True], inplace=True)
    meGO_LJ = meGO_LJ.drop_duplicates(subset=["ai", "aj"], keep="first")
    meGO_LJ = meGO_LJ.loc[meGO_LJ["ai"] <= meGO_LJ["aj"]]

    # Non-learned rows compete with the prior on sigma.
    if (meGO_LJ["learned"] == 0).any():
        a = prior.index(meGO_LJ["ai"].map(prior.sb_chem))
        b = prior.index(meGO_LJ["aj"].map(prior.sb_chem))
        has_rule = prior.has_rule[a, b]
        prior_sigma = prior.sigma[a, b]
        if prior.backbone_c12 is not None:
            # --hbond: the repulsive polar-backbone rule replaces the type rule
            role_i = meGO_LJ["ai"].map(prior.sb_role).fillna(0).to_numpy(dtype=np.intp)
            role_j = meGO_LJ["aj"].map(prior.sb_role).fillna(0).to_numpy(dtype=np.intp)
            bb_c12 = prior.backbone_c12[role_i, role_j]
            is_bb = ~np.isnan(bb_c12)
            has_rule = has_rule | is_bb
            prior_sigma = np.where(is_bb, np.nan_to_num(bb_c12) ** (1 / 12) / 2 ** (1 / 6), prior_sigma)
        prior_wins = (meGO_LJ["learned"].to_numpy() == 0) & has_rule & (prior_sigma < meGO_LJ["sigma"].to_numpy())
        meGO_LJ = meGO_LJ.loc[~prior_wins]
    return meGO_LJ.reset_index(drop=True)


# --hbond: polar backbone pairs whose non-local contacts describe the hydrogen
# bond itself (H-O directly, N-O on the heavy atoms). They are not learned
# between atoms >= min_residue_separation residues apart or on different
# chains: the hydrogen-bond term describes them, and learning them would put
# the hydrogen bond back into the isotropic matrix.
HBOND_FROZEN_PAIRS = (("H", "O"), ("N", "O"))


def _hbond_frozen_classes(meGO_LJ, prior):
    """
    For ``--hbond``: per row, whether the pair is one of HBOND_FROZEN_PAIRS of
    backbone atoms (by backbone role, see ``multiego.hbond``).
    """
    from . import hbond

    role_i = meGO_LJ["ai"].astype(str).map(prior.sb_role).fillna(0).to_numpy(dtype=np.intp)
    role_j = meGO_LJ["aj"].astype(str).map(prior.sb_role).fillna(0).to_numpy(dtype=np.intp)
    out = np.zeros(len(meGO_LJ), dtype=bool)
    for a, b in HBOND_FROZEN_PAIRS:
        ra, rb = hbond.ROLE_INDEX[a], hbond.ROLE_INDEX[b]
        out |= ((role_i == ra) & (role_j == rb)) | ((role_i == rb) & (role_j == ra))
    return out


def _hbond_frozen_mask(meGO_LJ, prior, sb_resnum, min_residue_separation):
    """
    Rows that must not be learned with ``--hbond``: HBOND_FROZEN_PAIRS between
    different chains, or on the same chain at least ``min_residue_separation``
    residues apart.
    """
    cls = _hbond_frozen_classes(meGO_LJ, prior)
    res_i = meGO_LJ["ai"].astype(str).map(sb_resnum).to_numpy(dtype=float)
    res_j = meGO_LJ["aj"].astype(str).map(sb_resnum).to_numpy(dtype=float)
    far = np.abs(res_i - res_j) >= min_residue_separation
    return cls & ((~meGO_LJ["same_chain"].astype(bool).to_numpy()) | far)


def _hbond_prior_companions(meGO_LJ, prior):
    """
    Intermolecular prior rows for the learned local (same chain, < min
    separation) H-O / N-O contacts. A learned intramolecular entry without an
    intermolecular one would be written in the matrix and apply between chains
    too (the sb_type pair is shared by the copies); with this companion row the
    learned value goes to ``[ pairs ]`` and the matrix keeps the prior.
    """
    local = meGO_LJ.loc[_hbond_frozen_classes(meGO_LJ, prior) & meGO_LJ["same_chain"].astype(bool).to_numpy()]
    if local.empty:
        return local
    comp = local.copy()
    roles_i = comp["ai"].astype(str).map(prior.sb_role).to_numpy(dtype=np.intp)
    roles_j = comp["aj"].astype(str).map(prior.sb_role).to_numpy(dtype=np.intp)
    c12 = prior.backbone_c12[roles_i, roles_j]
    comp["same_chain"] = False
    comp["learned"] = 0
    comp["source"] = "mg"
    comp["probability"] = 0.0
    comp["rc_probability"] = 0.0
    comp["epsilon"] = -c12
    comp["sigma"] = c12 ** (1 / 12) / 2 ** (1 / 6)
    return comp


def generate_LJ(meGO_ensemble, train_dataset, parameters, prior):
    """
    Generates production LJ interactions from the training dataset.

    Parameters
    ----------
    meGO_ensemble : dict
        The initialized meGO ensemble.
    train_dataset : pd.DataFrame
        Output of init_LJ_datasets.
    parameters : argparse.Namespace
        Parsed command-line arguments.
    prior : nonbonded.TypePrior
        The molten-globule prior (``mg.generate_MG_prior``); learned
        interactions override it.

    Returns
    -------
    meGO_LJ : pd.DataFrame
        Learned non-bonded LJ interactions, one row per atom pair: the sparse
        overrides of the mg prior (see ``nonbonded.NonbondedModel``).
    meGO_LJ_14 : pd.DataFrame
        1-4 LJ interactions for the topology pairs section.
    stat_str : str
        Statistics string for the header.
    """
    st = time.time()
    _term.sub("Set sigma and epsilon")
    meGO_LJ = train_dataset[train_dataset["learned"] == 1].copy()

    # Local pairs (4 to max_bond_separation bonds, written in [ pairs ]): --learn_local_pairs
    local_mode = getattr(parameters, "learn_local_pairs", "no_phipsi")
    local_frozen_pairs = None
    if local_mode != "all":
        if local_mode == "no_phipsi":
            local_frozen_pairs = bonded.phipsi_dependent_pairs(meGO_ensemble)

        def _local_frozen_mask(df):
            local = (df["same_chain"].astype(bool) & (df["bond_distance"] <= config.max_bond_separation)).to_numpy()
            if local_frozen_pairs is None:  # none
                return local
            keys = zip(df["ai"].astype(str), df["aj"].astype(str))
            return local & np.fromiter((k in local_frozen_pairs for k in keys), dtype=bool, count=len(df))

        frozen_local = _local_frozen_mask(meGO_LJ)
        _term.note(f"--learn_local_pairs {local_mode}: {int(frozen_local.sum())} local pairs keep the prior")
        meGO_LJ = meGO_LJ.loc[~frozen_local]

    hbond_freeze = getattr(prior, "backbone_c12", None) is not None
    if hbond_freeze:
        from .hbond import HBondSettings

        settings = HBondSettings.from_args(parameters)
        min_sep = settings.min_residue_separation if settings is not None else 3
        top = meGO_ensemble.topology_dataframe
        sb_resnum = dict(zip(top["sb_type"].astype(str), top["resnum"].astype(int)))
        frozen = _hbond_frozen_mask(meGO_LJ, prior, sb_resnum, min_sep)
        _term.note(f"--hbond: {int(frozen.sum())} non-local backbone H-O / N-O contacts not learned")
        meGO_LJ = meGO_LJ.loc[~frozen]

    needed_fields = [
        "molecule_name_ai",
        "ai",
        "molecule_name_aj",
        "aj",
        "probability",
        "same_chain",
        "source",
        "reference",
        "rc_probability",
        "sigma",
        "epsilon",
        "bond_distance",
        "rep",
        "sigma_prior",
        "epsilon_prior",
        "mg_sigma",
        "mg_epsilon",
        "md_threshold",
        "rc_threshold",
        "learned",
    ]

    with _term.spinner("computing sigma / epsilon"):
        meGO_LJ = set_sig_epsilon(meGO_LJ, parameters)[needed_fields]

    et = time.time()
    _term.timing(et - st)
    st = et

    if parameters.symmetry:
        _term.sub("Apply the defined atomic symmetries")
        with _term.spinner("applying symmetries"):
            meGO_LJ_sym = apply_symmetries(meGO_ensemble, meGO_LJ, parameters.symmetry)
        meGO_LJ = pd.concat([meGO_LJ, meGO_LJ_sym])
        meGO_LJ.reset_index(inplace=True)
        if local_mode != "all":
            meGO_LJ = meGO_LJ.loc[~_local_frozen_mask(meGO_LJ)]
        if hbond_freeze:
            meGO_LJ = meGO_LJ.loc[~_hbond_frozen_mask(meGO_LJ, prior, sb_resnum, min_sep)]
        et = time.time()
        _term.timing(et - st)
        st = et

    _term.sub("Merging multiple states (training, symmetries, inter/intra)")

    with _term.spinner("merging"):
        # Merging priority: learned > not learned, attractive > repulsive, shorter > longer, stronger attractive, weaker repulsive
        meGO_LJ["type"] = np.sign(meGO_LJ["epsilon"])
        meGO_LJ.sort_values(
            by=["ai", "aj", "same_chain", "learned", "type", "sigma", "epsilon"],
            ascending=[True, True, True, False, False, True, False],
            inplace=True,
        )
        meGO_LJ = meGO_LJ.drop_duplicates(subset=["ai", "aj", "same_chain"], keep="first")

        # Remove contacts that are effectively identical to the default MG prior
        meGO_LJ = meGO_LJ.loc[
            ~(
                (meGO_LJ["epsilon"] > 0)
                & (meGO_LJ["mg_epsilon"] > 0)
                & ((abs(meGO_LJ["epsilon"] - meGO_LJ["mg_epsilon"]) / meGO_LJ["mg_epsilon"]) < config.learn_tolerance)
                & ((abs(meGO_LJ["sigma"] - meGO_LJ["mg_sigma"]) / meGO_LJ["mg_sigma"]) < config.learn_tolerance)
                & ((meGO_LJ["bond_distance"] > config.bond14_separation) | (~meGO_LJ["same_chain"]))
            )
        ]
        meGO_LJ = meGO_LJ.loc[
            ~(
                (meGO_LJ["epsilon"] < 0)
                & (meGO_LJ["mg_epsilon"] < 0)
                & (
                    (abs(meGO_LJ["epsilon"] - meGO_LJ["mg_epsilon"]) / abs(meGO_LJ["mg_epsilon"]))
                    < config.learn_tolerance
                )
                & ((meGO_LJ["bond_distance"] > config.bond14_separation) | (~meGO_LJ["same_chain"]))
                & ~((meGO_LJ["bond_distance"] <= config.max_bond_separation) & (meGO_LJ["same_chain"]))
            )
        ]

        meGO_LJ = meGO_LJ[needed_fields]

        stat_str = fileio.print_stats(meGO_LJ)

        if hbond_freeze:
            meGO_LJ = pd.concat([meGO_LJ, _hbond_prior_companions(meGO_LJ, prior)[needed_fields]])

        # --- identify duplicates ON ORIGINAL DATA ---
        dup_mask = meGO_LJ.duplicated(subset=["ai", "aj"], keep=False)
        # --- pairs section (ONLY intramolecular interactions from duplicated pairs) ---
        dup_df = meGO_LJ.loc[dup_mask].copy()
        meGO_LJ_14 = (
            dup_df.assign(_priority=~dup_df["same_chain"])
            .sort_values(["ai", "aj", "_priority"])
            .drop_duplicates(subset=["ai", "aj"], keep="first")
            .drop(columns="_priority")
        )
        # ffnonbonded: prioritise intermolecular interactions on duplicate ai/aj
        meGO_LJ.sort_values(by=["ai", "aj", "same_chain"], ascending=[True, True, True], inplace=True)
        meGO_LJ = meGO_LJ.drop_duplicates(subset=["ai", "aj"], keep="first")

        # no cross interactions
        common = meGO_LJ_14["molecule_name_ai"].cat.categories.union(meGO_LJ_14["molecule_name_aj"].cat.categories)
        meGO_LJ_14["molecule_name_ai"] = meGO_LJ_14["molecule_name_ai"].cat.set_categories(common)
        meGO_LJ_14["molecule_name_aj"] = meGO_LJ_14["molecule_name_aj"].cat.set_categories(common)
        meGO_LJ_14 = meGO_LJ_14[meGO_LJ_14["molecule_name_ai"] == meGO_LJ_14["molecule_name_aj"]]
        # intramolecular interactions within few bonds should be move in pairs
        copy_intra = meGO_LJ.loc[(meGO_LJ["same_chain"]) & (meGO_LJ["bond_distance"] <= config.max_bond_separation)]
        meGO_LJ_14 = pd.concat([meGO_LJ_14, copy_intra], axis=0, sort=False, ignore_index=True)
        meGO_LJ = meGO_LJ.loc[~((meGO_LJ["same_chain"]) & (meGO_LJ["bond_distance"] <= config.max_bond_separation))]

        if parameters.force_split:
            split_ii = meGO_LJ.loc[(meGO_LJ["same_chain"])]
            meGO_LJ_14 = pd.concat([meGO_LJ_14, split_ii], axis=0, sort=False, ignore_index=True)
            meGO_LJ = meGO_LJ.loc[(~meGO_LJ["same_chain"])]

        # The MG prior interactions are NOT expanded here (that would be an
        # N x N DataFrame): the learned interactions become sparse overrides
        # of the type-level prior, combined at writing time
        # (nonbonded.NonbondedModel).
        meGO_LJ = resolve_overrides(meGO_LJ, prior)

        _add_c6_c12(meGO_LJ)
        _add_c6_c12(meGO_LJ_14)

    et = time.time()
    _term.timing(et - st)

    return meGO_LJ, meGO_LJ_14, stat_str
