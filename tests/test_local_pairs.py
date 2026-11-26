"""
--learn_local_pairs: which local pairs (4-5 bonds) depend on a backbone phi/psi torsion.
"""

import types

import pandas as pd
import pytest

from multiego import bonded

# Ace C, O | residue 2: N H CA CB C O, side chain CG CD CE CZ | residue 3: N H CA
ATOMS = [
    (1, "C", 1),
    (2, "O", 1),
    (3, "N", 2),
    (4, "H", 2),
    (5, "CA", 2),
    (6, "CB", 2),
    (7, "C", 2),
    (8, "O", 2),
    (9, "N", 3),
    (10, "H", 3),
    (11, "CA", 3),
    (12, "CG", 2),
    (13, "CD", 2),
    (14, "CE", 2),
    (15, "CZ", 2),
]
BONDS = [(1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7), (7, 8), (7, 9), (9, 10), (9, 11), (6, 12), (12, 13)]
BONDS += [(13, 14), (14, 15)]


def ensemble():
    top = pd.DataFrame(
        {
            "molecule_name": "X",
            "number": [str(n) for n, _, _ in ATOMS],
            "sb_type": [f"{name}_X_{res}" for _, name, res in ATOMS],
            "name": [name for _, name, _ in ATOMS],
            "resnum": [str(res) for _, _, res in ATOMS],
        }
    )
    return types.SimpleNamespace(topology_dataframe=top, bond_pairs={"X": [(str(a), str(b)) for a, b in BONDS]})


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("O_X_1", "C_X_2", True),  # O(i-1)..C(i): phi
        ("H_X_2", "O_X_2", True),  # H(i)..O(i): phi, psi
        ("O_X_1", "O_X_2", True),  # O(i-1)..O(i), 5 bonds: phi, psi
        ("H_X_2", "H_X_3", True),  # H(i)..H(i+1), 5 bonds: phi, psi
        ("CB_X_2", "H_X_3", True),  # CB(i)..H(i+1): psi
        ("O_X_1", "CB_X_2", True),  # O(i-1)..CB(i): phi
        ("CD_X_2", "C_X_2", False),  # CD(i)..C(i): chi1, chi2 only
        ("CB_X_2", "CZ_X_2", False),  # within the side chain
    ],
)
def test_phipsi_dependent_pairs(a, b, expected):
    pairs = bonded.phipsi_dependent_pairs(ensemble())
    assert ((a, b) in pairs) is expected
    assert ((b, a) in pairs) is expected


def test_only_local_pairs():
    pairs = bonded.phipsi_dependent_pairs(ensemble())
    # 1-4 pairs (3 bonds) and pairs beyond max_bond_separation are not included
    assert ("O_X_2", "H_X_3") not in pairs  # 3 bonds
    assert ("O_X_1", "H_X_3") not in pairs  # 6 bonds


@pytest.mark.parametrize("value, ok", [("no_phipsi", True), ("none", True), ("all", True), ("some", False)])
def test_option_validation(value, ok):
    from multiego import arguments

    args = arguments.build_parser().parse_args(["--system", "s", "--egos", "mg"])
    args.input_refs = []
    args = arguments.combine_configurations([{"learn_local_pairs": value}], args, arguments.args_dict_global)
    if ok:
        arguments.validate_args(args)
        assert args.learn_local_pairs == value
    else:
        with pytest.raises(ValueError, match="--learn_local_pairs must be"):
            arguments.validate_args(args)


def test_default_is_no_phipsi():
    from multiego import arguments

    assert arguments.build_parser().parse_args(["--system", "s", "--egos", "mg"]).learn_local_pairs == "no_phipsi"
