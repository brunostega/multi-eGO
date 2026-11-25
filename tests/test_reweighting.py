"""
Reweighting rule of learned contacts (``--reweighting``).

* ``interpolate`` (default, unchanged): eps = eps_p + (eps_0 - eps_p) x
* ``additive``: eps = eps_p + eps_0 x  (eps_0 = largest energy added to the prior)

with x = ln(p_MD / max(p_RC, p_RC^th)) / ln(1 / p_RC^th) and eps_p = max(0, eps_prior).
The thresholds of contacts.initialize_molecular_contacts follow the rule; the
repulsive rule is the same for both.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from multiego.contacts import initialize_molecular_contacts
from multiego.lj import set_sig_epsilon
from multiego.model_config import config

EPS_0 = 0.3
EPS_MIN = config.epsilon_min


def thresholds(eps_prior, rule, md_threshold=0.01):
    """rc_threshold and limit_rc_att for one contact with the given prior."""
    # two learned contacts whose cumulative mass puts md_threshold at the second one
    contact = pd.DataFrame({"probability": [1.0, md_threshold], "learned": [True, True], "cutoff": [0.75, 0.75]})
    prior = pd.DataFrame({"epsilon_prior": [eps_prior] * 2, "rc_learned": [True, True]})
    args = argparse.Namespace(reweighting=rule)
    out = initialize_molecular_contacts(contact, prior, args, {"reference": "ref", "epsilon": EPS_0})
    assert out["md_threshold"].iloc[0] == pytest.approx(md_threshold)
    return float(out["rc_threshold"].iloc[0]), float(out["limit_rc_att"].iloc[0])


def learn(p_md, p_rc, eps_prior, rule, md_threshold=0.01):
    """epsilon/sigma of one contact after set_sig_epsilon."""
    rc_th, limit = thresholds(eps_prior, rule, md_threshold)
    row = pd.DataFrame(
        [
            {
                "probability": p_md,
                "rc_probability": p_rc,
                "md_threshold": md_threshold,
                "rc_threshold": rc_th,
                "limit_rc_att": limit,
                "epsilon_prior": eps_prior,
                "sigma_prior": 0.35,
                "epsilon_0": EPS_0,
                "distance": 0.40,
                "rc_distance": 0.38,
                "rep": 1e-6,
                "learned": 0,
            }
        ]
    )
    out = set_sig_epsilon(row, argparse.Namespace(reweighting=rule))
    return out.iloc[0], rc_th


def formula(p_md, p_rc, eps_prior, rc_th, rule):
    eps_p = max(0.0, eps_prior)
    x = np.log(p_md / max(p_rc, rc_th)) / np.log(1.0 / rc_th)
    return eps_p + (EPS_0 if rule == "additive" else EPS_0 - eps_p) * x


@pytest.mark.parametrize("eps_prior", [0.0, -1e-7, 0.12, 0.2])
@pytest.mark.parametrize("rule", ["interpolate", "additive"])
def test_threshold_contact_gets_epsilon_min(eps_prior, rule):
    """p_MD = p_MD^th against p_RC = p_RC^th lands exactly on epsilon_min (definition of p_RC^th)."""
    rc_th, limit = thresholds(eps_prior, rule)
    assert formula(0.01, rc_th, eps_prior, rc_th, rule) == pytest.approx(EPS_MIN)
    # limit_rc_att is the ratio at which epsilon = epsilon_min
    if not (eps_prior < 0 and limit == 1.0):
        assert formula(limit * rc_th, rc_th, eps_prior, rc_th, rule) == pytest.approx(EPS_MIN)


@pytest.mark.parametrize("eps_prior", [0.0, 0.12, 0.2])
def test_additive_largest_correction_is_epsilon_0(eps_prior):
    rc_th, _ = thresholds(eps_prior, "additive")
    assert formula(1.0, rc_th * 1e-3, eps_prior, rc_th, "additive") == pytest.approx(eps_prior + EPS_0)


@pytest.mark.parametrize("eps_prior", [0.0, -1e-7])
def test_repulsive_prior_rules_identical(eps_prior):
    """With a repulsive prior (zero energy) the two rules give the same thresholds and epsilon."""
    a, b = thresholds(eps_prior, "interpolate"), thresholds(eps_prior, "additive")
    assert a == pytest.approx(b)
    for p_md, p_rc in [(0.9, 0.001), (0.5, 0.1), (0.05, 0.2)]:
        ea, _ = learn(p_md, p_rc, eps_prior, "interpolate")
        eb, _ = learn(p_md, p_rc, eps_prior, "additive")
        assert ea["epsilon"] == pytest.approx(eb["epsilon"])
        assert ea["learned"] == eb["learned"]


def test_default_is_interpolate():
    """Without the argument (older callers), the current rule is used."""
    rc_a, lim_a = thresholds(0.2, "interpolate")
    contact = pd.DataFrame({"probability": [1.0, 0.01], "learned": [True, True], "cutoff": [0.75, 0.75]})
    prior = pd.DataFrame({"epsilon_prior": [0.2] * 2, "rc_learned": [True, True]})
    out = initialize_molecular_contacts(contact, prior, argparse.Namespace(), {"reference": "r", "epsilon": EPS_0})
    assert float(out["rc_threshold"].iloc[0]) == pytest.approx(rc_a)


def test_additive_branches_with_attractive_prior():
    eps_prior = 0.2
    # strong training excess: attractive, additive value above the prior
    row, rc_th = learn(0.9, 0.01, eps_prior, "additive")
    assert row["learned"] == 1 and row["epsilon"] == pytest.approx(formula(0.9, 0.01, eps_prior, rc_th, "additive"))
    assert row["epsilon"] > eps_prior
    assert row["sigma"] == pytest.approx(0.40 / 2 ** (1 / 6))
    # training a bit below the reference: still attractive, weaker than the prior
    row, rc_th = learn(0.08, 0.1, eps_prior, "additive")
    eps = formula(0.08, 0.1, eps_prior, rc_th, "additive")
    assert EPS_MIN < eps < eps_prior
    assert row["learned"] == 1 and row["epsilon"] == pytest.approx(eps)
    # training well below the reference: epsilon <= epsilon_min -> repulsive, current c12 rule
    row, rc_th = learn(0.02, 0.9, eps_prior, "additive")
    assert formula(0.02, 0.9, eps_prior, rc_th, "additive") <= EPS_MIN
    assert row["learned"] == 1
    assert row["epsilon"] == pytest.approx(-1e-6 * (0.40 / 0.38) ** 12)
    # training below its threshold: the prior is kept
    row, _ = learn(0.005, 0.5, eps_prior, "additive")
    assert row["learned"] == 0 and row["epsilon"] == pytest.approx(eps_prior)


def test_interpolate_unchanged():
    """The default rule reproduces the formulas of the code before --reweighting."""
    eps_prior, md = 0.12, 0.01
    rc_th, limit = thresholds(eps_prior, "interpolate", md)
    assert rc_th == pytest.approx(md ** ((EPS_0 - eps_prior) / (EPS_0 - EPS_MIN)))
    assert limit == pytest.approx(rc_th ** ((eps_prior - EPS_MIN) / (EPS_0 - eps_prior)))
    row, _ = learn(0.9, 0.01, eps_prior, "interpolate", md)
    expected = eps_prior - (EPS_0 - eps_prior) / np.log(rc_th) * np.log(0.9 / max(0.01, rc_th))
    assert row["epsilon"] == pytest.approx(expected)
