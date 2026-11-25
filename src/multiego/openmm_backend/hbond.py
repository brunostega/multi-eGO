"""
Geometric backbone hydrogen bond in OpenMM (models generated with ``--hbond``).

Every pair of a backbone amide H (donor, with the N of its residue) and a
backbone carbonyl O (acceptor, with the C of its residue) is described by one
``CustomHbondForce`` instead of the H-O entry of the Lennard-Jones table:

    E = c12p/r^12 - c6p/r^6 + E_hb                      |dres| >= min_sep
    E = c12/r^12 - c6/r^6                               |dres| <  min_sep

    E_hb = -eps * f(r) * A_don(theta_NHO) * A_acc(theta_HOC)
    f(r) = 1 for r <= r0,  exp(-(r - r0)^2 / (2 width^2)) beyond
    A(theta) = exp(-(theta - theta0)^2 / (2 sigma^2)) for theta > 90 deg, else 0
    eps = epsilon_local    for min_sep <= |dres| < nonlocal_separation
          epsilon_nonlocal for |dres| >= nonlocal_separation or different chains

with r the H...O distance, theta_NHO the N-H...O angle, theta_HOC the H...O=C
angle and dres the residue separation (see ``multiego.hbond.HBondSettings``).

* **Hydrogen bonds read the prior only.**  ``c6p/c12p`` is the prior of the
  pair (with ``--hbond`` the repulsive polar-backbone rule: c6p = 0), never
  a learned contact, so the training data cannot move the wall or change the
  depth of a hydrogen bond (handoff 4.4).
* **Local pairs keep their Lennard-Jones**, as learned (``c6/c12``: the LJ
  table entry).  It is computed inside this force because the LJ table
  entries of all donor-acceptor type pairs are set to zero: the same pair of
  sb_types is local within a chain and non-local between copies.  Its cutoff
  is the cutoff of this force (0.5 nm); there is no potential shift.
* **Selection by parameters, not exclusions.**  The residue separation comes
  from per-donor/per-acceptor residue indices; copies of molecules are
  separated by a gap of residues so that bonds between chains are always
  non-local.  GPU platforms accept at most 4 exclusions per donor.
* **Exclusions** are only the H-O pairs excluded from the nonbonded force
  *and* within 5 bonds (the bonded terms and the topological ``[ pairs ]``
  describe them): at most two per donor.
* **Learned contacts in** ``[ pairs ]`` (intramolecular contacts written as
  pairs, e.g. with ``--force_split``; handoff 4.6): an H-O pair listed beyond
  5 bonds is not excluded from this force.  If |dres| >= min_sep it is
  removed from the pairs (the hydrogen bond describes it); otherwise the pair
  keeps its learned value through a correction in the pairs force,
  (c6_pair - c6, c12_pair - c12), since this force already adds the table LJ.

GPU platforms skip donor/acceptor blocks of 32 farther apart than the cutoff,
so the cost is about linear in the system size; the CPU platform evaluates
all donor-acceptor pairs.
"""

import numpy as np
import openmm

# Pairs within this many bonds are topological (bonded terms, [ pairs ] of
# the 4-5 bond rule); beyond, a listed pair is a learned contact.
TOPOLOGICAL_BONDS = 5


def find_donors_acceptors(names, resnums):
    """
    Backbone donors (H, N of the same residue) and acceptors (O, C of the same
    residue) of one molecule, 0-based indices within the molecule.

    Returns
    -------
    donors : list of (h, n, resnum)
    acceptors : list of (o, c, resnum)
    """
    names = [str(n) for n in names]
    resnums = [int(r) for r in resnums]
    index = {(r, n): k for k, (r, n) in enumerate(zip(resnums, names))}
    donors, acceptors = [], []
    for k, (r, n) in enumerate(zip(resnums, names)):
        if n == "H" and (r, "N") in index:
            donors.append((k, index[(r, "N")], r))
        elif n == "O" and (r, "C") in index:
            acceptors.append((k, index[(r, "C")], r))
    return donors, acceptors


def _f(x):
    return repr(float(x))


def energy_expression(settings):
    """CustomHbondForce expression; d1 = H, d2 = N, a1 = O, a2 = C."""
    half_pi = _f(np.pi / 2.0)

    def angular(name, angle, theta0, sigma):
        t0 = _f(np.deg2rad(theta0))
        den = _f(2.0 * np.deg2rad(sigma) ** 2)
        return [f"{name} = step(t_{name} - {half_pi})*exp(-(t_{name} - {t0})^2/{den})", f"t_{name} = {angle}"]

    local = f"step({_f(settings.min_residue_separation - 0.5)} - abs(dres - ares))"
    far = f"step(abs(dres - ares) - {_f(settings.nonlocal_separation - 0.5)})"
    return ";".join(
        [
            f"select({local}, c12/r^12 - c6/r^6, c12p/r^12 - c6p/r^6 + ehb)",
            f"ehb = -select({far}, {_f(settings.epsilon_nonlocal)}, {_f(settings.epsilon_local)})"
            f"*exp(-max(r - {_f(settings.r0)}, 0)^2/{_f(2.0 * settings.width**2)})*aNHO*aHOC",
            *angular("aNHO", "angle(d2, d1, a1)", settings.donor_theta0, settings.donor_width),
            *angular("aHOC", "angle(d1, a1, a2)", settings.acceptor_theta0, settings.acceptor_width),
            "r = distance(d1, a1)",
            "c6 = c6hb(dt, at)",
            "c12 = c12hb(dt, at)",
            "c6p = c6hbp(dt, at)",
            "c12p = c12hbp(dt, at)",
        ]
    )


def classify_pairs(pairs, donors, acceptors, resnums, within5, settings):
    """
    Split the ``[ pairs ]`` of one molecule for the hydrogen-bond term.

    Parameters
    ----------
    pairs : list of (i, j, c6, c12)
    donors, acceptors : from :func:`find_donors_acceptors`
    resnums : residue number of each atom
    within5 : set of (i, j), i < j: atom pairs within TOPOLOGICAL_BONDS bonds
    settings : HBondSettings

    Returns
    -------
    kept : list of (i, j, c6, c12)
        Pairs left as they are.
    local_learned : list of (i, j, c6, c12)
        Learned local H-O pairs: their values must be corrected by the LJ table
        entry, which the hydrogen-bond force adds.
    dropped : list of (i, j)
        Learned H-O pairs described by the hydrogen bond (|dres| >= min_sep).
    """
    h_atoms = {h for h, _, _ in donors}
    o_atoms = {o for o, _, _ in acceptors}
    kept, local_learned, dropped = [], [], []
    for i, j, c6, c12 in pairs:
        is_ho = (i in h_atoms and j in o_atoms) or (j in h_atoms and i in o_atoms)
        if not is_ho or (min(i, j), max(i, j)) in within5:
            kept.append((i, j, c6, c12))
        elif abs(int(resnums[i]) - int(resnums[j])) >= settings.min_residue_separation:
            dropped.append((i, j))
        else:
            local_learned.append((i, j, c6, c12))
    return kept, local_learned, dropped


def make_force(settings, c6, c12, c6p, c12p, donor_types, acceptor_types):
    """
    The CustomHbondForce with its tables; donors, acceptors and exclusions
    are added by the builder.

    ``c6, c12``: LJ tables (before the donor-acceptor entries are zeroed);
    ``c6p, c12p``: prior tables; ``donor_types``/``acceptor_types``: sorted
    type indices.  Returns (force, donor type -> row, acceptor type -> column).
    """
    d_index = {t: k for k, t in enumerate(donor_types)}
    a_index = {t: k for k, t in enumerate(acceptor_types)}
    nd, na = len(donor_types), len(acceptor_types)
    force = openmm.CustomHbondForce(energy_expression(settings))
    force.addPerDonorParameter("dt")
    force.addPerDonorParameter("dres")
    force.addPerAcceptorParameter("at")
    force.addPerAcceptorParameter("ares")
    sub = np.ix_(donor_types, acceptor_types)
    # Discrete2DFunction: the first index (donor type) varies fastest.
    for name, table in (("c6hb", c6), ("c12hb", c12), ("c6hbp", c6p), ("c12hbp", c12p)):
        values = np.asarray(table, float)[sub].ravel(order="F").tolist()
        force.addTabulatedFunction(name, openmm.Discrete2DFunction(nd, na, values))
    force.setNonbondedMethod(openmm.CustomHbondForce.CutoffPeriodic)
    force.setCutoffDistance(settings.cutoff)
    return force, d_index, a_index
