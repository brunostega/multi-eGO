"""
Geometric backbone hydrogen-bond term (OpenMM backend only).

What it does
------------
In the multi-eGO force field a backbone hydrogen bond N-H...O=C is described
only by an isotropic Lennard-Jones attraction between H and O.  This module
replaces that LJ interaction, for every pair of backbone amide H (donor) and
backbone carbonyl O (acceptor), by a term that also depends on the geometry:

    E = E_rep(r) + E_hb(r, theta_NHO, theta_HOC)

    E_rep = c12/r^12 - c6/r^6 + eps                       for r < r_min, else 0
    E_hb  = -eps_hb * f(r) * A_don(theta_NHO) * A_acc(theta_HOC)
    f(r)      = 1 for r <= r0,  exp(-(r - r0)^2 / (2 w^2)) for r > r0
    A(theta)  = exp(-(theta - theta_0)^2 / (2 sigma^2))   for theta > 90 deg,
                0 otherwise

with r the H...O distance, theta_NHO the N-H...O angle and theta_HOC the
H...O=C angle.

Where the angular optima come from
----------------------------------
The angular factors are *not* peaked at 180 degrees for both angles.  A
survey of the backbone hydrogen bonds of the reference structures shipped
with multi-eGO (tools/helix/hbond_geometry.py; 94 alpha and 29 beta hydrogen
bonds in GB1, lysozyme and Abeta42) gives:

===========  ==========  =============  =============  ============
contact      r (A)       N-H...O (deg)  H...O=C (deg)  out of plane
===========  ==========  =============  =============  ============
alpha        2.2 +- 0.2  152 +- 10      142 +- 10      31 deg
beta         2.0 +- 0.1  158 +-  9      152 +- 12      16 deg
3-10         2.4 +- 0.2  140 +- 14      101 +-  8      76 deg
===========  ==========  =============  =============  ============

The donor angle is close to linear, as electrostatics require, so
``A_don`` is centred on 180 degrees and is simply broad enough to cover the
observed 150-165 range.  The *acceptor* angle is not: the carbonyl lone pairs
point away from the C=O axis, so an approach along that axis falls between
them.  ``A_acc`` is therefore centred on 147 degrees, between the alpha and
beta optima.

This matters beyond the depth of the well.  With ``cos^2`` for both angles --
i.e. assuming both optima at 180 -- an alpha hydrogen bond retains only
0.62 x 0.78 = 0.48 of the well while a beta one retains 0.78 x 0.86 = 0.67:
the term is 40% stronger on beta sheets than on alpha helices, purely as an
artefact of the functional form.  With the measured optima the two become
0.76 and 0.84, so most of that bias disappears and, at the same time, far
less of ``eps_hb`` is wasted.

The 3-10 contacts are rejected by ``A_acc`` (101 degrees is 1.8 sigma away),
which is the intended behaviour: they are a different, much weaker
interaction.  c6 and c12 are the
LJ parameters this H-O pair had in the model (mg prior or learned), from
which eps = c6^2/(4 c12) and r_min = (2 c12/c6)^(1/6).

* E_rep is the repulsive part of that LJ in the Weeks-Chandler-Andersen
  decomposition: the original wall, unchanged.
* E_hb replaces the attraction.  By default eps_hb = eps, r0 = r_min and
  w = 0.193 r_min, the width at which the Gaussian reaches half depth at the
  same distance as the LJ well (1.227 r_min).  Hence, for an ideal (linear)
  geometry, E is *identical* to the original LJ for r <= r_min and follows it
  closely over the hydrogen-bond range beyond (-0.87 vs -0.81 eps at
  1.1 r_min, -0.30 vs -0.37 eps at 1.3 r_min), while the long -1/r^6 tail is
  cut (-0.03 vs -0.17 eps at 1.5 r_min).  The real change is that the
  attraction now vanishes for non-hydrogen-bonding orientations.
  (Matching the LJ curvature at the minimum instead would give a much
  narrower well, w = r_min/sqrt(72) = 0.118 r_min, missing typical
  hydrogen-bond distances.  A full rather than half Gaussian would add its
  curvature to that of the wall for r < r_min: hence the flat inner part.)
  eps_hb, r0 and w can be set to fixed values instead (``HBondSettings``).
* A pair whose LJ is purely repulsive (c6 = 0) keeps its c12/r^12 repulsion
  and gets no attraction.

Local and non-local bonds
-------------------------
The depth of the well can differ between *local* bonds (donor and acceptor
3-4 residues apart: alpha and 3-10 helices, turns) and *non-local* ones
(5 or more residues apart, or on different chains: beta sheets, tertiary and
intermolecular contacts), through ``HBondSettings.epsilon_nonlocal``.

Why: with a single depth calibrated on helices, the model forms far too much
beta structure.  Per formed bond the two kinds of contact get the same
energy (the angular factors treat them alike, ~0.8 of the well), but they
cost the chain very different amounts of order.  A helix must fix one
residue in the narrow helical region per bond (~8.3 kJ/mol of local free
energy, from the experimental coil distributions reproduced by the CMAP; the
same as -RT ln v with v = 0.036); non-local pairing between extended segments
costs ~2 kJ/mol per fixed residue.  Real chains additionally charge non-local
bonds for things this model lacks (competition with water, backbone
desolvation, loop entropy, register) and reinforce helical bonds through
cooperative polarization.  A weaker non-local depth stands in for both.

Which pairs
-----------
Donors are the backbone amide hydrogens (atom name ``H``, with the ``N`` of
the same residue); acceptors the backbone carbonyl oxygens (atom name ``O``,
with the ``C`` of the same residue).  Pairs excluded from the nonbonded
interactions (within ``nrexcl`` bonds, or in ``[ pairs ]``) are excluded here
too, so that they keep exactly their previous treatment.  All other H-O pairs,
within and between molecules, use this term; their entries of the LJ type
table are set to zero.

The GROMACS backend does not have this term: the GROMACS files always contain
the original isotropic H-O Lennard-Jones interaction.

Performance note
----------------
OpenMM's CustomHbondForce has no neighbour list: it evaluates all
donor-acceptor pairs (with a cut-off).  This is cheap for one protein but
grows as (number of copies)^2 for many-copy systems.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import openmm as mm


@dataclass
class HBondSettings:
    """
    Parameters of the hydrogen-bond term.  ``None`` means "derived from the
    LJ parameters of each H-O pair" (see the module docstring).
    """

    epsilon: float = None  # kJ/mol, depth of the well (None: from the LJ pair)
    r0: float = 0.21  # nm, position of the well (measured alpha/beta mean)
    width: float = 0.035  # nm, Gaussian width of the well
    donor_theta0: float = 180.0  # deg, optimum of the N-H...O angle
    donor_width: float = 40.0  # deg
    acceptor_theta0: float = 147.0  # deg, optimum of the H...O=C angle
    acceptor_width: float = 25.0  # deg
    cutoff: float = 0.5  # nm, H...O cut-off of the term
    # Minimum separation in residues between donor and acceptor.  The default
    # of 3 removes the C5/C7 (gamma-turn) and i->i+2 interactions, which are
    # local and already described by the backbone dihedrals / CMAP, and keeps
    # the 3-10 (i->i+3) and alpha (i->i+4) hydrogen bonds.  This keeps the
    # local conformational preference and the hydrogen-bond term independent,
    # so that they can be calibrated separately.
    min_residue_separation: int = 3
    # Depth of the well for NON-LOCAL hydrogen bonds: donor and acceptor at
    # least nonlocal_separation residues apart along the same chain, or on
    # different chains (beta sheets, tertiary and intermolecular contacts).
    # None: the same depth as the local bonds (``epsilon``).  Local bonds are
    # those between min_residue_separation and nonlocal_separation - 1
    # residues apart: with the defaults, i->i+3 (3-10, turns) and i->i+4
    # (alpha).  See the module docstring ("Local and non-local bonds").
    epsilon_nonlocal: float = None  # kJ/mol
    nonlocal_separation: int = 5

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))


def find_donors_acceptors(atoms):
    """
    Backbone donors (H, N) and acceptors (O, C) of one molecule type.

    Parameters
    ----------
    atoms : dict of arrays
        The ``atoms`` table of a molecule in the parameter file (name, resnum).

    Returns
    -------
    donors, acceptors : dict
        ``{"h": ..., "n": ...}`` and ``{"o": ..., "c": ...}`` arrays of atom
        indices (0-based within the molecule).  Residues without an amide H
        (proline, N-terminus) have no donor.
    """
    names = np.asarray(atoms["name"]).astype(str)
    resnum = np.asarray(atoms["resnum"]).astype(int)
    index = {(r, n): k for k, (r, n) in enumerate(zip(resnum, names))}

    donors = {"h": [], "n": []}
    acceptors = {"o": [], "c": []}
    for k, (r, n) in enumerate(zip(resnum, names)):
        if n == "H" and (r, "N") in index:
            donors["h"].append(k)
            donors["n"].append(index[(r, "N")])
        elif n == "O" and (r, "C") in index:
            acceptors["o"].append(k)
            acceptors["c"].append(index[(r, "C")])
    return {k: np.array(v, dtype=int) for k, v in donors.items()}, {
        k: np.array(v, dtype=int) for k, v in acceptors.items()
    }


def energy_expression(settings):
    """
    Energy expression of the CustomHbondForce (see the module docstring).

    Particles: d1 = H, d2 = N (donor); a1 = O, a2 = C (acceptor).
    ``c6hb``/``c12hb`` are tables indexed by the donor and acceptor type
    (per-donor parameter ``dtype``, per-acceptor parameter ``atype``).
    """
    eps_local = "eps" if settings.epsilon is None else repr(float(settings.epsilon))
    eps_nonlocal = eps_local if settings.epsilon_nonlocal is None else repr(float(settings.epsilon_nonlocal))
    # residues at least nonlocal_separation apart, or on different chains
    # (the builder separates chains by a gap of at least that many residues)
    nonlocal_switch = f"step(abs(dres - ares) - {float(settings.nonlocal_separation) - 0.5!r})"
    eps_hb = f"select({nonlocal_switch}, {eps_nonlocal}, {eps_local})"
    half_pi = repr(float(np.pi / 2.0))

    def angular(name, angle_expression, theta0, sigma):
        """Gaussian angular factor, zero below 90 degrees (wrong side of the atom)."""
        # float(): a numpy scalar would print as "np.float64(...)" in the expression
        theta0 = float(np.deg2rad(float(theta0)))
        denominator = float(2.0 * np.deg2rad(float(sigma)) ** 2)
        return [
            f"{name} = step(t_{name} - {half_pi}) * exp(-(t_{name} - {theta0!r})^2/{denominator!r})",
            f"t_{name} = {angle_expression}",
        ]

    # Pairs closer than min_residue_separation along a chain keep the plain
    # Lennard-Jones interaction they had before the term replaced it (the LJ
    # table entry of every donor-acceptor pair is zeroed, see add_hbond_force).
    # The rule is applied here, through the per-donor/per-acceptor residue
    # index, rather than with exclusions: GPU platforms accept at most four
    # exclusions per donor.
    local = f"step({float(settings.min_residue_separation) - 0.5!r} - abs(dres - ares))"
    return ";".join(
        [
            f"select({local}, c12/r^12 - c6/r^6, erep + ehb)",
            # attractive LJ (c6 > 0): WCA repulsion; repulsive LJ: plain c12/r^12
            "erep = select(c6, step(rmin - r)*(c12/r^12 - c6/r^6 + eps), c12/r^12)",
            # the angular attraction exists only for attractive pairs
            # flat for r <= r0, Gaussian beyond (see the module docstring)
            f"ehb = -select(c6, 1, 0) * ({eps_hb}) * exp(-max(r - {float(settings.r0)!r}, 0)^2"
            f"/{float(2.0 * float(settings.width) ** 2)!r}) * aNHO * aHOC",
            *angular("aNHO", "angle(d2, d1, a1)", settings.donor_theta0, settings.donor_width),
            *angular("aHOC", "angle(d1, a1, a2)", settings.acceptor_theta0, settings.acceptor_width),
            "r = distance(d1, a1)",
            # LJ well of this pair; the max() avoid 0/0 when c6 = 0 (unused then)
            "eps = c6^2/(4*max(c12, 1e-30))",
            "rmin = (2*c12/max(c6, 1e-30))^(1/6)",
            "c6 = c6hb(dtype, atype)",
            "c12 = c12hb(dtype, atype)",
        ]
    )


def add_hbond_force(system, settings, c6_table, c12_table, donors, acceptors, excluded_pairs, residue_of_atom=None):
    """
    Add the hydrogen-bond force and remove the H-O entries it replaces from
    the LJ type tables.

    Parameters
    ----------
    system : openmm.System
    settings : HBondSettings
    c6_table, c12_table : np.ndarray
        Dense LJ type tables (modified in place: donor-acceptor entries are
        set to zero, since this force now describes those pairs).
    donors, acceptors : list of (h, n, type) / (o, c, type)
        Global particle indices and LJ type of every donor/acceptor.
    excluded_pairs : set of (i, j)
        Particle pairs excluded from the nonbonded force (i < j).
    residue_of_atom : dict, optional
        Particle index -> global residue index.  Indices must be consecutive
        along a chain and differ by at least ``nonlocal_separation`` (and
        ``min_residue_separation``) between different molecule copies (the
        builder leaves such a gap), so that bonds between chains are
        non-local.  Used
        to apply ``settings.min_residue_separation``; without it the rule is
        not applied.

    Returns
    -------
    openmm.CustomHbondForce
    """
    d_types = sorted({t for _, _, t in donors})
    a_types = sorted({t for _, _, t in acceptors})
    d_index = {t: k for k, t in enumerate(d_types)}
    a_index = {t: k for k, t in enumerate(a_types)}

    # Sub-tables of the LJ parameters of the donor/acceptor type pairs.
    sub_c6 = c6_table[np.ix_(d_types, a_types)]
    sub_c12 = c12_table[np.ix_(d_types, a_types)]

    force = mm.CustomHbondForce(energy_expression(settings))
    force.setName("hbonds")
    force.addPerDonorParameter("dtype")
    force.addPerDonorParameter("dres")
    force.addPerAcceptorParameter("atype")
    force.addPerAcceptorParameter("ares")
    nd, na = len(d_types), len(a_types)
    # Discrete2DFunction: first index (donor type) varies fastest.
    force.addTabulatedFunction("c6hb", mm.Discrete2DFunction(nd, na, sub_c6.ravel(order="F")))
    force.addTabulatedFunction("c12hb", mm.Discrete2DFunction(nd, na, sub_c12.ravel(order="F")))
    force.setNonbondedMethod(mm.CustomHbondForce.CutoffPeriodic)
    force.setCutoffDistance(settings.cutoff)

    # Without residue information no pair is "local": give donors and
    # acceptors residue indices that are always far apart.
    if residue_of_atom is None:
        donor_residue, acceptor_residue = (lambda atom: 0.0), (lambda atom: 1.0e6)
    else:
        donor_residue = acceptor_residue = lambda atom: float(residue_of_atom[atom])

    donor_of_h = {}
    for h, n, t in donors:
        donor_of_h[h] = force.addDonor(h, n, -1, [float(d_index[t]), donor_residue(h)])
    acceptor_of_o = {}
    for o, c, t in acceptors:
        acceptor_of_o[o] = force.addAcceptor(o, c, -1, [float(a_index[t]), acceptor_residue(o)])
    # Only pairs excluded from the non-bonded force are excluded here (they are
    # described by the bonded terms and [ pairs ]).  For backbone H and O
    # these are within one residue, so there are at most a couple per donor:
    # well within the limit of four exclusions per donor of the GPU platforms.
    exclusions = set()
    for i, j in excluded_pairs:
        if i in donor_of_h and j in acceptor_of_o:
            exclusions.add((donor_of_h[i], acceptor_of_o[j]))
        elif j in donor_of_h and i in acceptor_of_o:
            exclusions.add((donor_of_h[j], acceptor_of_o[i]))
    per_donor = np.bincount([d for d, _ in exclusions], minlength=len(donor_of_h)) if exclusions else [0]
    if max(per_donor) > 4:
        raise ValueError(
            f"a hydrogen-bond donor has {max(per_donor)} exclusions; GPU platforms of OpenMM accept at most 4"
        )
    for d, a in sorted(exclusions):
        force.addExclusion(d, a)

    # These pairs are now described by the hydrogen-bond force.
    for table in (c6_table, c12_table):
        table[np.ix_(d_types, a_types)] = 0.0
        table[np.ix_(a_types, d_types)] = 0.0
    system.addForce(force)
    return force
