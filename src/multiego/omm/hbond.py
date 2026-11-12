"""
Geometric backbone hydrogen-bond term (OpenMM backend only).

What it does
------------
In the multi-eGO force field a backbone hydrogen bond N-H...O=C is described
only by an isotropic Lennard-Jones attraction between H and O.  This module
replaces that LJ interaction, for every pair of backbone amide H (donor) and
backbone carbonyl O (acceptor), by a term that also depends on the geometry:

    E = E_rep(r) + E_hb(r, theta_NHO, theta_HOC)

    E_rep = c12/r^12 - c6/r^6 + eps     for r < r_min,   0 otherwise
    E_hb  = -eps_hb * f(r) * A(theta_NHO) * A(theta_HOC)
    f(r)  = 1 for r <= r0,  exp(-(r - r0)^2 / (2 w^2)) for r > r0
    A(theta) = cos(theta)^2  if theta > 90 deg,  0 otherwise

with r the H...O distance, theta_NHO the N-H...O angle and theta_HOC the
H...O=C angle (both 180 deg for a linear hydrogen bond).  c6 and c12 are the
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

# Default Gaussian width, in units of r_min.  The LJ well reaches half its
# depth (-eps/2) on the outer side at r_half = r_min (1 - 1/sqrt(2))^(-1/6)
# = 1.227 r_min; a Gaussian of width w reaches half depth at
# r0 + w sqrt(2 ln 2).  Equating the two gives w = 0.193 r_min, so that the
# angular well covers the same distance range as the LJ well it replaces.
HALF_WIDTH_RATIO = float(((1.0 - 2.0**-0.5) ** (-1.0 / 6.0) - 1.0) / np.sqrt(2.0 * np.log(2.0)))


@dataclass
class HBondSettings:
    """
    Parameters of the hydrogen-bond term.  ``None`` means "derived from the
    LJ parameters of each H-O pair" (see the module docstring).
    """

    epsilon: float = None  # kJ/mol, depth of the angular well
    r0: float = None  # nm, position of the well
    width: float = None  # nm, Gaussian width of the well
    cutoff: float = 0.5  # nm, H...O cut-off of the term
    # Minimum separation in residues between donor and acceptor.  The default
    # of 3 removes the C5/C7 (gamma-turn) and i->i+2 interactions, which are
    # local and already described by the backbone dihedrals / CMAP, and keeps
    # the 3-10 (i->i+3) and alpha (i->i+4) hydrogen bonds.  This keeps the
    # local conformational preference and the hydrogen-bond term independent,
    # so that they can be calibrated separately.
    min_residue_separation: int = 3

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
    eps_hb = "eps" if settings.epsilon is None else repr(float(settings.epsilon))
    r0 = "rmin" if settings.r0 is None else repr(float(settings.r0))
    width = f"rmin*{HALF_WIDTH_RATIO!r}" if settings.width is None else repr(float(settings.width))
    return ";".join(
        [
            "erep + ehb",
            # attractive LJ (c6 > 0): WCA repulsion; repulsive LJ: plain c12/r^12
            "erep = select(c6, step(rmin - r)*(c12/r^12 - c6/r^6 + eps), c12/r^12)",
            # the angular attraction exists only for attractive pairs
            # flat for r <= r0, Gaussian beyond (see the module docstring)
            f"ehb = -select(c6, 1, 0) * {eps_hb} * exp(-max(r - {r0}, 0)^2/(2*w^2)) * aNHO * aHOC",
            f"w = {width}",
            "aNHO = step(-cNHO)*cNHO^2",
            "cNHO = cos(angle(d2, d1, a1))",
            "aHOC = step(-cHOC)*cHOC^2",
            "cHOC = cos(angle(d1, a1, a2))",
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
        Particle index -> (molecule copy, residue number).  Used to apply
        ``settings.min_residue_separation``; without it that rule is skipped.

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
    force.addPerAcceptorParameter("atype")
    nd, na = len(d_types), len(a_types)
    # Discrete2DFunction: first index (donor type) varies fastest.
    force.addTabulatedFunction("c6hb", mm.Discrete2DFunction(nd, na, sub_c6.ravel(order="F")))
    force.addTabulatedFunction("c12hb", mm.Discrete2DFunction(nd, na, sub_c12.ravel(order="F")))
    force.setNonbondedMethod(mm.CustomHbondForce.CutoffPeriodic)
    force.setCutoffDistance(settings.cutoff)

    donor_of_h = {}
    for h, n, t in donors:
        donor_of_h[h] = force.addDonor(h, n, -1, [float(d_index[t])])
    acceptor_of_o = {}
    for o, c, t in acceptors:
        acceptor_of_o[o] = force.addAcceptor(o, c, -1, [float(a_index[t])])
    exclusions = set()
    for i, j in excluded_pairs:
        if i in donor_of_h and j in acceptor_of_o:
            exclusions.add((donor_of_h[i], acceptor_of_o[j]))
        elif j in donor_of_h and i in acceptor_of_o:
            exclusions.add((donor_of_h[j], acceptor_of_o[i]))
    # Donor and acceptor too close along the same chain (see HBondSettings).
    if residue_of_atom is not None and settings.min_residue_separation > 0:
        for h, _, _ in donors:
            chain_h, res_h = residue_of_atom[h]
            for o, _, _ in acceptors:
                chain_o, res_o = residue_of_atom[o]
                if chain_h == chain_o and abs(res_h - res_o) < settings.min_residue_separation:
                    exclusions.add((donor_of_h[h], acceptor_of_o[o]))
    for d, a in sorted(exclusions):
        force.addExclusion(d, a)

    # These pairs are now described by the hydrogen-bond force.
    for table in (c6_table, c12_table):
        table[np.ix_(d_types, a_types)] = 0.0
        table[np.ix_(a_types, d_types)] = 0.0
    system.addForce(force)
    return force
