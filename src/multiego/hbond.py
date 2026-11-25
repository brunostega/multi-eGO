"""
Geometric backbone hydrogen bond (``--hbond``): generation-side definitions.

With ``--hbond`` the model describes a backbone hydrogen bond N-H...O=C with
an explicit, orientation-dependent term (OpenMM only, see
:mod:`multiego.openmm_backend.hbond`) instead of an isotropic Lennard-Jones
attraction.  At generation two things change:

1. **Polar backbone prior.**  The mg prior attracts the polar backbone atoms
   to each other at hydrogen-bond distances (H-O, and N-O at 2.8 A, C-N at
   3.9 A, N-N at 4.5 A, O-O): a mean-field stand-in for the hydrogen bond.
   With an explicit term these attractions would count it twice, so for the
   *backbone* atoms of these pairs the prior becomes purely repulsive (c6 = 0).
   In production, backbone H-O and N-O contacts between atoms at least 3
   residues apart or on different chains are not learned (they describe the
   hydrogen bond itself; ``lj.HBOND_FROZEN_PAIRS``).
   The c12 is the default repulsive c12 of the pair of chemical types used
   everywhere else in multi-eGO (``type_definitions._c12_df``, the ``rep`` of
   a contact).  C-O keeps its prior.  The other learned contacts (C-N, N-N,
   O-O included) are reweighted from this prior with the usual rules.  The change is in
   every output (``ffnonbonded.itp``, ``topol_mego.top`` and
   ``mego_params.h5``), because the GROMACS files of the reference model
   define the prior of the next production step.

2. **Hydrogen-bond settings**, stored in ``mego_params.h5`` and used by the
   OpenMM builder (:class:`HBondSettings`).

Backbone atoms are recognised by name *and* chemical type: N (type N),
H (type H), C (type C), O (type O).  This excludes the charged termini
(NL/OM types, H1-H3, O1/O2) and every side chain.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np

# Backbone roles (index into the 5 x 5 backbone tables); 0 = not a polar backbone atom.
ROLES = ("", "N", "H", "C", "O")
ROLE_INDEX = {name: k for k, name in enumerate(ROLES) if name}
# Pairs of backbone roles whose mg prior becomes repulsive with --hbond.
REPULSIVE_BACKBONE_PAIRS = (("H", "O"), ("N", "O"), ("C", "N"), ("N", "N"), ("O", "O"))


def backbone_roles(atom_names, chem_types):
    """
    Backbone role of each atom: the role index if the atom name and its
    chemical type are both one of N, H, C, O; 0 otherwise.
    """
    names = np.asarray(atom_names).astype(str)
    types = np.asarray(chem_types).astype(str)
    roles = np.zeros(len(names), dtype=int)
    for name, k in ROLE_INDEX.items():
        roles[(names == name) & (types == name)] = k
    return roles


def sbtype_roles(meGO_ensemble):
    """sb_type -> backbone role, for every atom of the system."""
    top = meGO_ensemble.topology_dataframe
    roles = backbone_roles(top["name"], top["type"])
    return dict(zip(top["sb_type"].astype(str), roles.tolist()))


def backbone_repulsive_c12(meGO_ensemble):
    """
    The 5 x 5 table of the repulsive backbone prior: c12 for the role pairs of
    ``REPULSIVE_BACKBONE_PAIRS`` (symmetric), NaN elsewhere (no backbone rule).

    The value is the default repulsive c12 of the two chemical types
    (``type_definitions._c12_df``, as for the ``rep`` of a contact); if the
    pair is not listed there, the geometric mean of the per-type rc_c12.
    """
    from . import lj, type_definitions

    table = lj._nthbond_c12_table(meGO_ensemble)
    code = table["type_to_code"]
    rc_c12 = dict(zip(type_definitions.gromos_atp["name"], type_definitions.gromos_atp["rc_c12"]))
    out = np.full((len(ROLES), len(ROLES)), np.nan)
    for a, b in REPULSIVE_BACKBONE_PAIRS:
        value = table["c12"][code[a], code[b]] if a in code and b in code else 0.0
        if not value > 0.0:
            value = float(np.sqrt(rc_c12[a] * rc_c12[b]))
        out[ROLE_INDEX[a], ROLE_INDEX[b]] = out[ROLE_INDEX[b], ROLE_INDEX[a]] = value
    return out


@dataclass
class HBondSettings:
    """
    Parameters of the geometric hydrogen bond, stored in ``mego_params.h5``.

    E = -eps * f(r) * A_don(theta_NHO) * A_acc(theta_HOC), r = H...O distance;
    f = 1 for r <= r0, exp(-(r - r0)^2 / (2 width^2)) beyond;
    A = exp(-(theta - theta0)^2 / (2 sigma^2)) for theta > 90 deg, 0 otherwise.

    eps is ``epsilon_local`` for donor and acceptor ``min_residue_separation``
    to ``nonlocal_separation - 1`` residues apart, ``epsilon_nonlocal`` beyond
    and between chains.  Pairs closer than ``min_residue_separation`` residues
    keep their Lennard-Jones interaction.
    """

    epsilon_local: float  # kJ/mol
    epsilon_nonlocal: float  # kJ/mol
    r0: float = 0.20  # nm
    width: float = 0.035  # nm
    donor_theta0: float = 180.0  # deg, N-H...O
    donor_width: float = 40.0  # deg
    acceptor_theta0: float = 147.0  # deg, H...O=C
    acceptor_width: float = 25.0  # deg
    cutoff: float = 0.5  # nm, H...O
    min_residue_separation: int = 3
    nonlocal_separation: int = 5

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))

    @classmethod
    def from_args(cls, args):
        """Settings from the command-line arguments; None without --hbond."""
        if not getattr(args, "hbond", False):
            return None
        local = args.hbond_epsilon_local
        nonlocal_ = local if args.hbond_epsilon_nonlocal is None else args.hbond_epsilon_nonlocal
        return cls(
            epsilon_local=float(local),
            epsilon_nonlocal=float(nonlocal_),
            r0=float(args.hbond_r0),
            width=float(args.hbond_width),
        )
