"""
Build an OpenMM ``System`` (and ``Topology``) from a multi-eGO parameter file.

This is the OpenMM counterpart of ``grompp``: it turns the parameters written
by the generation step (:mod:`multiego.omm.paramfile`) into the forces that
are simulated.  The resulting System is equivalent to the GROMACS one built
from ``topol_mego.top`` + ``ffnonbonded.itp`` with ``mdps/ff_aa.mdp``; the
regression test ``tests/test_openmm_parity.py`` checks this term by term.

Forces created (each named after its term, see ``multiego.omm.energy.TERMS``)
----------------------------------------------------------------------------
bonds               HarmonicBondForce        k/2 (r - r0)^2
angles              HarmonicAngleForce       k/2 (theta - theta0)^2
proper_dihedrals    PeriodicTorsionForce     k (1 + cos(n phi - phase))
improper_dihedrals  CustomTorsionForce       k/2 (xi - xi0)^2, periodic difference
pairs               CustomBondForce          c12/r^12 - c6/r^6, no cut-off
nonbonded           CustomNonbondedForce     c12/r^12 - c6/r^6 from type tables,
                                             cut-off + potential shift
hbonds              CustomHbondForce         optional geometric backbone hydrogen
                                             bonds, replacing the H-O LJ (see
                                             multiego.omm.hbond)
plus a CMMotionRemover (GROMACS ``comm_mode = linear``, ``nstcomm = 20``).

There is no electrostatics term: all multi-eGO charges are zero.

Periodic boundary conditions
----------------------------
All bonded forces and pairs use the minimum-image convention
(``setUsesPeriodicBoundaryConditions(True)``), so molecules may cross the box
boundary in the trajectory, as in GROMACS with ``pbc = xyz``.
"""

from collections import defaultdict, deque

import numpy as np
import openmm as mm
import openmm.app as app

from multiego.omm import hbond as hbond_term

# Non-bonded cut-off of the multi-eGO .mdp files (rvdw, nm).
DEFAULT_CUTOFF = 2.5

# Harmonic improper dihedral: the difference xi - xi0 is wrapped into
# [-pi, pi) so the potential is continuous across the +-pi discontinuity of
# the dihedral angle.  Same expression as openmm.app.GromacsTopFile (func 2).
IMPROPER_EXPRESSION = (
    "0.5*k*dtheta^2;"
    "dtheta = theta - theta0 + step(-(theta - theta0 + pi))*2*pi - step(theta - theta0 - pi)*2*pi;"
    f"pi = {np.pi:.15g}"
)


def lj_expression(cutoff, potential_shift):
    """
    Energy expression of the type-table LJ force.

    ``c6tab``/``c12tab`` are the (n_types x n_types) tables; ``type1``/``type2``
    the per-particle type indices.  With ``potential_shift`` the energy at the
    cut-off is subtracted (GROMACS ``vdw-modifier = Potential-shift-Verlet``):
    ``E(r) - E(rc) = c12 (1/r^12 - 1/rc^12) - c6 (1/r^6 - 1/rc^6)``.
    The shift is a constant per type pair: forces are unaffected.
    """
    # float(): a numpy scalar would print as "np.float64(...)" in the expression
    irc6 = 1.0 / float(cutoff) ** 6 if potential_shift else 0.0
    irc12 = irc6**2
    return (
        f"c12*(ir6*ir6 - {irc12!r}) - c6*(ir6 - {irc6!r});"
        "ir6 = 1/r^6;"
        "c6 = c6tab(type1, type2);"
        "c12 = c12tab(type1, type2)"
    )


def bonded_exclusions(n_atoms, bonds_i, bonds_j, nrexcl):
    """
    Atom pairs (i < j) separated by at most ``nrexcl`` bonds.

    This reproduces the GROMACS ``nrexcl`` rule: non-bonded interactions are
    excluded between atoms up to ``nrexcl`` bonds apart (3 in multi-eGO).
    A breadth-first search from every atom is cheap for molecules of a few
    thousand atoms.
    """
    neighbours = defaultdict(list)
    for i, j in zip(bonds_i, bonds_j):
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))

    pairs = set()
    for start in range(n_atoms):
        depth = {start: 0}
        queue = deque([start])
        while queue:
            atom = queue.popleft()
            if depth[atom] == nrexcl:
                continue
            for nxt in neighbours[atom]:
                if nxt not in depth:
                    depth[nxt] = depth[atom] + 1
                    queue.append(nxt)
                    if start < nxt:
                        pairs.add((start, nxt))
    return pairs


def _molecule_list(params, molecules):
    """Default composition: one copy of every molecule type, in file order."""
    if molecules is None:
        return [(name, 1) for name in params.molecules]
    for name, _ in molecules:
        if name not in params.molecules:
            raise ValueError(f"molecule '{name}' is not in the parameter file ({list(params.molecules)})")
    return list(molecules)


def create_topology(params, molecules=None, box_vectors=None):
    """
    OpenMM ``Topology`` of the simulated system (one chain per molecule copy).

    Needed by reporters (XTC/PDB/DCD writers) and for analysis; it is not
    needed to build the System.
    """
    topology = app.Topology()
    for name, count in _molecule_list(params, molecules):
        mol = params.molecules[name]
        for _ in range(count):
            chain = topology.addChain(id=name)
            residues = {}
            atoms = []
            for k in range(mol.n_atoms):
                resnum = int(mol.atoms["resnum"][k])
                if resnum not in residues:
                    residues[resnum] = topology.addResidue(mol.atoms["resname"][k], chain, id=str(resnum))
                element = app.Element.getByAtomicNumber(int(mol.atoms["atomic_number"][k]))
                atoms.append(topology.addAtom(mol.atoms["name"][k], element, residues[resnum]))
            for i, j in zip(mol.bonded["bonds"]["ai"], mol.bonded["bonds"]["aj"]):
                topology.addBond(atoms[int(i)], atoms[int(j)])
    if box_vectors is not None:
        topology.setPeriodicBoxVectors(box_vectors)
    return topology


def create_system(
    params,
    molecules=None,
    box_vectors=None,
    cutoff=DEFAULT_CUTOFF,
    potential_shift=True,
    com_removal_interval=20,
    hbond=None,
):
    """
    Build the OpenMM System of a multi-eGO model.

    Parameters
    ----------
    params : multiego.omm.paramfile.MegoParameters
        Parameters read with ``paramfile.read``.
    molecules : list of (str, int), optional
        System composition as (molecule type, number of copies), in the order
        of the coordinates; the equivalent of the GROMACS ``[ molecules ]``
        section.  Default: one copy of each molecule type.
    box_vectors : tuple of 3 openmm.Vec3 (nm), optional
        Periodic box.  Required: multi-eGO always uses periodic boundaries.
    cutoff : float
        Non-bonded cut-off (nm).
    potential_shift : bool
        Shift the LJ potential to zero at the cut-off.
    com_removal_interval : int
        Steps between centre-of-mass motion removal (0 disables it).
    hbond : None, False or multiego.omm.hbond.HBondSettings
        Geometric hydrogen-bond term.  ``None`` (default): as stored in the
        parameter file (``--hbond`` at generation); ``False``: never;
        ``HBondSettings``: always, with these settings.

    Returns
    -------
    openmm.System
    """
    if box_vectors is None:
        raise ValueError("multi-eGO systems are periodic: box_vectors are required")
    composition = _molecule_list(params, molecules)

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # --- create one force per term; names identify the terms (energy.TERMS) ---
    bonds = mm.HarmonicBondForce()
    angles = mm.HarmonicAngleForce()
    propers = mm.PeriodicTorsionForce()
    impropers = mm.CustomTorsionForce(IMPROPER_EXPRESSION)
    impropers.addPerTorsionParameter("theta0")
    impropers.addPerTorsionParameter("k")
    pairs = mm.CustomBondForce("c12/r^12 - c6/r^6")
    pairs.addPerBondParameter("c6")
    pairs.addPerBondParameter("c12")

    nonbonded = mm.CustomNonbondedForce(lj_expression(cutoff, potential_shift))
    nonbonded.addPerParticleParameter("type")
    nonbonded.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(cutoff)
    # DispCorr = No.  Also essential for speed: with one type per atom the
    # correction would be integrated for every pair of types.
    nonbonded.setUseLongRangeCorrection(False)

    for force, name in (
        (bonds, "bonds"),
        (angles, "angles"),
        (propers, "proper_dihedrals"),
        (impropers, "improper_dihedrals"),
        (pairs, "pairs"),
        (nonbonded, "nonbonded"),
    ):
        force.setName(name)
        if force is not nonbonded:
            force.setUsesPeriodicBoundaryConditions(True)

    hbond_settings = params.hbond if hbond is None else (hbond or None)
    donors, acceptors, all_exclusions = [], [], set()

    # --- add the atoms and interactions of every molecule copy ---
    offset = 0
    for name, count in composition:
        mol = params.molecules[name]
        b = mol.bonded
        # Exclusions are the same for every copy: compute them once.
        exclusions = bonded_exclusions(mol.n_atoms, b["bonds"]["ai"], b["bonds"]["aj"], params.nrexcl)
        for key in ("exclusions", "pairs"):
            exclusions.update((min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(b[key]["ai"], b[key]["aj"]))
        exclusions = sorted(exclusions)
        if hbond_settings is not None:
            mol_donors, mol_acceptors = hbond_term.find_donors_acceptors(mol.atoms)

        for _ in range(count):
            for k in range(mol.n_atoms):
                system.addParticle(float(mol.atoms["mass"][k]))
                nonbonded.addParticle([float(mol.atoms["nb_type"][k])])

            t = b["bonds"]
            for i, j, r0, k in zip(t["ai"], t["aj"], t["r0"], t["k"]):
                bonds.addBond(offset + int(i), offset + int(j), float(r0), float(k))
            t = b["angles"]
            for i, j, kk, theta0, k in zip(t["ai"], t["aj"], t["ak"], t["theta0"], t["k"]):
                angles.addAngle(offset + int(i), offset + int(j), offset + int(kk), float(theta0), float(k))
            t = b["proper_dihedrals"]
            for i, j, kk, ll, n, phase, k in zip(
                t["ai"], t["aj"], t["ak"], t["al"], t["periodicity"], t["phase"], t["k"]
            ):
                propers.addTorsion(
                    offset + int(i), offset + int(j), offset + int(kk), offset + int(ll), int(n), float(phase), float(k)
                )
            t = b["improper_dihedrals"]
            for i, j, kk, ll, theta0, k in zip(t["ai"], t["aj"], t["ak"], t["al"], t["theta0"], t["k"]):
                impropers.addTorsion(
                    offset + int(i), offset + int(j), offset + int(kk), offset + int(ll), [float(theta0), float(k)]
                )
            t = b["pairs"]
            for i, j, c6, c12 in zip(t["ai"], t["aj"], t["c6"], t["c12"]):
                pairs.addBond(offset + int(i), offset + int(j), [float(c6), float(c12)])
            for i, j in exclusions:
                nonbonded.addExclusion(offset + i, offset + j)
            if hbond_settings is not None:
                types = mol.atoms["nb_type"]
                donors += [(offset + h, offset + n, int(types[h])) for h, n in zip(mol_donors["h"], mol_donors["n"])]
                acceptors += [
                    (offset + o, offset + c, int(types[o])) for o, c in zip(mol_acceptors["o"], mol_acceptors["c"])
                ]
                all_exclusions.update((offset + i, offset + j) for i, j in exclusions)
            offset += mol.n_atoms

    # --- LJ type tables (and the optional hydrogen-bond term, which takes
    #     over the donor-acceptor entries of the tables) ---
    c6_table, c12_table = params.nonbonded.dense_tables()
    if hbond_settings is not None and donors and acceptors:
        hbond_term.add_hbond_force(system, hbond_settings, c6_table, c12_table, donors, acceptors, all_exclusions)
    n_types = params.nonbonded.n_types
    # Discrete2DFunction stores values with the first index varying fastest;
    # the tables are symmetric, so the flattening order is irrelevant anyway.
    nonbonded.addTabulatedFunction("c6tab", mm.Discrete2DFunction(n_types, n_types, c6_table.ravel(order="F")))
    nonbonded.addTabulatedFunction("c12tab", mm.Discrete2DFunction(n_types, n_types, c12_table.ravel(order="F")))
    del c6_table, c12_table

    for force in (bonds, angles, propers, impropers, pairs, nonbonded):
        system.addForce(force)
    if com_removal_interval:
        system.addForce(mm.CMMotionRemover(com_removal_interval))
    return system
