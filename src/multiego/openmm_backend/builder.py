"""
Build an OpenMM System from a multi-eGO GROMACS topology.

The translation reproduces what GROMACS computes with ``mdps/ff_aa.mdp``:

=================================  ==============================================
GROMACS                            OpenMM
=================================  ==============================================
bonds funct 1                      HarmonicBondForce
angles funct 1                     HarmonicAngleForce
dihedrals funct 1/4/9              PeriodicTorsionForce (one term per line)
dihedrals funct 2 (impropers)      CustomTorsionForce, 0.5 k dxi^2, dxi wrapped
                                   to [-pi, pi)
pairs funct 1                      CustomBondForce c12/r^12 - c6/r^6, no cutoff,
                                   no shift
atomtypes + nonbond_params         CustomNonbondedForce, per-atom type index,
                                   tabulated c6/c12 (nonbond_params, otherwise
                                   geometric combination), periodic cutoff,
                                   potential shift, no switch, no dispersion
                                   correction
nrexcl + [ exclusions ]            exclusions on the CustomNonbondedForce
comm every 20 steps                CMMotionRemover(20)
backbone CMAP (optional)           CMAPTorsionForce on (C-1, N, CA, C, N+1), maps
                                   from ``cmap_dir`` (see ``cmap.py``)
=================================  ==============================================

Each term has its own force group (see ``FORCE_GROUPS``), so energies can be
compared term by term with ``gmx energy``.

Only atom types that occur in the system enter the c6/c12 tables. Copies of a
molecule share their types, so the table size depends on the molecule types,
not on the number of copies.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

import openmm
from openmm import app, unit

from . import cmap as _cmap
from .gromacs_reader import GromacsTopology, read_topology

# Force group of each term. Names match the GROMACS energy terms.
FORCE_GROUPS = {
    "Bond": 0,
    "Angle": 1,
    "Proper Dih.": 2,
    "Improper Dih.": 3,
    "LJ-14": 4,
    "LJ (SR)": 5,
    "CMAP": 6,
}

DEFAULT_CUTOFF = 2.5  # nm, rvdw in mdps/ff_aa.mdp
DEFAULT_TIMESTEP = 0.005  # ps
DEFAULT_FRICTION = 0.04  # 1/ps, 1/tau_t with tau_t = 25 ps
DEFAULT_COM_INTERVAL = 20  # steps, nstcomm


def _f(x):
    """Literal for an OpenMM expression (never ``np.float64(...)``)."""
    return repr(float(x))


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
def molecule_exclusions(moltype):
    """
    Excluded pairs of one molecule type, as GROMACS builds them.

    All pairs within ``nrexcl`` bonds (through ``[ bonds ]``) plus the explicit
    ``[ exclusions ]``. ``[ pairs ]`` are *not* excluded automatically.

    Returns
    -------
    set of (i, j) with i < j, 0-based indices within the molecule
    """
    n = len(moltype.atoms)
    neighbours = [[] for _ in range(n)]
    for i, j, *_ in moltype.bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)

    excluded = set()
    for start in range(n):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            if dist[a] == moltype.nrexcl:
                continue
            for b in neighbours[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    queue.append(b)
        for b in dist:
            if b > start:
                excluded.add((start, b))
    excluded.update(moltype.exclusions)
    return excluded


# ---------------------------------------------------------------------------
# Nonbonded tables
# ---------------------------------------------------------------------------
def lj_tables(top: GromacsTopology):
    """
    c6 and c12 tables over the atom types used in the system.

    Returns
    -------
    type_index : dict  type name -> row/column in the tables
    c6, c12 : (n, n) ndarrays, symmetric
    """
    used = []
    seen = set()
    for name, count in top.molecules:
        if count == 0:
            continue
        for atom in top.moleculetypes[name].atoms:
            if atom.type not in seen:
                seen.add(atom.type)
                used.append(atom.type)
    type_index = {t: k for k, t in enumerate(used)}

    c6_i = np.array([top.atomtypes[t].c6 for t in used])
    c12_i = np.array([top.atomtypes[t].c12 for t in used])
    # combination rule 1: geometric mean of c6 and of c12
    c6 = np.sqrt(np.outer(c6_i, c6_i))
    c12 = np.sqrt(np.outer(c12_i, c12_i))
    for (a, b), (p6, p12) in top.nonbond_params.items():
        ia, ib = type_index.get(a), type_index.get(b)
        if ia is None or ib is None:
            continue
        c6[ia, ib] = c6[ib, ia] = p6
        c12[ia, ib] = c12[ib, ia] = p12
    return type_index, c6, c12


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
def create_system(
    top: GromacsTopology,
    box_vectors,
    cutoff=DEFAULT_CUTOFF,
    com_interval=DEFAULT_COM_INTERVAL,
    cmap_dir=None,
):
    """
    Build the OpenMM System.

    Parameters
    ----------
    top : GromacsTopology
        From :func:`read_topology`.
    box_vectors : sequence of 3 Vec3 or (3, 3) array, in nm
        Periodic box (e.g. from ``app.GromacsGroFile.getPeriodicBoxVectors()``).
    cutoff : float
        Nonbonded cutoff in nm.
    com_interval : int or None
        Steps between centre-of-mass motion removal; ``None`` or 0 adds none.
    cmap_dir : str or None
        Directory with ``<RES>.cmap`` maps. Every residue with both
        neighbours (see ``cmap.backbone_quintuplets``) gets the map of its
        residue type (D-residues: mirrored L map). Raises if a residue has no
        map, or if it still has a backbone phi/psi proper dihedral (generate
        the model with ``--backbone cmap``). ``None``: no CMAP.

    Returns
    -------
    openmm.System
    """
    charged = sorted(
        {a.type for name, count in top.molecules if count for a in top.moleculetypes[name].atoms if a.charge != 0.0}
    )
    if charged:
        raise ValueError(f"non-zero charges are not supported (types {charged[:5]}...)")

    system = openmm.System()
    box = [openmm.Vec3(*[float(x) for x in _strip(v)]) for v in box_vectors]
    system.setDefaultPeriodicBoxVectors(*box)
    half_min_box = 0.5 * min(box[0][0], box[1][1], box[2][2])
    if cutoff > half_min_box:
        raise ValueError(f"cutoff {cutoff} nm is longer than half the box ({half_min_box:.3f} nm)")

    bond_f = openmm.HarmonicBondForce()
    angle_f = openmm.HarmonicAngleForce()
    proper_f = openmm.PeriodicTorsionForce()
    improper_f = openmm.CustomTorsionForce(
        "0.5*k*d^2; d = dt - 2*pi*floor((dt + pi)/(2*pi)); dt = theta - theta0; pi = 3.141592653589793"
    )
    improper_f.addPerTorsionParameter("theta0")
    improper_f.addPerTorsionParameter("k")
    pair_f = openmm.CustomBondForce("c12/r^12 - c6/r^6")
    pair_f.addPerBondParameter("c6")
    pair_f.addPerBondParameter("c12")

    type_index, c6, c12 = lj_tables(top)
    n_types = len(type_index)
    rc6 = cutoff**-6
    rc12 = cutoff**-12
    nb_f = openmm.CustomNonbondedForce(
        f"c12tab(t1, t2)*(ir6*ir6 - {_f(rc12)}) - c6tab(t1, t2)*(ir6 - {_f(rc6)}); ir6 = r^-6"
    )
    nb_f.addPerParticleParameter("t")
    # Discrete2DFunction value (i, j) is at i + n*j; the tables are symmetric.
    nb_f.addTabulatedFunction("c6tab", openmm.Discrete2DFunction(n_types, n_types, c6.ravel(order="F").tolist()))
    nb_f.addTabulatedFunction("c12tab", openmm.Discrete2DFunction(n_types, n_types, c12.ravel(order="F").tolist()))
    nb_f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    nb_f.setCutoffDistance(cutoff)
    nb_f.setUseSwitchingFunction(False)
    nb_f.setUseLongRangeCorrection(False)

    deg = math.pi / 180.0
    offset = 0
    for name, count in top.molecules:
        mt = top.moleculetypes[name]
        excl = sorted(molecule_exclusions(mt))
        for _ in range(count):
            for atom in mt.atoms:
                system.addParticle(atom.mass)
                nb_f.addParticle([float(type_index[atom.type])])
            o = offset
            for i, j, b0, kb in mt.bonds:
                bond_f.addBond(o + i, o + j, b0, kb)
            for i, j, k, th0, kth in mt.angles:
                angle_f.addAngle(o + i, o + j, o + k, th0 * deg, kth)
            for i, j, k, l, phi0, kphi, mult in mt.propers:
                proper_f.addTorsion(o + i, o + j, o + k, o + l, mult, phi0 * deg, kphi)
            for i, j, k, l, xi0, kxi in mt.impropers:
                improper_f.addTorsion(o + i, o + j, o + k, o + l, [xi0 * deg, kxi])
            for i, j, p6, p12 in mt.pairs:
                pair_f.addBond(o + i, o + j, [p6, p12])
            for i, j in excl:
                nb_f.addExclusion(o + i, o + j)
            offset += len(mt.atoms)

    for label, force in (
        ("Bond", bond_f),
        ("Angle", angle_f),
        ("Proper Dih.", proper_f),
        ("Improper Dih.", improper_f),
        ("LJ-14", pair_f),
    ):
        force.setUsesPeriodicBoundaryConditions(True)
        force.setForceGroup(FORCE_GROUPS[label])
        force.setName(label)
        system.addForce(force)
    nb_f.setForceGroup(FORCE_GROUPS["LJ (SR)"])
    nb_f.setName("LJ (SR)")
    system.addForce(nb_f)

    if cmap_dir is not None:
        cmap_f = _cmap_force(top, cmap_dir)
        cmap_f.setUsesPeriodicBoundaryConditions(True)
        cmap_f.setForceGroup(FORCE_GROUPS["CMAP"])
        cmap_f.setName("CMAP")
        system.addForce(cmap_f)

    if com_interval:
        system.addForce(openmm.CMMotionRemover(int(com_interval)))
    return system


def _cmap_force(top, cmap_dir):
    """CMAPTorsionForce with only the maps the system uses."""
    per_moltype = {}
    needed = set()
    problems = []
    for name, count in top.molecules:
        if count == 0 or name in per_moltype:
            continue
        mt = top.moleculetypes[name]
        quints = _cmap.backbone_quintuplets(mt)
        per_moltype[name] = quints
        needed.update(_cmap.map_residue(resname)[0] for resname, _, _ in quints)
        # a CMAP on top of a phi/psi proper would count the backbone potential twice
        torsions = {tuple(p[:4]) for p in mt.propers}
        for resname, resnum, (cp, n, ca, c, nn) in quints:
            for quad, label in (((cp, n, ca, c), "phi"), ((n, ca, c, nn), "psi")):
                if quad in torsions or quad[::-1] in torsions:
                    problems.append(f"{name} {resname}{resnum} still has a {label} proper dihedral")
    if problems:
        raise ValueError(
            "CMAP requested but backbone phi/psi dihedrals are present (generate with --backbone cmap): "
            + "; ".join(problems[:5])
            + (" ..." if len(problems) > 5 else "")
        )
    maps = _cmap.load_maps(cmap_dir, needed)
    missing = sorted(needed - set(maps))
    if missing:
        raise ValueError(f"no CMAP in {cmap_dir} for residue type(s) {missing}")

    force = openmm.CMAPTorsionForce()
    map_index = {}
    offset = 0
    for name, count in top.molecules:
        mt = top.moleculetypes[name]
        quints = per_moltype.get(name, [])
        for _ in range(count):
            for resname, _resnum, atoms in quints:
                key = _cmap.map_residue(resname)
                if key not in map_index:
                    grid = maps[key[0]]
                    if key[1]:
                        grid = _cmap.mirror(grid)
                    map_index[key] = force.addMap(grid.shape[0], _cmap.to_openmm_map(grid))
                cp, n, ca, c, nn = (offset + a for a in atoms)
                force.addTorsion(map_index[key], cp, n, ca, c, n, ca, c, nn)
            offset += len(mt.atoms)
    return force


def create_topology(top: GromacsTopology, box_vectors=None):
    """
    OpenMM ``app.Topology`` with one chain per molecule copy, for writing
    trajectories and selecting atoms.
    """
    omm_top = app.Topology()
    for name, count in top.molecules:
        mt = top.moleculetypes[name]
        for _ in range(count):
            chain = omm_top.addChain(id=name)
            atoms = []
            residue, last = None, None
            for atom in mt.atoms:
                key = (atom.resnum, atom.resname)
                if key != last:
                    residue = omm_top.addResidue(atom.resname, chain, id=str(atom.resnum))
                    last = key
                z = top.atomtypes[atom.type].atomic_number
                element = app.Element.getByAtomicNumber(z) if z > 0 else None
                atoms.append(omm_top.addAtom(atom.name, element, residue))
            for i, j, *_ in mt.bonds:
                omm_top.addBond(atoms[i], atoms[j])
    if box_vectors is not None:
        omm_top.setPeriodicBoxVectors(
            [openmm.Vec3(*[float(x) for x in _strip(v)]) for v in box_vectors] * unit.nanometer
        )
    return omm_top


def create_integrator(temperature, timestep=DEFAULT_TIMESTEP, friction=DEFAULT_FRICTION, seed=None):
    """
    Langevin integrator matching ``mdps/ff_aa.mdp`` (sd, tau_t 25 ps, dt 5 fs).

    It samples the same ensemble as GROMACS ``sd``; the discretisation differs,
    so trajectories are not identical.

    Parameters
    ----------
    temperature : float
        Kelvin.
    timestep : float
        ps.
    friction : float
        1/ps.
    """
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    return integrator


def load(top_path, gro_path, cutoff=DEFAULT_CUTOFF, com_interval=DEFAULT_COM_INTERVAL, cmap_dir=None):
    """
    Read a multi-eGO model and its coordinates.

    Parameters
    ----------
    top_path : str
        ``topol_mego.top``; ``[ molecules ]`` counts are used as written.
    gro_path : str
        Coordinates and box, atoms in topology order.
    cmap_dir : str or None
        Backbone CMAP maps, see :func:`create_system`.

    Returns
    -------
    system : openmm.System
    topology : openmm.app.Topology
    positions : Quantity, (n_atoms, 3) nm
    """
    top = read_topology(top_path)
    gro = app.GromacsGroFile(gro_path)
    positions = gro.getPositions()
    if len(positions) != top.n_atoms:
        raise ValueError(f"{gro_path} has {len(positions)} atoms, the topology {top.n_atoms}")
    box = gro.getPeriodicBoxVectors()
    if box is None:
        raise ValueError(f"{gro_path} has no box")
    box_nm = [v.value_in_unit(unit.nanometer) for v in box]
    system = create_system(top, box_nm, cutoff=cutoff, com_interval=com_interval, cmap_dir=cmap_dir)
    topology = create_topology(top, box_nm)
    return system, topology, positions


def _strip(v):
    """Vec3 / Quantity / sequence -> plain floats in nm."""
    if unit.is_quantity(v):
        v = v.value_in_unit(unit.nanometer)
    return [float(x) for x in v]
