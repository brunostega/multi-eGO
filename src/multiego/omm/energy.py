"""
Term-by-term energy and force decomposition of an OpenMM ``System``.

Why this module exists
----------------------
Two OpenMM Systems that describe the same multi-eGO model can be built in
different ways (e.g. by OpenMM's own ``GromacsTopFile`` reader from the GROMACS
output files, or by the native multi-eGO builder from the parameter file).
They may use different Force classes, a different number of forces, or a
different force order.  To compare them we therefore do not compare forces
one by one; instead every Force is assigned to a physical *term* (bonds,
angles, ..., nonbonded) and energies and per-atom forces are compared term by
term.

Term names
----------
The canonical term names are listed in ``TERMS``.  ``FORCE_NAME_TO_TERM`` maps
the names OpenMM gives to forces (``Force.getName()``) onto these terms.  It
contains both the names used by ``openmm.app.GromacsTopFile`` and the names
used by the native multi-eGO builder (which simply names each force after its
term).  Adding a new kind of force only requires adding its name here.
"""

import numpy as np
import openmm as mm
import openmm.unit as unit

# Canonical multi-eGO energy terms, in the order they are reported.
TERMS = (
    "bonds",
    "angles",
    "proper_dihedrals",
    "improper_dihedrals",
    "pairs",  # intramolecular [ pairs ]: 1-4 and learned short-range contacts
    "nonbonded",  # type-table LJ: molten-globule prior + learned contacts
    "hbonds",  # optional geometric hydrogen-bond term (OpenMM only)
    "electrostatics",  # always zero in multi-eGO (all charges are 0)
)

# Force.getName() -> term.  Names on the left of each block are those produced
# by openmm.app.GromacsTopFile; the term names themselves are used by the
# native multi-eGO builder.
FORCE_NAME_TO_TERM = {
    # GromacsTopFile names
    "HarmonicBondForce": "bonds",
    "GROMOSBondForce": "bonds",
    "HarmonicAngleForce": "angles",
    "GROMOSAngleForce": "angles",
    "PeriodicTorsionForce": "proper_dihedrals",
    "HarmonicTorsionForce": "improper_dihedrals",
    "LennardJonesExceptions": "pairs",
    "LennardJonesForce": "nonbonded",
    "NonbondedForce": "electrostatics",
    # Native builder names (identity mapping)
    **{term: term for term in TERMS},
}

# Forces that do not contribute to the potential energy and are ignored.
IGNORED_FORCE_NAMES = {"CMMotionRemover"}


def classify_force(force):
    """
    Return the multi-eGO term a Force belongs to.

    Parameters
    ----------
    force : openmm.Force

    Returns
    -------
    str or None
        The term name, or ``None`` for forces that carry no energy
        (e.g. ``CMMotionRemover``).

    Raises
    ------
    ValueError
        If the force name is unknown: silently ignoring a force would make a
        comparison pass while a whole energy term is missing.
    """
    name = force.getName()
    if name in IGNORED_FORCE_NAMES:
        return None
    if name not in FORCE_NAME_TO_TERM:
        raise ValueError(f"Unknown force '{name}' ({type(force).__name__}): add it to FORCE_NAME_TO_TERM")
    return FORCE_NAME_TO_TERM[name]


def decompose(system, positions, platform_name="Reference"):
    """
    Compute the potential energy and the per-atom forces of every term.

    The System is cloned, each force is put in the force group of its term,
    and one ``getState`` call per group gives that term's energy and forces.
    The input System is not modified.

    Parameters
    ----------
    system : openmm.System
    positions : array-like or openmm.unit.Quantity
        Particle positions (nm if no unit is attached).
    platform_name : str
        OpenMM platform.  ``Reference`` (double precision, deterministic) is
        the default because these numbers are used for regression tests.

    Returns
    -------
    dict
        ``{term: {"energy": float [kJ/mol], "forces": ndarray (N, 3) [kJ/mol/nm]}}``
        for every term present in the System.
    """
    system = mm.XmlSerializer.clone(system)
    term_of_group = {}
    for force in system.getForces():
        term = classify_force(force)
        if term is None:
            # Group 31 is never queried: the force is effectively ignored.
            force.setForceGroup(31)
            continue
        group = TERMS.index(term)
        force.setForceGroup(group)
        term_of_group[group] = term

    # The integrator is irrelevant: no dynamics is run.
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform_name))
    context.setPositions(positions)

    result = {}
    for group, term in sorted(term_of_group.items()):
        state = context.getState(getEnergy=True, getForces=True, groups={group})
        result[term] = {
            "energy": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            "forces": state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        }
    return result


def compare(reference, candidate, energy_rtol=1e-5, energy_atol=1e-6, force_rtol=1e-4):
    """
    Compare two decompositions produced by :func:`decompose`.

    A term passes when
      * it is present in both decompositions (a term with zero energy and zero
        forces may be missing on one side: e.g. a zero-charge NonbondedForce);
      * ``|E_ref - E_cand| <= energy_atol + energy_rtol * |E_ref|``;
      * the largest per-atom force deviation is at most ``force_rtol`` times
        the largest reference force component of that term (plus
        ``energy_atol`` to allow for exactly-zero terms).

    The tolerances absorb the 6-significant-digit rounding of the GROMACS
    text files (``%.6e``), which is the only expected source of difference.

    Returns
    -------
    list of str
        Human-readable descriptions of the failures (empty if all terms pass).
    """
    failures = []
    for term in TERMS:
        ref, cand = reference.get(term), candidate.get(term)
        if ref is None and cand is None:
            continue
        # A term present on one side only must be identically zero.
        if ref is None or cand is None:
            present = ref if ref is not None else cand
            if abs(present["energy"]) > energy_atol or np.abs(present["forces"]).max() > energy_atol:
                side = "reference" if ref is not None else "candidate"
                failures.append(f"{term}: only present in {side} (E = {present['energy']:.6f} kJ/mol)")
            continue

        de = abs(ref["energy"] - cand["energy"])
        if de > energy_atol + energy_rtol * abs(ref["energy"]):
            failures.append(
                f"{term}: energy {cand['energy']:.6f} != reference {ref['energy']:.6f} kJ/mol (diff {de:.3e})"
            )

        df = np.abs(ref["forces"] - cand["forces"]).max()
        fscale = np.abs(ref["forces"]).max()
        if df > energy_atol + force_rtol * fscale:
            failures.append(f"{term}: max force deviation {df:.3e} kJ/mol/nm (reference max force {fscale:.3e})")
    return failures


def format_table(reference, candidate=None):
    """
    Return a printable table of per-term energies (and differences, when a
    candidate decomposition is given).  Useful for reports and debugging.
    """
    lines = [f"{'term':<20}{'reference':>18}" + (f"{'candidate':>18}{'diff':>14}" if candidate else "")]
    for term in TERMS:
        if term not in reference and (candidate is None or term not in candidate):
            continue
        e_ref = reference.get(term, {}).get("energy", 0.0)
        line = f"{term:<20}{e_ref:>18.6f}"
        if candidate is not None:
            e_cand = candidate.get(term, {}).get("energy", 0.0)
            line += f"{e_cand:>18.6f}{e_cand - e_ref:>14.3e}"
        lines.append(line)
    return "\n".join(lines)
