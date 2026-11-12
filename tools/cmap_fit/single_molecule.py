"""
A faster, exactly equivalent System for *one isolated molecule* in a large box.

The multi-eGO non-bonded term is a ``CustomNonbondedForce`` with a type table
and a neighbour list.  For a single small molecule in a large box the
neighbour-list machinery costs more than the interactions themselves.  Since
every atom pair is within the cut-off anyway, the same energy can be written
as an explicit list of pairs in a ``CustomBondForce``:

    E = step(rc - r) * [ c12 (r^-12 - rc^-12) - c6 (r^-6 - rc^-6) ]

which reproduces the cut-off and the potential shift exactly.

Validity
--------
Exact as long as periodic images never come within the cut-off, i.e. the box
must be larger than the molecule plus the cut-off.  This is checked by the
caller (for the blocked amino acids used here, a 5 nm box with a 2.5 nm
cut-off and a molecule under 1 nm across).  It is **not** valid for several
molecules, nor for anything that can diffuse apart: use
``multiego.omm.builder.create_system`` there.

Measured speed-up on a blocked amino acid: about 2x (7800 vs 4100 ns/day on
one CPU core).  Equivalence with the standard System is verified in
tests/test_cmap.py.
"""

import numpy as np
import openmm as mm

from multiego.omm import builder


def create_single_molecule_system(params, box_vectors, cutoff=builder.DEFAULT_CUTOFF, **kwargs):
    """
    Build the System with ``builder.create_system`` and replace its type-table
    non-bonded force by the equivalent explicit pair list.

    Extra keyword arguments are passed to ``create_system`` (``hbond``, ...).
    """
    system = builder.create_system(params, box_vectors=box_vectors, cutoff=cutoff, **kwargs)
    index = next(i for i, force in enumerate(system.getForces()) if force.getName() == "nonbonded")
    nonbonded = system.getForces()[index]

    # Recover the effective (c6, c12) of every pair from the force's own
    # tables, so that this helper cannot drift from the builder.
    tables = {
        nonbonded.getTabulatedFunctionName(i): nonbonded.getTabulatedFunction(i).getFunctionParameters()
        for i in range(nonbonded.getNumTabulatedFunctions())
    }
    size, _, values6 = tables["c6tab"]
    _, _, values12 = tables["c12tab"]
    c6_table = np.array(values6).reshape(size, size).T
    c12_table = np.array(values12).reshape(size, size).T
    types = [int(nonbonded.getParticleParameters(i)[0]) for i in range(nonbonded.getNumParticles())]
    excluded = {tuple(sorted(nonbonded.getExclusionParticles(e))) for e in range(nonbonded.getNumExclusions())}

    inverse_rc6 = 1.0 / float(cutoff) ** 6
    pairs = mm.CustomBondForce(
        f"step({float(cutoff)!r} - r)*(c12*(1/r^12 - {inverse_rc6 ** 2!r}) - c6*(1/r^6 - {inverse_rc6!r}))"
    )
    pairs.addPerBondParameter("c6")
    pairs.addPerBondParameter("c12")
    pairs.setName("nonbonded")  # same term name, so energy decompositions match
    pairs.setUsesPeriodicBoundaryConditions(False)
    n = system.getNumParticles()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            c6, c12 = c6_table[types[i], types[j]], c12_table[types[i], types[j]]
            if c6 != 0.0 or c12 != 0.0:
                pairs.addBond(i, j, [float(c6), float(c12)])

    system.removeForce(index)
    system.addForce(pairs)
    return system
