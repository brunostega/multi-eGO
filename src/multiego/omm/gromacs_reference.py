"""
Load a GROMACS multi-eGO model into OpenMM, reproducing GROMACS settings.

This is the *ground truth* used to validate the native OpenMM builder: OpenMM's
own ``GromacsTopFile`` reader is an independent implementation of the GROMACS
topology format, so agreement between the two Systems means that the native
builder reproduces exactly what GROMACS simulates.

Settings reproduced from ``mdps/ff_aa.mdp``
-------------------------------------------
* ``cutoff-scheme = verlet``, ``vdw-type = Cut-off``, ``rvdw = 2.5`` nm
    -> ``CutoffPeriodic`` with a 2.5 nm cut-off.
* ``vdw-modifier = Potential-shift-Verlet``
    -> the LJ energy is shifted so that it is zero at the cut-off.  OpenMM has
    no built-in equivalent for custom forces, so the shift is added to the
    energy expression (see :func:`add_potential_shift`).  The shift is a
    constant per type pair: it changes energies but not forces.
* ``DispCorr = No``
    -> long-range (dispersion) correction disabled.  This is also essential
    for speed: with one atom type per atom, OpenMM would otherwise integrate
    the correction numerically for every pair of types (~3.5 minutes for a
    1,448-atom protein on one CPU core).
* All charges are zero, so the ``NonbondedForce`` created by the reader
  contributes nothing (it is kept only because the reader always creates it).
"""

import os

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# Van der Waals cut-off of the multi-eGO GROMACS .mdp files (rvdw, nm).
DEFAULT_CUTOFF = 2.5


def add_potential_shift(force, cutoff):
    """
    Shift a type-table LJ ``CustomNonbondedForce`` to zero at the cut-off.

    ``GromacsTopFile`` writes the LJ energy as
    ``(a/r6)^2-b/r6; r6=r^6; a=acoef(type1, type2); b=bcoef(type1, type2)``
    with ``a = sqrt(c12)`` and ``b = c6``.  GROMACS' Potential-shift-Verlet
    subtracts the same function evaluated at the cut-off ``rc``, giving
    ``E(r) - E(rc)``.  The shifted expression is built by appending
    ``- ((a/rc6)^2-b/rc6)`` to the first (energy) line of the expression.

    Parameters
    ----------
    force : openmm.CustomNonbondedForce
        Modified in place.
    cutoff : float
        Cut-off distance in nm.
    """
    energy, *definitions = force.getEnergyFunction().split(";")
    shifted = f"{energy} - ((a/rc6)^2-b/rc6); rc6={cutoff}^6"
    force.setEnergyFunction(";".join([shifted] + definitions))


def load_gromacs_system(top_file, gro_file, cutoff=DEFAULT_CUTOFF, potential_shift=True):
    """
    Build an OpenMM System from GROMACS multi-eGO output files.

    Parameters
    ----------
    top_file : str
        Path to ``topol_mego.top``.  ``#include`` files (``ffnonbonded.itp``)
        are searched in the same directory.
    gro_file : str
        Coordinates (and periodic box) in GROMACS ``.gro`` format.
    cutoff : float
        Non-bonded cut-off in nm (``rvdw`` of the .mdp file).
    potential_shift : bool
        Reproduce GROMACS ``vdw-modifier = Potential-shift-Verlet``.

    Returns
    -------
    system : openmm.System
    positions : openmm.unit.Quantity
    """
    gro = app.GromacsGroFile(gro_file)
    top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=os.path.dirname(os.path.abspath(top_file)),
    )
    system = top.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff * unit.nanometer,
        useDispersionCorrection=False,  # DispCorr = No
    )

    for force in system.getForces():
        if isinstance(force, mm.CustomNonbondedForce):
            # The reader creates the LJ table force with the dispersion
            # correction flag taken from useDispersionCorrection; make sure.
            force.setUseLongRangeCorrection(False)
            if potential_shift:
                add_potential_shift(force, cutoff)
    return system, gro.getPositions()
