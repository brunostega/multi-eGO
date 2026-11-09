"""
OpenMM support for multi-eGO.

This sub-package contains everything that is specific to the OpenMM engine.
It is deliberately kept separate from the force-field *generation* code in
``multiego`` (contacts, LJ learning, molten-globule prior, ...), which stays
engine-independent: generation produces parameters, and the modules here turn
those parameters into an OpenMM ``System``.

Modules
-------
energy
    Engine-neutral energy/force decomposition of an OpenMM ``System`` into
    multi-eGO terms (bonds, angles, dihedrals, pairs, nonbonded, ...).  Used by
    the regression tests to compare two Systems term by term.
gromacs_reference
    Loads a GROMACS multi-eGO topology (``topol_mego.top`` +
    ``ffnonbonded.itp``) into OpenMM with the same settings used by the
    multi-eGO GROMACS ``.mdp`` files (plain cut-off, potential shift,
    no dispersion correction).  This is the ground truth the native OpenMM
    builder is validated against.

The package is named ``omm`` rather than ``openmm`` so that it can never be
confused with (or shadow) the ``openmm`` library itself.
"""
