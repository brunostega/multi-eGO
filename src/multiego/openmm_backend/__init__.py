"""
OpenMM backend for multi-eGO models.

Builds an OpenMM System from the GROMACS files that multi-eGO writes
(``topol_mego.top`` and ``ffnonbonded.itp``). Requires OpenMM, installed with
``pip install "mego[openmm]"`` (add ``openmm[cuda12]`` or similar for GPUs).

Typical use::

    from multiego.openmm_backend import load, create_integrator
    system, topology, positions = load("topol_mego.top", "conf.gro")
    integrator = create_integrator(300.0)
"""

from .gromacs_reader import TopologyError, read_topology
from .builder import (
    FORCE_GROUPS,
    create_integrator,
    create_system,
    create_topology,
    lj_tables,
    load,
    molecule_exclusions,
)

__all__ = [
    "FORCE_GROUPS",
    "TopologyError",
    "create_integrator",
    "create_system",
    "create_topology",
    "lj_tables",
    "load",
    "molecule_exclusions",
    "read_topology",
]
