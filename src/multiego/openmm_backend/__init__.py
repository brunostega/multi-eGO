"""
OpenMM backend for multi-eGO models.

- ``paramfile``: the OpenMM parameter file ``mego_params.h5`` written by
  ``multiego.py`` (needs h5py, not OpenMM).
- ``cmap``: backbone CMAP maps.
- ``gromacs_reader``: strict reader of the GROMACS files multi-eGO writes.
- ``builder``: OpenMM System from either of them (needs OpenMM, installed with
  ``pip install "mego[openmm]"``; add ``openmm[cuda12]`` or similar for GPUs).

Typical use (``openmm/run.py`` does this)::

    from multiego.openmm_backend import load_params, create_integrator
    system, topology, positions = load_params("mego_params.h5", "conf.gro")
    integrator = create_integrator(300.0)
"""

from .gromacs_reader import TopologyError, read_topology
from .cmap import read_cmap, write_cmap

_BUILDER_NAMES = (
    "FORCE_GROUPS",
    "create_integrator",
    "create_system",
    "create_system_from_params",
    "create_topology",
    "create_topology_from_params",
    "lj_tables",
    "load",
    "load_params",
    "molecule_exclusions",
)


def __getattr__(name):
    # The builder needs OpenMM; import it only when one of its names is used, so
    # that the generator can write the parameter file without OpenMM installed.
    if name in _BUILDER_NAMES:
        from . import builder

        return getattr(builder, name)
    raise AttributeError(name)


__all__ = ["TopologyError", "read_cmap", "read_topology", "write_cmap", *_BUILDER_NAMES]
