"""
Inspect a multi-eGO parameter file: which terms it contains, and what they
contribute on a given structure.

Usage::

    python -m multiego.omm.inspect outputs/NAME/mg_1/mego_params.h5 [conf.gro] [--molecules NAME:N,...]

It answers the questions "is the hydrogen-bond term really in my model?" and
"is the CMAP applied, and to which residues?" without running a simulation:

* the generation options stored in the file (egos, CMAP, hydrogen bond);
* for every molecule type: atoms, residues, hydrogen-bond donors/acceptors
  and the residues that carry a CMAP (and the residue names that do NOT have
  a map, which silently get no correction);
* the forces of the OpenMM System built from it;
* with a structure: the energy of every term, and the number of backbone
  hydrogen bonds by geometry, so that a hydrogen-bond energy of zero on a
  structure that does contain hydrogen bonds is immediately visible.
"""

import argparse
import json
from collections import Counter

import numpy as np
import openmm.app as app

from multiego.omm import builder, energy, hbond, paramfile
from multiego.omm.simulate import parse_molecules


def describe(params, molecules=None):
    """Print the content of a parameter file and of the System built from it."""
    attrs = params.attrs
    arguments = json.loads(attrs.get("arguments", "{}"))
    print(f"parameter file format {attrs.get('format_version')}, egos = {attrs.get('egos')}")
    print(
        f"generated with: --hbond {arguments.get('hbond')}  --hbond_epsilon {arguments.get('hbond_epsilon')}  --cmap '{arguments.get('cmap', '')}'"
    )

    settings = params.hbond
    if settings is None:
        print("HYDROGEN BOND: NOT in this model (generate with --hbond)")
    else:
        depth = "from each H-O LJ pair" if settings.epsilon is None else f"{settings.epsilon} kJ/mol"
        nonlocal_depth = depth if settings.epsilon_nonlocal is None else f"{settings.epsilon_nonlocal} kJ/mol"
        depth = (
            f"{depth} local (|dres| {settings.min_residue_separation}-{settings.nonlocal_separation - 1}), "
            f"{nonlocal_depth} non-local (|dres| >= {settings.nonlocal_separation} or between chains)"
        )
        print(
            f"HYDROGEN BOND: present, epsilon {depth}, r0 {settings.r0} nm, N-H...O {settings.donor_theta0} deg, "
            f"H...O=C {settings.acceptor_theta0} deg, min residue separation {settings.min_residue_separation}"
        )

    maps = params.cmap.map_order() if params.cmap is not None else []
    print(f"CMAP: {'maps for ' + ', '.join(maps) if maps else 'NOT in this model (generate with --cmap DIR)'}")

    for name, mol in params.molecules.items():
        residues = sorted({(int(r), str(n)) for r, n in zip(mol.atoms["resnum"], mol.atoms["resname"])})
        donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
        with_cmap = 0
        if params.cmap is not None:
            with_cmap = int(np.sum(params.cmap.residues["molecule"] == name))
        missing = Counter(n for _, n in residues if n not in maps)
        print(
            f"molecule {name}: {mol.n_atoms} atoms, {len(residues)} residues, "
            f"{len(donors['h'])} H-bond donors, {len(acceptors['o'])} acceptors, {with_cmap} residues with CMAP"
        )
        if maps and missing:
            print(f"   residues WITHOUT a CMAP (no correction): {dict(missing)}")
        if settings is not None and (len(donors["h"]) == 0 or len(acceptors["o"]) == 0):
            print("   WARNING: no backbone donors/acceptors found: the hydrogen-bond term has nothing to act on")
            print("            (it looks for atoms named H/N and O/C in the same residue)")


def benchmark(system, positions, platform_name, repeats):
    """
    Time the force evaluation of every term separately on a platform.

    Each force is put in its own force group; forces and energies of one
    group at a time are requested ``repeats`` times.  The sum over terms is
    close to the cost of one MD step (the integrator itself is cheap).
    """
    import time

    import openmm as mm

    system = mm.XmlSerializer.clone(system)
    names = []
    for group, force in enumerate(system.getForces()):
        force.setForceGroup(group)
        names.append(force.getName())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform_name))
    context.setPositions(positions)
    context.getState(getForces=True)  # compile kernels / build neighbour lists
    print(f"benchmark on {platform_name}, {system.getNumParticles()} atoms, {repeats} evaluations per term:")
    timings = []
    for group, name in enumerate(names):
        context.getState(getForces=True, groups={group})
        start = time.time()
        for _ in range(repeats):
            context.getState(getForces=True, groups={group})
        timings.append(((time.time() - start) / repeats * 1000.0, name))
    total = sum(t for t, _ in timings)
    for t, name in sorted(timings, reverse=True):
        print(f"   {name:20s} {t:9.3f} ms  ({100 * t / total:5.1f}%)")
    print(
        f"   {'total':20s} {total:9.3f} ms per step  ->  about {0.005 * 86400 / (total / 1000) / 1000:.1f} ns/day at 5 fs"
    )
    for force in system.getForces():
        if force.getName() == "hbonds":
            pairs = force.getNumDonors() * force.getNumAcceptors()
            print(
                f"   (H-bond term: {force.getNumDonors()} donors x {force.getNumAcceptors()} acceptors = {pairs / 1e6:.1f} M pairs per step, no neighbour list)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("params", help="mego_params.h5")
    parser.add_argument("conf", nargs="?", help="optional structure (.gro) to evaluate the terms on")
    parser.add_argument("--molecules", default="", help="composition as NAME:COUNT,... (as in mego_simulate.py)")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="time every term on --platform (default CPU) over this many force evaluations, to find what "
        "limits the speed of a simulation",
    )
    parser.add_argument(
        "--platform",
        default="",
        help="also evaluate every term on this OpenMM platform (e.g. CUDA, OpenCL, CPU) and compare it with the "
        "Reference platform: a check that custom forces (H-bond, CMAP, tabulated LJ) behave the same on the GPU",
    )
    args = parser.parse_args()

    params = paramfile.read(args.params)
    describe(params)
    if not args.conf:
        return

    gro = app.GromacsGroFile(args.conf)
    system = builder.create_system(params, parse_molecules(args.molecules), box_vectors=gro.getPeriodicBoxVectors())
    print("forces in the System:", ", ".join(f.getName() for f in system.getForces()))
    terms = energy.decompose(system, gro.getPositions())
    if args.benchmark:
        benchmark(system, gro.getPositions(), args.platform or "CPU", args.benchmark)
        return
    other = energy.decompose(system, gro.getPositions(), args.platform) if args.platform else None
    print(f"energies on {args.conf} (kJ/mol)" + (f", Reference vs {args.platform}:" if other else ":"))
    for term in energy.TERMS:
        if term in terms:
            line = f"   {term:20s} {terms[term]['energy']:12.3f}"
            if other:
                e = other[term]["energy"]
                df = np.abs(other[term]["forces"] - terms[term]["forces"]).max()
                flag = (
                    "" if abs(e - terms[term]["energy"]) < 1e-3 * max(1.0, abs(e)) and df < 1.0 else "   <-- DIFFERENT"
                )
                line += f" {e:12.3f}   max force difference {df:8.3f} kJ/mol/nm{flag}"
            print(line)

    # Geometric count of backbone hydrogen bonds, independent of the term.
    x = np.array(gro.getPositions()._value)
    count, offset = 0, 0
    for name, copies in parse_molecules(args.molecules) or [(n, 1) for n in params.molecules]:
        mol = params.molecules[name]
        donors, acceptors = hbond.find_donors_acceptors(mol.atoms)
        for _ in range(copies):
            for h, n in zip(donors["h"] + offset, donors["n"] + offset):
                for o in acceptors["o"] + offset:
                    r = np.linalg.norm(x[h] - x[o])
                    u, v = x[n] - x[h], x[o] - x[h]
                    if r < 0.25 and np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) > 120:
                        count += 1
            offset += mol.n_atoms
    print(f"backbone hydrogen bonds in the structure (H...O < 0.25 nm, N-H...O > 120 deg): {count}")


if __name__ == "__main__":
    main()
