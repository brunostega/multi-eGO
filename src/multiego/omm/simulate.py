"""
Run a multi-eGO simulation with OpenMM (the equivalent of ``gmx mdrun``).

This replaces the GROMACS ``mdps/`` files: every default below is taken from
``mdps/ff_aa.mdp`` (dynamics) and ``mdps/ff_cg.mdp`` (minimisation), so a run
with default options is the OpenMM version of the standard multi-eGO protocol.

=========================  ===================================  =================
GROMACS (.mdp)             OpenMM                               option
=========================  ===================================  =================
integrator = sd            LangevinMiddleIntegrator             --
tau_t = 25 ps              friction = 1/tau_t = 0.04 ps^-1      --friction
dt = 0.005 ps              5 fs                                 --dt
ref_t                      temperature                          --temperature
rvdw = 2.5, shift, no DC   see multiego.omm.builder             --cutoff
comm_mode = linear, 20     CMMotionRemover(20)                  --
nstxout-compressed=50000   XTCReporter                          --xtc-interval
nstlog / nstenergy=50000   StateDataReporter                    --log-interval
integrator = cg/steep      LocalEnergyMinimizer (L-BFGS),       --minimize
  emtol = 10                 tolerance 10 kJ/mol/nm             --em-tolerance
constraints = none         no constraints                       --
define = -DPOSRES          harmonic heavy-atom restraints to    --posre
  (ff_aa-posre.mdp)          the --conf positions (posre.itp of
                             pdb2gmx: 1000 kJ/mol/nm^2)
=========================  ===================================  =================

The standard protocol of the README (ff_em -> ff_cg -> ff_aa-posre -> ff_aa)
therefore becomes: one run with ``--minimize --posre 1000`` (equilibration),
then a run from its final configuration without ``--posre`` (production).

Note: the GROMACS ``sd`` integrator and OpenMM's Langevin middle integrator
sample the same (canonical) ensemble with the same friction, but are different
discretisations: trajectories are statistically, not numerically, equivalent.

Outputs (``<output>`` is the --output prefix)
---------------------------------------------
``<output>.xtc``        compressed trajectory (readable by cmdata / GROMACS tools)
``<output>.log``        energies, temperature, speed (CSV)
``<output>.chk``        checkpoint for restarts (``--restart``)
``<output>_final.gro``  final configuration
``<output>_final.pdb``  final configuration with topology (for visualisation)

Example
-------
::

    python -m multiego.omm.simulate --params outputs/gpref/mg_1/mego_params.h5 \\
        --conf gb1.gro --temperature 300 --nsteps 1000000 --minimize --output gb1_mg
"""

import argparse
import os
import sys
import time

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from multiego.omm import builder, paramfile


def parse_molecules(text):
    """Parse ``"GB1:2,LIG:10"`` into ``[("GB1", 2), ("LIG", 10)]``."""
    if not text:
        return None
    molecules = []
    for item in text.split(","):
        name, _, count = item.partition(":")
        molecules.append((name, int(count) if count else 1))
    return molecules


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--params", required=True, help="multi-eGO parameter file (mego_params.h5)")
    p.add_argument("--conf", required=True, help="starting coordinates and box (.gro)")
    p.add_argument("--output", required=True, help="prefix of the output files")
    p.add_argument(
        "--molecules",
        default="",
        help="system composition as NAME:COUNT,... in coordinate order (default: one copy of each molecule type)",
    )
    p.add_argument("--temperature", type=float, default=300.0, help="temperature in K (ref_t)")
    p.add_argument("--nsteps", type=int, default=0, help="number of MD steps (0: minimisation only)")
    p.add_argument("--dt", type=float, default=0.005, help="time step in ps")
    p.add_argument("--friction", type=float, default=1.0 / 25.0, help="Langevin friction in 1/ps (1/tau_t)")
    p.add_argument("--cutoff", type=float, default=builder.DEFAULT_CUTOFF, help="non-bonded cut-off in nm (rvdw)")
    p.add_argument("--minimize", action="store_true", help="minimise the energy before the dynamics")
    p.add_argument(
        "--posre",
        type=float,
        default=0.0,
        help="heavy-atom position restraints to the --conf positions, force constant in kJ/mol/nm^2 "
        "(pdb2gmx default: 1000; 0: off)",
    )
    p.add_argument("--em-tolerance", type=float, default=10.0, help="minimisation tolerance in kJ/mol/nm (emtol)")
    p.add_argument("--xtc-interval", type=int, default=50000, help="steps between trajectory frames")
    p.add_argument("--log-interval", type=int, default=50000, help="steps between log lines")
    p.add_argument("--checkpoint-interval", type=int, default=500000, help="steps between checkpoints")
    p.add_argument("--restart", default="", help="checkpoint file to continue from (positions, velocities, step)")
    p.add_argument("--seed", type=int, default=0, help="random seed (0: chosen by OpenMM)")
    p.add_argument("--platform", default="", help="OpenMM platform (CUDA, OpenCL, CPU, Reference; default: fastest)")
    return p


def write_gro(path, topology, positions, box_vectors, title="multi-eGO OpenMM"):
    """
    Write coordinates in GROMACS .gro format (OpenMM has no .gro writer).

    Only a rectangular box is supported, which is what multi-eGO uses.
    """
    xyz = positions.value_in_unit(unit.nanometer)
    box = [box_vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
    with open(path, "w") as f:
        f.write(f"{title}\n{topology.getNumAtoms()}\n")
        for atom, (x, y, z) in zip(topology.atoms(), xyz):
            # .gro fixed columns: residue number/name, atom name/number wrap at 100000
            f.write(
                f"{int(atom.residue.id) % 100000:5d}{atom.residue.name:<5.5s}{atom.name:>5.5s}"
                f"{(atom.index + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")


def add_position_restraints(system, topology, positions, k):
    """
    Restrain every heavy atom to its reference position (GROMACS posre.itp).

    ``E = k/2 |r - r_ref|^2`` with the minimum-image distance, applied to all
    atoms except hydrogens, like the posre.itp written by pdb2gmx.
    """
    force = mm.CustomExternalForce("0.5*k_posre*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k_posre", k)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    xyz = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.element is not None and atom.element.atomic_number > 1:
            force.addParticle(atom.index, list(xyz[atom.index]))
    force.setName("position_restraints")
    system.addForce(force)


def run(args):
    """Set up and run the simulation described by the parsed arguments."""
    params = paramfile.read(args.params)
    gro = app.GromacsGroFile(args.conf)
    box = gro.getPeriodicBoxVectors()
    molecules = parse_molecules(args.molecules)

    topology = builder.create_topology(params, molecules, box)
    system = builder.create_system(params, molecules, box_vectors=box, cutoff=args.cutoff)
    if system.getNumParticles() != len(gro.getPositions()):
        raise ValueError(
            f"{args.conf} has {len(gro.getPositions())} atoms but the system has {system.getNumParticles()}: "
            "check --molecules"
        )

    if args.posre > 0:
        add_position_restraints(system, topology, gro.getPositions(), args.posre)

    integrator = mm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin, args.friction / unit.picosecond, args.dt * unit.picosecond
    )
    if args.seed:
        integrator.setRandomNumberSeed(args.seed)
    platform = [mm.Platform.getPlatformByName(args.platform)] if args.platform else []
    simulation = app.Simulation(topology, system, integrator, *platform)
    print(f"Platform: {simulation.context.getPlatform().getName()}, {system.getNumParticles()} atoms")

    if args.restart:
        # Positions, velocities, box and step counter from the checkpoint.
        simulation.loadCheckpoint(args.restart)
    else:
        simulation.context.setPositions(gro.getPositions())
        if args.minimize:
            e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            simulation.minimizeEnergy(tolerance=args.em_tolerance * unit.kilojoule_per_mole / unit.nanometer)
            e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            print(f"Minimisation: {e0} -> {e1}")
        # No velocities in the input (.gro readers ignore them): GROMACS
        # gen_vel = yes equivalent.  Use --restart to continue a run instead.
        simulation.context.setVelocitiesToTemperature(
            args.temperature * unit.kelvin, *([args.seed] if args.seed else [])
        )

    if args.nsteps > 0:
        append = bool(args.restart)
        simulation.reporters.append(app.XTCReporter(f"{args.output}.xtc", args.xtc_interval, append=append))
        simulation.reporters.append(
            app.StateDataReporter(
                f"{args.output}.log",
                args.log_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                speed=True,
                append=append,
            )
        )
        simulation.reporters.append(app.CheckpointReporter(f"{args.output}.chk", args.checkpoint_interval))
        t0 = time.time()
        simulation.step(args.nsteps)
        simulation.saveCheckpoint(f"{args.output}.chk")
        print(f"{args.nsteps} steps in {time.time() - t0:.1f} s")

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
    write_gro(f"{args.output}_final.gro", topology, state.getPositions(), state.getPeriodicBoxVectors())
    with open(f"{args.output}_final.pdb", "w") as f:
        app.PDBFile.writeFile(topology, state.getPositions(), f)
    print(f"Final configuration written to {args.output}_final.gro/.pdb")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.params):
        sys.exit(f"parameter file {args.params} not found")
    run(args)


if __name__ == "__main__":
    main()
