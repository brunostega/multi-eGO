"""
Run a multi-eGO simulation with OpenMM (the equivalent of ``gmx mdrun``).

Reads the OpenMM parameter file written by ``multiego.py`` (``mego_params.h5``,
in the output folder next to ``topol_mego.top`` and ``ffnonbonded.itp``) and a
starting ``.gro``. Backbone CMAPs stored in the file (``--backbone cmap``) are
applied.

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
rvdw = 2.5, shift, no DC   see multiego.openmm_backend.builder  --cutoff
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
``<output>_energy.csv`` energy of every term (the GROMACS terms + CMAP), like the .edr
``<output>.chk``        checkpoint for restarts (``--restart``)
``<output>_final.gro``  final configuration
``<output>_final.pdb``  final configuration with topology (for visualisation)

Example
-------
::

    python openmm/run.py --params outputs/gpref/mg_1 --conf gb1.gro \\
        --temperature 300 --nsteps 1000000 --minimize --output gb1_mg
"""

import argparse
import os
import sys
import time

import openmm as mm
import openmm.app as app
import openmm.unit as unit

try:
    from multiego.openmm_backend import builder, paramfile
except ImportError:  # running from a source checkout without installing
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from multiego.openmm_backend import builder, paramfile


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
    p.add_argument(
        "--params",
        required=True,
        help="multi-eGO parameter file (mego_params.h5), or the multiego.py output folder that contains it",
    )
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
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU platform only: number of threads (0: all cores). GPU platforms use one host thread per "
        "simulation; to use more of a GPU, run several simulations on it at the same time.",
    )
    p.add_argument("--device", default="", help="CUDA/OpenCL: GPU index (e.g. 0, or 0,1); default: chosen by OpenMM")
    p.add_argument(
        "--precision",
        default="",
        choices=["", "single", "mixed", "double"],
        help="CUDA/OpenCL: floating-point precision (default mixed). 'single' is faster but less accurate "
        "for energy conservation; 'mixed' is the safe choice for production.",
    )
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


def platform_and_properties(args):
    """
    Platform and platform properties for app.Simulation.

    * CPU: ``Threads`` (all cores by default).  This is the only platform
      where the number of threads controls the speed of one simulation.
    * CUDA/OpenCL: ``DeviceIndex`` and ``Precision``.  The force calculation
      runs on the GPU; the host uses one thread per simulation, so threads
      are not a setting there.  Small systems (a peptide, a single protein)
      leave most of a GPU idle: running several independent simulations on
      the same GPU at the same time (e.g. replicas, temperatures, seeds)
      raises the total throughput, and NVIDIA MPS helps further when many
      processes share one GPU.
    """
    if not args.platform:
        if args.threads or args.device or args.precision:
            raise SystemExit("--threads, --device and --precision require --platform")
        return []
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform == "CPU":
        if args.device or args.precision:
            raise SystemExit("--device and --precision apply to CUDA/OpenCL, not to the CPU platform")
        if args.threads:
            properties["Threads"] = str(args.threads)
    elif args.platform in ("CUDA", "OpenCL"):
        if args.threads:
            raise SystemExit("--threads applies to the CPU platform only (GPU platforms use one host thread)")
        if args.device:
            properties["DeviceIndex"] = args.device
        if args.precision:
            properties["Precision"] = args.precision
    return [platform, properties]


class EnergyTermReporter:
    """CSV with the energy of every force group (kJ/mol), like ``gmx energy`` on the .edr."""

    def __init__(self, path, interval, system, append=False):
        self._interval = interval
        self._groups = {}
        for force in system.getForces():
            if force.getName() in builder.FORCE_GROUPS:
                self._groups[force.getName()] = force.getForceGroup()
        self._file = open(path, "a" if append else "w")
        if not append:
            self._file.write(",".join(["step", "time_ps"] + list(self._groups) + ["potential"]) + "\n")

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return {"steps": steps, "periodic": None, "include": []}

    def report(self, simulation, state):
        ctx = simulation.context
        values = [
            ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            for g in self._groups.values()
        ]
        total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        t = state.getTime().value_in_unit(unit.picosecond)
        self._file.write(
            ",".join([str(simulation.currentStep), f"{t:.3f}"] + [f"{v:.6f}" for v in values + [total]]) + "\n"
        )
        self._file.flush()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


def describe(params, molecules):
    """Short summary of the model read from the parameter file."""
    composition = molecules or [(name, 1) for name in params.molecules]
    print(
        "Model:",
        params.attrs.get("system", "?"),
        f"({params.attrs.get('egos', '?')}),",
        "molecules:",
        ", ".join(f"{n} x{c}" for n, c in composition),
    )
    print(f"  backbone: {params.attrs.get('backbone', 'gromos')}", end="")
    if params.cmap is not None:
        print(f", CMAP on {len(params.cmap.residues['index'])} residues, maps: {', '.join(params.cmap.map_order())}")
    else:
        print()


def resolve_params(path):
    """Accept the parameter file or the output folder that contains it."""
    if os.path.isdir(path):
        path = os.path.join(path, paramfile.DEFAULT_FILENAME)
    if not os.path.exists(path):
        sys.exit(f"parameter file {path} not found")
    return path


def run(args):
    """Set up and run the simulation described by the parsed arguments."""
    params = paramfile.read(resolve_params(args.params))
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    gro = app.GromacsGroFile(args.conf)
    box = gro.getPeriodicBoxVectors()
    molecules = parse_molecules(args.molecules)

    box_nm = [v.value_in_unit(unit.nanometer) for v in box]
    topology = builder.create_topology_from_params(params, molecules, box_nm)
    system = builder.create_system_from_params(params, box_nm, molecules, cutoff=args.cutoff)
    if system.getNumParticles() != len(gro.getPositions()):
        raise ValueError(
            f"{args.conf} has {len(gro.getPositions())} atoms but the system has {system.getNumParticles()}: "
            "check --molecules"
        )

    if args.posre > 0:
        add_position_restraints(system, topology, gro.getPositions(), args.posre)

    integrator = builder.create_integrator(
        args.temperature, timestep=args.dt, friction=args.friction, seed=args.seed or None
    )
    simulation = app.Simulation(topology, system, integrator, *platform_and_properties(args))
    used = simulation.context.getPlatform()
    details = {
        name: used.getPropertyValue(simulation.context, name)
        for name in ("Threads", "DeviceIndex", "DeviceName", "Precision")
        if name in used.getPropertyNames()
    }
    print(f"Platform: {used.getName()} {details}, {system.getNumParticles()} atoms")
    print("Forces:", ", ".join(f.getName() for f in system.getForces()))
    describe(params, molecules)

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
        simulation.reporters.append(EnergyTermReporter(f"{args.output}_energy.csv", args.log_interval, system, append))
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
    run(args)


if __name__ == "__main__":
    main()
