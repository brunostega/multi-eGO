"""
Temperature replica exchange for one isolated molecule (OpenMM, multi-eGO).

Why
---
Once the backbone CMAP makes helix nucleation as rare as it should be, plain
molecular dynamics no longer equilibrates a peptide on the nanosecond scale:
runs started from a helix and from an extended chain give different
helicities (hysteresis).  Replica exchange couples simulations at several
temperatures, so that configurations cross the barriers at high temperature
and are carried down by the exchanges.  It also gives the temperature
dependence -- the melting curve -- from a single run.

Scheme
------
One Context per temperature, all sharing the same System.  Every
``--exchange`` steps, neighbouring temperatures attempt to swap their
configurations with the Metropolis criterion

    P = min(1, exp[(1/kT_i - 1/kT_j) (U_i - U_j)])

(alternating the pairs (0,1),(2,3),... and (1,2),(3,4),...).  On a swap the
velocities are rescaled by sqrt(T_new/T_old).  Each temperature therefore
samples its own canonical ensemble, and the frames saved for a temperature
can be analysed as an ordinary trajectory.

To detect lack of convergence the replicas are started from different
structures (``--conf`` takes several files, assigned cyclically), and the
output records which starting structure every saved frame descends from: at
equilibrium the result must not depend on it.

Output (``.npz``): ``temperatures``, ``positions`` (temperature, frame, atom,
xyz; nm), ``walker`` (temperature, frame): index of the replica occupying
that temperature, ``start`` (replica): index of its starting structure,
``acceptance`` (per neighbouring pair).
"""

import argparse
import os
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "cmap_fit"))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "src"))

from multiego.omm import paramfile  # noqa: E402
from single_molecule import create_single_molecule_system  # noqa: E402

KB = 0.008314462618  # kJ/mol/K


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True)
    parser.add_argument("--conf", required=True, nargs="+", help="starting structures (.gro), assigned cyclically")
    parser.add_argument("--temperatures", required=True, type=float, nargs="+")
    parser.add_argument("--ns", type=float, default=3.0, help="simulated time per replica (ns)")
    parser.add_argument("--exchange", type=int, default=500, help="steps between exchange attempts")
    parser.add_argument("--save", type=int, default=1000, help="steps between saved frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--platform",
        default="Reference",
        help="OpenMM platform for all replicas (Reference, CPU, CUDA, OpenCL). On a GPU all replicas share it.",
    )
    parser.add_argument("--device", default="", help="CUDA/OpenCL device index")
    parser.add_argument("--threads", type=int, default=0, help="CPU platform: threads per replica (0: OpenMM default)")
    args = parser.parse_args()

    params = paramfile.read(args.params)
    structures = [app.GromacsGroFile(path) for path in args.conf]
    box = structures[0].getPeriodicBoxVectors()
    system = create_single_molecule_system(params, box)
    temperatures = np.array(args.temperatures)
    n = len(temperatures)
    rng = np.random.default_rng(args.seed)

    contexts, start = [], []
    for k, temperature in enumerate(temperatures):
        integrator = mm.LangevinMiddleIntegrator(temperature * u.kelvin, 0.04 / u.picosecond, 0.005 * u.picosecond)
        integrator.setRandomNumberSeed(args.seed + 17 * k)
        properties = {}
        if args.platform == "CPU" and args.threads:
            properties["Threads"] = str(args.threads)
        if args.platform in ("CUDA", "OpenCL") and args.device:
            properties["DeviceIndex"] = args.device
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName(args.platform), properties)
        context.setPositions(structures[k % len(structures)].getPositions())
        mm.LocalEnergyMinimizer.minimize(context, 10.0)
        context.setVelocitiesToTemperature(temperature * u.kelvin, args.seed + 17 * k)
        contexts.append(context)
        start.append(k % len(structures))

    walker = np.arange(n)  # replica currently at each temperature
    cycles = int(args.ns * 1000 / 0.005) // args.exchange
    save_every = max(1, args.save // args.exchange)
    attempts, accepted = np.zeros(n - 1), np.zeros(n - 1)
    frames, walkers = [], []
    began = time.time()
    for cycle in range(cycles):
        for context in contexts:
            context.getIntegrator().step(args.exchange)
        energies = [
            c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) for c in contexts
        ]
        for i in range(cycle % 2, n - 1, 2):
            j = i + 1
            delta = (1.0 / (KB * temperatures[i]) - 1.0 / (KB * temperatures[j])) * (energies[i] - energies[j])
            attempts[i] += 1
            if delta >= 0 or rng.random() < np.exp(delta):
                accepted[i] += 1
                si = contexts[i].getState(getPositions=True, getVelocities=True)
                sj = contexts[j].getState(getPositions=True, getVelocities=True)
                scale = np.sqrt(temperatures[i] / temperatures[j])
                contexts[i].setPositions(sj.getPositions())
                contexts[i].setVelocities(sj.getVelocities(asNumpy=True) * scale)
                contexts[j].setPositions(si.getPositions())
                contexts[j].setVelocities(si.getVelocities(asNumpy=True) / scale)
                walker[i], walker[j] = walker[j], walker[i]
                energies[i], energies[j] = energies[j], energies[i]
        if (cycle + 1) % save_every == 0:
            frames.append(
                [c.getState(getPositions=True).getPositions(asNumpy=True)._value.astype(np.float32) for c in contexts]
            )
            walkers.append(walker.copy())

    np.savez_compressed(
        args.output,
        temperatures=temperatures,
        positions=np.transpose(np.array(frames), (1, 0, 2, 3)),
        walker=np.array(walkers).T,
        start=np.array(start),
        acceptance=accepted / np.maximum(attempts, 1),
    )
    print(
        f"{args.output}: {cycles * args.exchange} steps x {n} replicas in {time.time() - began:.0f} s, "
        f"acceptance {' '.join(f'{a:.2f}' for a in accepted / np.maximum(attempts, 1))}"
    )


if __name__ == "__main__":
    main()
