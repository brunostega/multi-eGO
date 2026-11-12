"""
Fit a backbone CMAP for a blocked amino acid (Ace-Xaa-Nme) by iterative
Boltzmann inversion against a target (phi, psi) distribution.

Usage
-----
::

    python tools/cmap_fit/fit_cmap.py --params OUT/mego_params.h5 --conf X.gro \\
        --target ALA.csv --resname ALA --output ALA_cmap.npz \\
        --iterations 8 --ns 20

At every iteration the dipeptide is simulated with the current CMAP, the
sampled (phi, psi) distribution is compared with the target and the map is
updated (see ``multiego.cmap``).  The resulting ``.npz`` holds the map, the
sampled angles of the last iteration and the convergence history.

Why the dipeptide can be sampled exhaustively
---------------------------------------------
A blocked amino acid has ~12-20 atoms and only two slow degrees of freedom
(three for residues with a rotatable side chain), so plain molecular dynamics
at 300 K crosses all the barriers of the Ramachandran map many times within a
few nanoseconds.  No enhanced sampling is needed; the runs below are long
enough that the statistical error on the basin populations is well under a
percent.
"""

import argparse
import os
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))

from multiego import cmap  # noqa: E402
from multiego.omm import builder, paramfile  # noqa: E402
from single_molecule import create_single_molecule_system  # noqa: E402


def torsion_indices(params):
    """The five atoms (c0, n, ca, c, n1) of the single CMAP residue."""
    residues = params.cmap.residues
    if len(residues["index"]) != 1:
        raise ValueError(f"expected exactly one CMAP residue, found {len(residues['index'])}")
    return [int(residues[column][0]) for column in ("c0", "n", "ca", "c", "n1")]


def dihedrals(positions, i, j, k, m):
    """Dihedral angles (degrees) of a trajectory, shape (frames, atoms, 3)."""
    b0 = positions[:, i] - positions[:, j]
    b1 = positions[:, k] - positions[:, j]
    b2 = positions[:, m] - positions[:, k]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1)))


def sample(params, positions, box, grid, temperature, nsteps, interval, seed, platform, fast=True):
    """
    Run the dipeptide with ``grid`` as its CMAP and return the sampled angles.

    Positions are propagated in place between iterations (the final frame of
    one iteration starts the next), which removes any equilibration transient
    after the first iteration.
    """
    params.cmap.maps[next(iter(params.cmap.maps))] = grid
    system = create_single_molecule_system(params, box) if fast else builder.create_system(params, box_vectors=box)
    integrator = mm.LangevinMiddleIntegrator(temperature * u.kelvin, 0.04 / u.picosecond, 0.005 * u.picosecond)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature * u.kelvin, seed)

    atoms = torsion_indices(params)
    frames = []
    for _ in range(nsteps // interval):
        integrator.step(interval)
        frames.append(context.getState(getPositions=True).getPositions(asNumpy=True)._value)
    frames = np.array(frames)
    phi = dihedrals(frames, *atoms[:4])
    psi = dihedrals(frames, *atoms[1:])
    return phi, psi, frames[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="parameter file of the blocked amino acid")
    parser.add_argument("--conf", required=True, help="starting coordinates (.gro)")
    parser.add_argument("--target", required=True, help="target landscape (144x144 CSV)")
    parser.add_argument("--resname", required=True, help="residue name the CMAP applies to")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--size", type=int, default=36, help="CMAP grid size (must divide 144)")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--ns", type=float, default=20.0, help="simulated time per iteration (ns)")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--interval", type=int, default=100, help="steps between stored frames")
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument(
        "--max-step", type=float, default=5.0, help="largest change of a cell in one iteration (kJ/mol)"
    )
    parser.add_argument(
        "--average",
        type=int,
        default=4,
        help="average the maps of the last N iterations instead of taking the last one. "
        "Each iteration injects the statistical noise of its own run into the map; "
        "once the fit has converged the maps fluctuate around the right answer, so "
        "averaging them is the cheapest way to remove that noise.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--platform", default="Reference")
    args = parser.parse_args()

    target = cmap.load_target(args.target, args.size)
    populations = cmap.basin_populations(target)
    print(
        f"target {args.resname}: "
        + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in populations.items())
        + f"   <3J> {cmap.mean_j(target):.2f} Hz"
    )

    params = paramfile.read(args.params)
    gro = app.GromacsGroFile(args.conf)
    box = gro.getPeriodicBoxVectors()
    grid = np.zeros((args.size, args.size))
    params.cmap = paramfile.cmap_from_maps(params.molecules, {args.resname: grid})

    positions = gro.getPositions()
    nsteps = int(args.ns * 1000 / 0.005)
    history, maps = [], []
    for iteration in range(args.iterations):
        start = time.time()
        phi, psi, positions = sample(
            params, positions, box, grid, args.temperature, nsteps, args.interval, args.seed + iteration, args.platform
        )
        sampled = cmap.histogram(phi, psi, args.size)
        populations = cmap.basin_populations(sampled)
        # Divergence from the target, on the sampled distribution produced by
        # the *current* map: this is the quantity that must go to zero.
        overlap = float(np.minimum(sampled, target).sum())
        history.append({"overlap": overlap, **populations, "j": cmap.mean_j(sampled)})
        grid, rms = cmap.update(grid, sampled, target, args.temperature, damping=args.damping, max_step=args.max_step)
        maps.append(grid)
        print(
            f"  iter {iteration}: "
            + "  ".join(f"{k} {100 * populations[k]:5.1f}%" for k in ("alphaR", "PPII", "beta"))
            + f"  <3J> {history[-1]['j']:.2f}  overlap {overlap:.3f}"
            + f"  |dU| {rms:.2f} kJ/mol  ({time.time() - start:.0f} s, {len(phi)} frames)"
        )

    # Average the converged maps, then remove the artificial wells left in
    # never-sampled regions, and run once more to report what it produces.
    if args.average > 1:
        grid = np.mean(maps[-args.average :], axis=0)
    grid = cmap.flatten_unpopulated(grid, target)

    # One last run with the final map, to report what it actually produces.
    phi, psi, positions = sample(
        params, positions, box, grid, args.temperature, nsteps, args.interval, args.seed + 999, args.platform
    )
    sampled = cmap.histogram(phi, psi, args.size)
    final = cmap.basin_populations(sampled)
    print(
        f"final  {args.resname}: "
        + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in final.items())
        + f"   <3J> {cmap.mean_j(sampled):.2f} Hz"
        + f"   overlap {float(np.minimum(sampled, target).sum()):.3f}"
    )
    np.savez(
        args.output,
        cmap=grid,
        target=target,
        sampled=sampled,
        phi=phi.astype(np.float32),
        psi=psi.astype(np.float32),
        resname=args.resname,
        size=args.size,
        temperature=args.temperature,
        history=np.array([[h[k] for k in ("overlap", "alphaR", "PPII", "beta", "j")] for h in history]),
    )
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
