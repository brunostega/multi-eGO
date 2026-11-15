"""
Independent validation of a fitted backbone CMAP on its blocked amino acid.

    python tools/cmap_fit/validate_cmap.py --params P.h5 --conf X_box.gro --target X.csv \\
        --map X.npz --resname XXX [--ns 50] [--flat-ns 20]

Two runs with the map, both independent of the fit (new seeds, new starting
velocities):

* an unbiased run of --ns nanoseconds: basin populations (alphaR, PPII, beta)
  with block-averaged standard errors;
* a run of --flat-ns nanoseconds with the flattening bias of the fit,
  reweighted: the free energy of alphaL relative to alphaR, which an
  unbiased run cannot resolve (target population ~0.1-0.3%).

Acceptance criteria (printed as PASS/FAIL):
* alphaR, PPII, beta: |model - target| <= max(2 percentage points, 2 SE);
* alphaL: |dG_model - dG_target| <= 1 kJ/mol, with
  dG = -RT ln(P_alphaL / P_alphaR).
"""

import argparse
import os
import sys

import numpy as np
import openmm.app as app

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "src"))

from multiego import cmap  # noqa: E402
from multiego.omm import paramfile  # noqa: E402
import fit_cmap  # noqa: E402

BASINS = ("alphaR", "PPII", "beta")


def block_populations(phi, psi, size, blocks=5, weights=None):
    """Basin populations of each block of the trajectory."""
    out = []
    for idx in np.array_split(np.arange(len(phi)), blocks):
        w = None if weights is None else weights[idx]
        out.append(cmap.basin_populations(cmap.histogram(phi[idx], psi[idx], size, w)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True)
    parser.add_argument("--conf", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--resname", required=True)
    parser.add_argument("--ns", type=float, default=50.0)
    parser.add_argument("--flat-ns", type=float, default=20.0)
    parser.add_argument("--flatten", type=float, default=12.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    grid = np.array(np.load(args.map)["cmap"], dtype=float)
    size = grid.shape[0]
    target = cmap.load_target(args.target, size)
    params = paramfile.read(args.params)
    params.cmap = paramfile.cmap_from_maps(params.molecules, {args.resname: grid})
    gro = app.GromacsGroFile(args.conf)
    box, kt = gro.getPeriodicBoxVectors(), cmap.KB * args.temperature
    steps = lambda ns: int(ns * 1000 / 0.005)  # noqa: E731

    reference = cmap.basin_populations(target)
    verdict = True
    if args.ns > 0:
        phi, psi, _ = fit_cmap.sample(
            params, gro.getPositions(), box, grid, args.temperature, steps(args.ns), 100, args.seed, "Reference"
        )
        model = cmap.basin_populations(cmap.histogram(phi, psi, size))
        blocks = block_populations(phi, psi, size)
        reference = cmap.basin_populations(target)

        verdict = True
        print(f"{args.resname}: unbiased run {args.ns:.0f} ns ({len(phi)} frames)")
        print(f"{'basin':8s} {'target':>8s} {'model':>8s} {'SE':>6s} {'diff':>7s}  result")
        for b in BASINS:
            se = np.std([x[b] for x in blocks], ddof=1) / np.sqrt(len(blocks))
            diff = model[b] - reference[b]
            ok = abs(diff) <= max(0.02, 2 * se)
            verdict &= ok
            print(
                f"{b:8s} {100 * reference[b]:8.1f} {100 * model[b]:8.1f} {100 * se:6.1f} {100 * diff:+7.1f}  {'PASS' if ok else 'FAIL'}"
            )
    else:
        print(f"{args.resname}: unbiased run skipped (--ns 0)")

    bias = cmap.flattening_bias(target, args.temperature, cap=args.flatten)
    phi_b, psi_b, _ = fit_cmap.sample(
        params,
        gro.getPositions(),
        box,
        grid,
        args.temperature,
        steps(args.flat_ns),
        100,
        args.seed + 1,
        "Reference",
        bias=bias,
    )
    weights = cmap.bias_weights(phi_b, psi_b, bias, args.temperature)
    biased = cmap.basin_populations(cmap.histogram(phi_b, psi_b, size, weights))
    g_model = -kt * np.log(biased["alphaL"] / biased["alphaR"])
    g_target = -kt * np.log(reference["alphaL"] / reference["alphaR"])
    g_blocks = [
        -kt * np.log(x["alphaL"] / x["alphaR"])
        for x in block_populations(phi_b, psi_b, size, weights=weights)
        if x["alphaL"] > 0
    ]
    se_g = np.std(g_blocks, ddof=1) / np.sqrt(len(g_blocks)) if len(g_blocks) > 1 else float("nan")
    ok = abs(g_model - g_target) <= 1.0
    verdict &= ok
    print(
        f"alphaL (reweighted {args.flat_ns:.0f} ns): population {100 * biased['alphaL']:.2f}% (target {100 * reference['alphaL']:.2f}%), "
        f"dG(aL-aR) {g_model:.2f} +- {se_g:.2f} vs {g_target:.2f} kJ/mol  {'PASS' if ok else 'FAIL'}"
    )
    # Number of separate visits to alphaL (phi > 0) that last at least 5 ps:
    # sets the real statistical resolution of the alphaL estimate.
    inside = phi_b > 0
    entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
    lasting = sum(1 for e in entries if inside[e : e + 10].all())  # noqa: E203
    print(f"visits to phi > 0 lasting >= 5 ps in the reweighted run: {lasting}")
    print("cross-check, reweighted run: " + "  ".join(f"{b} {100 * biased[b]:.1f}" for b in BASINS))
    print("OVERALL:", "PASS" if verdict else "FAIL")


if __name__ == "__main__":
    main()
