"""
Helix content and helix-coil parameters of a peptide trajectory.

What it measures
----------------
Helicity alone does not say whether a model forms helices *cooperatively*: a
chain whose residues each adopt alpha independently can show the same average
helicity as one that forms a few long helices.  This tool therefore reports,
for every trajectory:

* **populations** -- residues in the alpha basin, residues inside a helical
  segment (3 or more consecutive alpha residues), and i -> i+4 backbone
  hydrogen bonds (H...O < 0.25 nm and N-H...O > 120 degrees);
* **the Lifson-Roig parameters** w and v, fitted by maximum likelihood to the
  per-frame alpha/coil strings;
* **the Zimm-Bragg nucleation parameter** sigma = v^2 / (1+v)^4 and the
  propagation parameter s = w / (1+v);
* **the dispersion** D of the number of alpha residues per frame, relative to
  independent residues (binomial).  D = 1 means no cooperativity at all.

The Lifson-Roig model
---------------------
Each residue is helical (h) or coil (c).  A helical residue whose two
neighbours are also helical carries the weight ``w`` -- it closes a hydrogen
bond, so this is *propagation*; any other helical residue carries ``v`` --
it is part of a nucleus that does not yet close a bond, so this is
*nucleation*.  Coil residues carry 1.  The statistical weight of a
configuration is the product over residues, and the probability is that
weight divided by the partition function.

For a peptide of 15 guest residues the 2^15 configurations can be enumerated
exactly, so the likelihood and its gradient are computed without
approximation and maximised directly.  Experimental values for alanine-based
peptides are roughly v ~ 0.05 (sigma ~ 0.003) and w slightly above 1 near
room temperature; v is essentially entropic and nearly temperature
independent, while w carries the enthalpy of the hydrogen bond.
"""

import argparse
import itertools
import os
import sys

import numpy as np
from openmm.app.internal.xtc_utils import read_xtc
from scipy.optimize import minimize

# Alpha basin used to define a helical residue (degrees).
ALPHA_PHI = (-100.0, -30.0)
ALPHA_PSI = (-80.0, -5.0)

# i -> i+4 hydrogen bond criterion.
HBOND_DISTANCE = 0.25  # nm, H...O
HBOND_ANGLE = 120.0  # degrees, N-H...O


def atom_index(gro_file):
    """(residue number, atom name) -> atom index, from a GROMACS .gro file."""
    with open(gro_file) as handle:
        lines = handle.read().splitlines()
    return {(int(line[0:5]), line[10:15].strip()): k for k, line in enumerate(lines[2 : 2 + int(lines[1])])}  # noqa: E203


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1)))


def angle(a, b, c):
    u, v = a - b, c - b
    cosine = np.sum(u * v, -1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def helical_states(positions, index, residues):
    """Boolean (frames, residues) array: is the residue in the alpha basin?"""
    phi = np.stack(
        [
            dihedral(
                positions[:, index[(r - 1, "C")]],
                positions[:, index[(r, "N")]],
                positions[:, index[(r, "CA")]],
                positions[:, index[(r, "C")]],
            )
            for r in residues
        ],
        axis=1,
    )
    psi = np.stack(
        [
            dihedral(
                positions[:, index[(r, "N")]],
                positions[:, index[(r, "CA")]],
                positions[:, index[(r, "C")]],
                positions[:, index[(r + 1, "N")]],
            )
            for r in residues
        ],
        axis=1,
    )
    return (phi > ALPHA_PHI[0]) & (phi < ALPHA_PHI[1]) & (psi > ALPHA_PSI[0]) & (psi < ALPHA_PSI[1]), phi, psi


def segments(alpha):
    """Residues inside a run of 3 or more consecutive alpha residues."""
    padded = np.pad(alpha, ((0, 0), (1, 1)))
    left, right = padded[:, :-2], padded[:, 2:]
    return alpha & (
        (left & right)  # middle of a run
        | (right & np.pad(alpha, ((0, 0), (0, 2)))[:, 2:])  # start of a run
        | (left & np.pad(alpha, ((0, 0), (2, 0)))[:, :-2])  # end of a run
    )


def hydrogen_bonds(positions, index, first, last):
    """Boolean (frames, bonds) array of i -> i+4 backbone hydrogen bonds."""
    bonds = []
    for k in range(first, last + 1):
        if (k + 4, "H") not in index:
            continue
        o = index[(k, "O")]
        h, n = index[(k + 4, "H")], index[(k + 4, "N")]
        distance = np.linalg.norm(positions[:, h] - positions[:, o], axis=-1)
        bonds.append(
            (distance < HBOND_DISTANCE) & (angle(positions[:, n], positions[:, h], positions[:, o]) > HBOND_ANGLE)
        )
    return np.stack(bonds, axis=1)


def lifson_roig(alpha):
    """
    Maximum-likelihood fit of the Lifson-Roig w and v (see module docstring).

    Returns ``(w, v, gradient)``; the gradient should be ~0 at the optimum and
    is reported so that a failed fit cannot pass unnoticed.
    """
    n = alpha.shape[1]
    if n > 20:
        raise ValueError(f"exact enumeration is limited to 20 residues, got {n}")
    configurations = np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)
    padded = np.pad(configurations, ((0, 0), (1, 1)))
    n_w = (configurations & padded[:, :-2] & padded[:, 2:]).sum(axis=1).astype(float)
    n_v = configurations.sum(axis=1) - n_w

    observed = np.pad(alpha, ((0, 0), (1, 1)))
    w_counts = (alpha & observed[:, :-2] & observed[:, 2:]).sum(axis=1)
    mean_w, mean_v = w_counts.mean(), (alpha.sum(axis=1) - w_counts).mean()

    def negative_log_likelihood(x):
        exponent = n_w * x[0] + n_v * x[1]
        shift = exponent.max()
        weights = np.exp(exponent - shift)
        partition = weights.sum()
        probability = weights / partition
        value = -(mean_w * x[0] + mean_v * x[1] - (shift + np.log(partition)))
        gradient = -np.array([mean_w - (probability * n_w).sum(), mean_v - (probability * n_v).sum()])
        return value, gradient

    best = min(
        (
            minimize(negative_log_likelihood, x0, jac=True, method="L-BFGS-B")
            for x0 in ([0.0, -1.0], [0.5, -0.5], [-0.5, -2.0])
        ),
        key=lambda result: result.fun,
    )
    return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(np.abs(best.jac).max())


def block_error(series, blocks=5):
    """Standard error of a mean from block averages (handles correlation)."""
    means = np.array([block.mean() for block in np.array_split(series, blocks)])
    return float(means.std(ddof=1) / np.sqrt(blocks))


def analyse(xtc, gro, first, last, skip_ps=0.0):
    """Analyse one trajectory; ``first``/``last`` are the guest residue numbers."""
    index = atom_index(gro)
    coordinates, _, times, _ = read_xtc(xtc.encode() if isinstance(xtc, str) else xtc)
    positions = np.transpose(coordinates, (2, 0, 1))  # (frames, atoms, xyz), nm
    times = times - times[0] + (times[1] - times[0])
    keep = times > skip_ps
    positions = positions[keep]

    residues = list(range(first, last + 1))
    alpha, phi, psi = helical_states(positions, index, residues)
    inside = segments(alpha)
    bonds = hydrogen_bonds(positions, index, first - 1, last)

    counts = alpha.sum(axis=1)
    fraction = counts.mean() / len(residues)
    dispersion = counts.var() / (len(residues) * fraction * (1 - fraction)) if 0 < fraction < 1 else float("nan")
    w, v, gradient = lifson_roig(alpha)
    return {
        "frames": int(keep.sum()),
        "alpha": float(alpha.mean()),
        "segment": float(inside.mean()),
        "segment_err": block_error(inside.mean(axis=1)),
        "hbond": float(bonds.mean()),
        "hbond_err": block_error(bonds.mean(axis=1)),
        "w": w,
        "v": v,
        "sigma": v**2 / (1 + v) ** 4,
        "s": w / (1 + v),
        "dispersion": float(dispersion),
        "gradient": gradient,
        "phi": phi,
        "psi": psi,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories", nargs="+", help="XTC trajectories")
    parser.add_argument("--conf", required=True, help="matching .gro (for the atom names)")
    parser.add_argument("--first", type=int, required=True, help="first guest residue number")
    parser.add_argument("--last", type=int, required=True, help="last guest residue number")
    parser.add_argument("--skip", type=float, default=1000.0, help="discard this much time (ps)")
    args = parser.parse_args()

    print(f"{'trajectory':26s} {'helix%':>14} {'i->i+4%':>12} {'w':>7} {'v':>7} {'sigma':>8} {'s':>6} {'D':>6}")
    for path in args.trajectories:
        if not os.path.exists(path):
            print(f"{os.path.basename(path):26s} missing")
            continue
        r = analyse(path, args.conf, args.first, args.last, args.skip)
        if r["gradient"] > 1e-3:
            print(f"  warning: Lifson-Roig fit did not converge (gradient {r['gradient']:.1e})", file=sys.stderr)
        print(
            f"{os.path.basename(path):26s} {100 * r['segment']:7.1f}+-{100 * r['segment_err']:<5.1f}"
            f" {100 * r['hbond']:6.1f}+-{100 * r['hbond_err']:<4.1f}"
            f" {r['w']:7.3f} {r['v']:7.3f} {r['sigma']:8.4f} {r['s']:6.3f} {r['dispersion']:6.2f}"
        )


if __name__ == "__main__":
    main()
