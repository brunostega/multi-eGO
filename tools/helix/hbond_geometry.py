"""
Measure the geometry of backbone hydrogen bonds in reference structures.

Why
---
The hydrogen-bond term of ``multiego.omm.hbond`` originally used
``cos^2(theta)`` for both angles, i.e. it assumed that a hydrogen bond is
strongest when N-H...O *and* H...O=C are linear.  The first is right; the
second is not.  The carbonyl lone pairs point away from the C=O axis, so
H...O=C in real hydrogen bonds sits well below 180 degrees, and the preferred
value is not the same in alpha helices and in beta sheets.  Assuming 180 both
wastes most of the well depth and biases the alpha/beta balance.

This script measures the actual distributions, so that the angular factors
can be centred on them.

What it measures
----------------
For every backbone donor (amide H, with its N) and acceptor (carbonyl O, with
its C) within ``--cutoff``:

* ``r``      H...O distance (nm)
* ``nho``    N-H...O angle (degrees), 180 = linear at the donor
* ``hoc``    H...O=C angle (degrees), 180 = linear at the acceptor
* ``out``    angle between the H...O vector and the plane of the carbonyl
             (O, C and the C-alpha of the same residue), 0 = in plane

and classifies each contact by the separation of donor and acceptor:

* ``alpha``  same chain, acceptor = donor - 4 (the i -> i+4 helical bond)
* ``310``    same chain, acceptor = donor - 3
* ``beta``   different chains, or same chain with a separation above
             ``--beta-separation`` (default 5): the long-range contacts that
             make sheets
* ``other``  everything else

Only contacts that are hydrogen bonds by a geometric criterion
(``r < 0.26`` nm and ``nho > 110`` degrees) enter the statistics.

Input
-----
PDB or GROMACS .gro files.  In a .gro, a new chain is assumed wherever the
residue number decreases.
"""

import argparse
import os
from collections import defaultdict

import numpy as np

HBOND_MAX_R = 0.26  # nm
HBOND_MIN_NHO = 110.0  # degrees


def read_structure(path):
    """Return a list of (chain, resnum, resname, atom name, position in nm)."""
    atoms = []
    if path.endswith(".gro"):
        with open(path) as handle:
            lines = handle.read().splitlines()
        chain, previous = 0, None
        last = 2 + int(lines[1])
        for line in lines[2:last]:
            resnum = int(line[0:5])
            if previous is not None and resnum < previous:
                chain += 1
            previous = resnum
            atoms.append(
                (
                    chain,
                    resnum,
                    line[5:10].strip(),
                    line[10:15].strip(),
                    np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])]),
                )
            )
    else:
        with open(path) as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    atoms.append(
                        (
                            line[21],
                            int(line[22:26]),
                            line[17:20].strip(),
                            line[12:16].strip(),
                            np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) / 10.0,
                        )
                    )
    return atoms


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0)))


def out_of_plane(h, o, c, ca):
    """Angle (degrees) between the H...O vector and the carbonyl plane."""
    normal = np.cross(c - o, ca - c)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.nan
    direction = h - o
    sine = abs(normal @ direction) / norm / np.linalg.norm(direction)
    return np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0)))


def contacts(atoms, cutoff, beta_separation):
    """Yield (class, r, nho, hoc, out) for every hydrogen bond in a structure."""
    index = {(chain, resnum, name): position for chain, resnum, _, name, position in atoms}
    donors = [(chain, resnum) for (chain, resnum, name) in index if name == "H" and (chain, resnum, "N") in index]
    acceptors = [(chain, resnum) for (chain, resnum, name) in index if name == "O" and (chain, resnum, "C") in index]

    for chain_d, residue_d in donors:
        h, n = index[(chain_d, residue_d, "H")], index[(chain_d, residue_d, "N")]
        for chain_a, residue_a in acceptors:
            o, c = index[(chain_a, residue_a, "O")], index[(chain_a, residue_a, "C")]
            r = np.linalg.norm(h - o)
            if r > cutoff:
                continue
            nho = angle(n, h, o)
            if r > HBOND_MAX_R or nho < HBOND_MIN_NHO:
                continue
            if chain_d == chain_a:
                separation = residue_d - residue_a
                if separation == 4:
                    kind = "alpha"
                elif separation == 3:
                    kind = "310"
                elif abs(separation) >= beta_separation:
                    kind = "beta"
                else:
                    kind = "other"
            else:
                kind = "beta"
            ca = index.get((chain_a, residue_a, "CA"))
            yield kind, r, nho, angle(h, o, c), (out_of_plane(h, o, c, ca) if ca is not None else np.nan)


def summarise(values, label):
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) == 0:
        return f"{label}: no data"
    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    return (
        f"{label}: n={len(values):4d}  mean {values.mean():6.1f}  sd {values.std():5.1f}  "
        f"median {percentiles[2]:6.1f}  [10-90%] {percentiles[0]:6.1f} .. {percentiles[4]:6.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structures", nargs="+")
    parser.add_argument("--cutoff", type=float, default=0.35, help="H...O search cut-off (nm)")
    parser.add_argument("--beta-separation", type=int, default=5)
    parser.add_argument("--save", default="", help="optional .npz with every measured contact")
    args = parser.parse_args()

    data = defaultdict(lambda: defaultdict(list))
    for path in args.structures:
        if not os.path.exists(path):
            print(f"  (missing: {path})")
            continue
        found = defaultdict(int)
        for kind, r, nho, hoc, out in contacts(read_structure(path), args.cutoff, args.beta_separation):
            found[kind] += 1
            for name, value in (("r", r), ("nho", nho), ("hoc", hoc), ("out", out)):
                data[kind][name].append(value)
        print(f"{os.path.basename(path):24s} " + "  ".join(f"{k} {v}" for k, v in sorted(found.items())))

    print()
    for kind in ("alpha", "310", "beta", "other"):
        if kind not in data:
            continue
        print(f"--- {kind} ---")
        print("  " + summarise(np.array(data[kind]["r"]) * 10, "r (A)     "))
        for name, label in (("nho", "N-H...O   "), ("hoc", "H...O=C   "), ("out", "out-of-pln")):
            print("  " + summarise(data[kind][name], label))
    if args.save:
        np.savez(args.save, **{f"{k}_{n}": np.array(v) for k, d in data.items() for n, v in d.items()})
        print(f"\nwritten {args.save}")


if __name__ == "__main__":
    main()
