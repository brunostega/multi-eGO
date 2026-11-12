"""
Analysis of the replica-exchange runs of tools/remd/remd.py.

``helix`` mode  -- per temperature: helicity, i->i+4 hydrogen bonds, the
Lifson-Roig parameters and the dispersion D (see tools/helix/lifson_roig.py),
plus two convergence checks: the helicity of the first and second half of the
run, and the helicity of the frames descending from each starting structure
(at equilibrium neither split may matter).

``hairpin`` mode -- per temperature: backbone hydrogen bonds between residues
at least 5 apart (the sheet-forming ones), the fraction of the *native*
cross-strand hydrogen bonds (taken from a reference structure), the
probability of a folded hairpin (at least half of the native bonds) and of
any beta ladder (3 or more long-range bonds), the i->i+4 (helical) bonds and
the radius of gyration.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "helix"))

import lifson_roig as lr  # noqa: E402

HBOND_R, HBOND_ANGLE = 0.26, 120.0


def backbone_hbonds(positions, index):
    """{(donor residue, acceptor residue): boolean array over frames}."""
    residues = sorted({r for r, _ in index})
    bonds = {}
    for d in residues:
        if (d, "H") not in index or (d, "N") not in index:
            continue
        for a in residues:
            if (a, "O") not in index or abs(d - a) < 3:
                continue
            h, n, o = index[(d, "H")], index[(d, "N")], index[(a, "O")]
            r = np.linalg.norm(positions[:, h] - positions[:, o], axis=-1)
            bonds[(d, a)] = (r < HBOND_R) & (lr.angle(positions[:, n], positions[:, h], positions[:, o]) > HBOND_ANGLE)
    return bonds


def block_error(series, blocks=4):
    means = np.array([b.mean() for b in np.array_split(np.asarray(series, float), blocks)])
    return means.std(ddof=1) / np.sqrt(blocks)


def helix(data, index, first, last, skip):
    rows = []
    residues = list(range(first, last + 1))
    for k, temperature in enumerate(data["temperatures"]):
        x = data["positions"][k][skip:]
        origin = data["start"][data["walker"][k][skip:]]
        alpha, _, _ = lr.helical_states(x, index, residues)
        inside = lr.segments(alpha).mean(axis=1)
        bonds = lr.hydrogen_bonds(x, index, first - 1, last).mean(axis=1)
        counts = alpha.sum(axis=1)
        p = counts.mean() / len(residues)
        w, v, _ = lr.lifson_roig(alpha)
        conditional = lr.conditional_lifson_roig(alpha)
        half = len(inside) // 2
        rows.append(
            {
                "T": float(temperature),
                "helix": inside.mean(),
                "err": block_error(inside),
                "hbond": bonds.mean(),
                "w": w,
                "v": v,
                "sigma": v**2 / (1 + v) ** 4,
                "D": counts.var() / (len(residues) * p * (1 - p)) if 0 < p < 1 else np.nan,
                "first_half": inside[:half].mean(),
                "second_half": inside[half:].mean(),
                "from_start": [inside[origin == s].mean() if (origin == s).any() else np.nan for s in (0, 1)],
                "conditional": conditional,
            }
        )
    return rows


def hairpin(data, index, reference, skip):
    native = [
        pair
        for pair, formed in backbone_hbonds(reference[None], index).items()
        if formed[0] and abs(pair[0] - pair[1]) >= 5
    ]
    rows = []
    for k, temperature in enumerate(data["temperatures"]):
        x = data["positions"][k][skip:]
        bonds = backbone_hbonds(x, index)
        long_range = np.sum([f for (d, a), f in bonds.items() if abs(d - a) >= 5], axis=0)
        native_count = np.sum([bonds[p] for p in native], axis=0)
        helical = np.mean([f for (d, a), f in bonds.items() if d - a == 4], axis=0)
        rg = np.sqrt(((x - x.mean(axis=1, keepdims=True)) ** 2).sum(-1).mean(-1))
        folded = native_count >= len(native) / 2
        rows.append(
            {
                "T": float(temperature),
                "long_range": long_range.mean(),
                "native_fraction": native_count.mean() / len(native),
                "folded": folded.mean(),
                "folded_err": block_error(folded),
                "ladder": (long_range >= 3).mean(),
                "helical": helical.mean(),
                "rg": rg.mean(),
            }
        )
    return rows, native


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["helix", "hairpin"])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--conf", required=True, help=".gro with the atom names")
    parser.add_argument("--reference", default="", help="hairpin mode: native structure (.gro)")
    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--last", type=int, default=16)
    parser.add_argument("--skip", type=int, default=100, help="frames discarded at the start")
    args = parser.parse_args()
    index = lr.atom_index(args.conf)
    for path in args.runs:
        data = np.load(path)
        print(f"== {os.path.basename(path)}  acceptance {' '.join(f'{a:.2f}' for a in data['acceptance'])}")
        if args.mode == "helix":
            print(
                "    T   helix%        i->i+4%    w      v     sigma     D    1st/2nd half   from ext/helix"
                "  | equilibrium (conditional LR): w      v    sigma   helix%  P(helix)"
            )
            for r in helix(data, index, args.first, args.last, args.skip):
                print(
                    f"  {r['T']:4.0f}  {100 * r['helix']:5.1f}+-{100 * r['err']:<4.1f}  {100 * r['hbond']:5.1f}   "
                    f"{r['w']:6.3f} {r['v']:6.3f} {r['sigma']:7.4f} {r['D']:5.2f}   "
                    f"{100 * r['first_half']:5.1f}/{100 * r['second_half']:<5.1f}  "
                    f"{100 * r['from_start'][0]:5.1f}/{100 * r['from_start'][1]:<5.1f}"
                    f"  |                           {r['conditional']['w']:6.3f} {r['conditional']['v']:6.3f} "
                    f"{r['conditional']['v'] ** 2 / (1 + r['conditional']['v']) ** 4:7.4f}  "
                    f"{100 * r['conditional']['helix']:5.1f}   {r['conditional']['p_helix']:.2f}"
                )
        else:
            with open(args.reference) as handle:
                lines = handle.read().splitlines()
            atoms = lines[2 : 2 + int(lines[1])]  # noqa: E203
            reference = np.array([[float(row[20:28]), float(row[28:36]), float(row[36:44])] for row in atoms])
            rows, native = hairpin(data, index, reference, args.skip)
            print(f"  native cross-strand H-bonds (donor, acceptor): {native}")
            print("    T   long-range HB  native%   folded%      ladder%  i->i+4%   Rg(nm)")
            for r in rows:
                print(
                    f"  {r['T']:4.0f}   {r['long_range']:6.2f}      {100 * r['native_fraction']:5.1f}   "
                    f"{100 * r['folded']:5.1f}+-{100 * r['folded_err']:<4.1f}  {100 * r['ladder']:5.1f}    "
                    f"{100 * r['helical']:5.1f}    {r['rg']:.3f}"
                )


if __name__ == "__main__":
    main()
