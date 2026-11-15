"""
Backbone diagnostics for a peptide trajectory.

    python tools/helix/diagnose_backbone.py run.xtc --conf box.gro [--skip 1000]

Written to explain trajectories where hydrogen-bond counts and the
Ramachandran plot disagree (e.g. many i->i+4 hydrogen bonds but little
alpha-helical phi/psi).  It reports, frame-averaged:

1. whole molecules: the longest N-CA, CA-C and C-N bond in any frame.  If a
   bond is much longer than 0.15 nm the molecule is split by the periodic
   boundary in the file, and every dihedral across the split is garbage;
2. chirality at every C-alpha, from the sign of the improper N-CA-C-CB.  The
   L/D assignment is calibrated on an ideal L residue built in the same
   way, so the sign convention cannot be wrong;
3. peptide bonds: the fraction of cis omega (|CA-C-N-CA| < 90 deg);
4. Ramachandran basins: alphaR, alphaL, PPII, beta and the rest, for all
   residues and for the residues whose N-H forms an i->i+4 hydrogen bond;
5. the residue separation of every backbone hydrogen bond formed.

A right-handed alpha helix of L residues shows: whole molecules, all L,
no cis bonds, and i->i+4-bonded residues in alphaR.
"""

import argparse
from collections import Counter

import numpy as np
from openmm.app.internal.xtc_utils import read_xtc

import lifson_roig as lr

L_SIGN = None  # sign of the N-CA-C-CB improper for an L residue, set below


def _calibrate_l_sign():
    """Improper N-CA-C-CB of an ideal L-alanine (standard coordinates, nm)."""
    n = np.array([[-0.0525, 0.1363, 0.0]])
    ca = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[0.1526, 0.0, 0.0]])
    cb = np.array([[-0.0529, -0.0774, -0.1205]])  # L configuration
    return np.sign(lr.dihedral(n, ca, c, cb)[0])


def basins(phi, psi):
    a_r = (phi < 0) & (phi > -160) & (psi > -120) & (psi < 30)
    a_l = (phi > 0) & (psi > -60) & (psi < 100)
    extended = (phi < 0) & ((psi >= 30) | (psi <= -120))
    ppii = extended & (phi > -100)
    beta = extended & (phi <= -100)
    total = phi.size
    out = {"alphaR": a_r.sum(), "alphaL": a_l.sum(), "PPII": ppii.sum(), "beta": beta.sum()}
    out["other"] = total - sum(out.values())
    return {k: 100.0 * v / max(total, 1) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xtc")
    parser.add_argument("--conf", required=True, help=".gro with the same atoms")
    parser.add_argument("--skip", type=float, default=0.0, help="ps discarded at the start")
    args = parser.parse_args()

    index = lr.atom_index(args.conf)
    coordinates, _, times, _ = read_xtc(args.xtc.encode())
    x = np.transpose(coordinates, (2, 0, 1))
    times = times - times[0]
    x = x[times >= args.skip]
    residues = sorted({r for r, name in index if name == "CA" and (r, "N") in index and (r, "C") in index})
    print(f"{len(x)} frames, residues with N, CA, C: {residues[0]}..{residues[-1]}")

    # 1. whole molecules
    longest = 0.0
    for r in residues:
        pairs = [((r, "N"), (r, "CA")), ((r, "CA"), (r, "C"))]
        if (r + 1, "N") in index:
            pairs.append(((r, "C"), (r + 1, "N")))
        for a, b in pairs:
            longest = max(longest, np.linalg.norm(x[:, index[a]] - x[:, index[b]], axis=-1).max())
    verdict = "OK" if longest < 0.2 else "BROKEN: molecule split across the periodic boundary in the file"
    print(f"1. longest backbone bond in any frame: {longest:.3f} nm  -> {verdict}")

    # 2. chirality
    l_sign = _calibrate_l_sign()
    chir = []
    for r in residues:
        if (r, "CB") in index:
            s = np.sign(
                lr.dihedral(
                    x[:, index[(r, "N")]], x[:, index[(r, "CA")]], x[:, index[(r, "C")]], x[:, index[(r, "CB")]]
                )
            )
            chir.append(np.mean(s == l_sign))
    if chir:
        print(
            f"2. fraction of L chirality over residues with CB: {np.mean(chir):.3f} (min per residue {np.min(chir):.3f})"
        )

    # 3. cis peptide bonds
    cis = []
    for r in residues:
        if (r + 1, "N") in index and (r + 1, "CA") in index:
            w = lr.dihedral(
                x[:, index[(r, "CA")]], x[:, index[(r, "C")]], x[:, index[(r + 1, "N")]], x[:, index[(r + 1, "CA")]]
            )
            cis.append(np.mean(np.abs(w) < 90))
    if cis:
        print(f"3. fraction of cis peptide bonds: {np.mean(cis):.3f}")

    # 4. basins, all residues and residues with an i->i+4 bond on their N-H
    inner = [r for r in residues if (r - 1, "C") in index and (r + 1, "N") in index]
    phi = {
        r: lr.dihedral(x[:, index[(r - 1, "C")]], x[:, index[(r, "N")]], x[:, index[(r, "CA")]], x[:, index[(r, "C")]])
        for r in inner
    }
    psi = {
        r: lr.dihedral(x[:, index[(r, "N")]], x[:, index[(r, "CA")]], x[:, index[(r, "C")]], x[:, index[(r + 1, "N")]])
        for r in inner
    }
    all_b = basins(np.concatenate(list(phi.values())), np.concatenate(list(psi.values())))
    print("4. Ramachandran basins, all residues (%):  " + "  ".join(f"{k} {v:5.1f}" for k, v in all_b.items()))
    bonded_phi, bonded_psi = [], []
    separations = Counter()
    for d in residues:
        if (d, "H") not in index:
            continue
        h, n = index[(d, "H")], index[(d, "N")]
        for a in residues:
            if (a, "O") not in index or abs(d - a) < 3:
                continue
            o = index[(a, "O")]
            formed = (np.linalg.norm(x[:, h] - x[:, o], axis=-1) < 0.25) & (lr.angle(x[:, n], x[:, h], x[:, o]) > 120)
            separations[d - a] += int(formed.sum())
            if d - a == 4:
                # residues inside the i->i+4 turn: a+1 .. d-1
                for r in range(a + 1, d):
                    if r in phi:
                        bonded_phi.append(phi[r][formed])
                        bonded_psi.append(psi[r][formed])
    if bonded_phi:
        b = basins(np.concatenate(bonded_phi), np.concatenate(bonded_psi))
        bp, bs = np.concatenate(bonded_phi), np.concatenate(bonded_psi)
        print(
            "   residues inside formed i->i+4 bonds (%): "
            + "  ".join(f"{k} {v:5.1f}" for k, v in b.items())
            + f"   median phi {np.median(bp):.0f} psi {np.median(bs):.0f}"
        )
    total = sum(separations.values())
    print(
        "5. hydrogen bonds by separation donor-acceptor (% of all formed): "
        + "  ".join(f"{s:+d}: {100 * c / total:.1f}" for s, c in sorted(separations.items()) if c > 0.005 * total)
    )


if __name__ == "__main__":
    main()
