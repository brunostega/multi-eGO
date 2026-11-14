"""
Derive the CMAPs of residue variants that have no experimental target.

    python tools/cmap_fit/derive_variants.py cmaps

* D-amino acids (DALA, DARG, ...): the mirror image of an L residue.  Under
  reflection both backbone dihedrals change sign, so the map is exactly
  U_D(phi, psi) = U_L(-phi, -psi).  On the grid of multiego.cmap (nodes at
  -180 + i * 360/S) the node of -angle(i) is (S - i) mod S.
* CYS2 (cysteine in a disulfide bridge): the CYS map.  This is an
  approximation -- the bridge constrains the backbone as well -- adopted
  because no blocked-amino-acid data exist for a bridged cysteine.

Glycine is achiral, so DGLY is not needed (and is not a residue of the force
field).  Existing files are overwritten.
"""

import os
import sys

import numpy as np


def mirror(grid):
    """Map of the enantiomer: U'(phi, psi) = U(-phi, -psi) on the node grid."""
    return np.roll(np.flip(grid, axis=(0, 1)), 1, axis=(0, 1))


def main(directory):
    written = []
    for filename in sorted(os.listdir(directory)):
        name, extension = os.path.splitext(filename)
        if extension != ".npz" or name.startswith("D") and name[1:].isupper() and len(name) == 4:
            continue  # skip non-maps and derived D-maps
        with np.load(os.path.join(directory, filename)) as data:
            content = dict(data)
        if name == "GLY":
            continue
        mirrored = {k: mirror(v) if np.ndim(v) == 2 else v for k, v in content.items()}
        np.savez_compressed(os.path.join(directory, f"D{name}.npz"), **mirrored)
        written.append(f"D{name}")
        if name == "CYS":
            np.savez_compressed(os.path.join(directory, "CYS2.npz"), **content)
            mirrored_cys = {k: mirror(v) if np.ndim(v) == 2 else v for k, v in content.items()}
            np.savez_compressed(os.path.join(directory, "DCYS2.npz"), **mirrored_cys)
            written += ["CYS2", "DCYS2"]
    print("written:", " ".join(written))


if __name__ == "__main__":
    main(sys.argv[1])
