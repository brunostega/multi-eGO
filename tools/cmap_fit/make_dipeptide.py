"""
Build the blocked amino acid Ace-Xaa-NMe used to fit a backbone CMAP.

    GMXLIB=/path/to/multi-eGO python tools/cmap_fit/make_dipeptide.py XXX OUTDIR

Writes OUTDIR/XXX/{XXX.pdb, XXX.gro, XXX_box.gro (5 nm box), topol.top}
using PeptideBuilder for the coordinates and ``gmx pdb2gmx`` with the
multi-ego-basic force field for the topology (no terminal patches: the Ace
and NMe caps are residues of the force field).  The topology can then be fed
to ``multiego.py --egos mg`` and the result to ``fit_cmap.py``.
"""

import os
import subprocess
import sys

import PeptideBuilder
from Bio.PDB import PDBIO

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}  # fmt: skip


def build(resname, outdir, gmx="gmx"):
    os.makedirs(outdir, exist_ok=True)
    structure = PeptideBuilder.make_extended_structure("G" + ONE_LETTER[resname] + "G")
    first, _, last = list(structure.get_residues())
    for atom in list(first):  # N-cap: keep the acetyl group (CA, C, O)
        if atom.get_id() not in ("CA", "C", "O"):
            first.detach_child(atom.get_id())
    first.resname = "ACE"
    for atom in list(last):  # C-cap: N-methyl amide (N, CA; H added by pdb2gmx)
        if atom.get_id() not in ("N", "CA"):
            last.detach_child(atom.get_id())
    last.resname = "NME"
    io = PDBIO()
    io.set_structure(structure)
    pdb = os.path.join(outdir, f"{resname}.pdb")
    io.save(pdb)

    # "2\n2\n": no terminal patch at either end (the caps are residues)
    subprocess.run(
        [gmx, "pdb2gmx", "-f", pdb, "-o", f"{outdir}/{resname}.gro", "-p", f"{outdir}/topol.top",
         "-i", f"{outdir}/posre.itp", "-ff", "multi-ego-basic", "-water", "none", "-ignh", "-ter"],
        input="2\n2\n", text=True, check=True, capture_output=True,
    )  # fmt: skip
    subprocess.run(
        [gmx, "editconf", "-f", f"{outdir}/{resname}.gro", "-o", f"{outdir}/{resname}_box.gro", "-c", "-box", "5"],
        check=True, capture_output=True,
    )  # fmt: skip


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], os.environ.get("GMX", "gmx"))
