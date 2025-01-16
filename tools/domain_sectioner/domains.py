import numpy as np
import argparse
import os

# import parmed as pmd
import warnings

# import pandas as pd
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)
from src.multiego.util import mat_modif_functions as functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO!")
    parser.add_argument("--intra", type=str, required=True, help="matrix to be masked with the domain ranges")
    parser.add_argument("--top", type=str, required=True, help="topology file associated to the matrix")
    parser.add_argument(
        "--dom_res",
        nargs="+",
        type=str,
        default=[],
        help="Residue ranges to be masked which will be learned in multiego. Example: 1-10 20-30 40-50",
        required=True,
    )
    parser.add_argument("--out", type=str, default=".", help="path for ouput")
    parser.add_argument(
        "--invert", action="store_true", default=False, help="Invert domain mask: Learn all but in the domain ranges"
    )

    args = parser.parse_args()

    if args.out:
        if not os.path.isdir(args.out):
            print(f"{args.out} does not exists. Insert an existing directory")
            exit()
        else:
            if args.out[-1] != "/":
                args.out = args.out + "/"

    # read topology
    topology_mego, top_df = functions.read_topologies(args.top)

    # check if there is only one molecule. This code should modify only intramat of one molecule
    if len(top_df) > 1:
        raise ValueError("Only one molecule specie is allowed, topology contains more than one molecule")

    # define atom_num and res_num of the molecule
    n_atoms = top_df.N_tot_atoms[0]
    n_res = len(top_df.residues[0])

    ranges = functions.dom_range(args.dom_res)
    print(f"\n Total number of residues {n_res} and total number of atoms {n_atoms} \n")

    # read intramat and check consistency
    intramat = args.intra
    intra_md = np.loadtxt(intramat, unpack=True)
    dim = int(np.sqrt(len(intra_md[0])))
    if dim != n_atoms:
        raise ValueError(f"ERROR: number of atoms in intramat ({dim}) does not correspond to that of topology ({n_atoms})")

    # define domain mask
    domain_mask_linear = np.full(dim**2, False)
    for r in ranges:
        start = functions.find_atom_start(topology_mego, r[0])
        end = functions.find_atom_end(topology_mego, r[1])
        if start >= end:
            appo_end = end
            end = start
            start = appo_end
            print(f"  Domain range: {r[0]}-{r[1]} INVERTED")
            print(f"     Atom index range start-end: {start+1} - {end+1}")
            print(f"     Number of atoms in domain range:  {end+1 - (start)}")
            print(f"     Atom and Residue of start-end {topology_mego.atoms[start]} - {topology_mego.atoms[end]}")
            print("\n")
            map_appo = np.invert(np.array([True if x >= start and x <= end else False for x in range(dim)]))
        else:
            print(f"  Domain range: {r[0]}-{r[1]}")
            print(f"     Atom index range start-end: {start+1} - {end+1}")
            print(f"     Number of atoms in domain range:  {end+1 - (start)}")
            print(f"     Atom and Residue of start-end {topology_mego.atoms[start]} - {topology_mego.atoms[end]}")
            print("\n")
            map_appo = np.array([True if x >= start and x <= end else False for x in range(dim)])
        domain_mask_linear = np.logical_or(domain_mask_linear, (map_appo * map_appo[:, np.newaxis]).reshape(dim**2))

    if args.invert:
        domain_mask_linear = np.logical_not(domain_mask_linear)

    # add an eigth column with the domain_mask
    if intra_md.shape[0] == 7:
        intra_md = np.concatenate((intra_md, domain_mask_linear[np.newaxis, :]), axis=0)
    else:
        intra_md[7] = domain_mask_linear

    if "/" in intramat:
        intramat = intramat.split("/")[-1]

    if args.invert:
        out_name = f'{args.out}/inverted_split_{"-".join(np.array(args.dom_res, dtype=str))}_{intramat}'
    else:
        out_name = f'{args.out}/split_{"-".join(np.array(args.dom_res, dtype=str))}_{intramat}'
    np.savetxt(
        out_name,
        intra_md.T,
        delimiter=" ",
        fmt=["%i", "%i", "%i", "%i", "%2.6f", "%.6e", "%2.6f", "%1i"],
    )
    print("Finished creating the masked matrix :)")
