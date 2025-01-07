import numpy as np 
import sys
import argparse
import os
import parmed as pmd
import warnings
import pandas as pd
import itertools
import gzip

COLUMNS = ["mi", "ai", "mj", "aj", "c12dist", "p", "cutoff"]
exclude = ["SOL", "CL","NA"]

def read_topologies(top):
    '''
    Reads the input topologies using parmed. Ignores warnings to prevent printing
    of GromacsWarnings regarding 1-4 interactions commonly seen when using
    parmed in combination with multi-eGO topologies.

    Parameters
    ----------
    mego_top : str
        Path to the multi-eGO topology obtained from gmx pdb2gmx with multi-ego-basic force fields
    target_top : str
        Path to the toplogy of the system on which the analysis is to be performed
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topology = pmd.load_file(top)

    #Return topology and a dataframe with:
    #molecule name, number of molecules?, residue list, atom_list_per_residue 
    top_df = pd.DataFrame()
    n_mol=len(list(topology.molecules.keys()))
    mol_names=list(topology.molecules.keys())
    top_df["name"] = [m for m in mol_names if m not in exclude]
    mol_list=np.arange(1,n_mol+1,1)
    res = []
    atoms = []
    atoms_name_per_res = []
    atoms_name = []
    tot_num_atoms = []
    for name in mol_names:
        if name in exclude: continue
        res.append([r.name for r in topology.molecules.values().mapping[name][0].residues])
        atoms.append([len(r.atoms) for r in topology.molecules.values().mapping[name][0].residues])
        atoms_name.append(np.array([ a.name for a in topology.molecules.values().mapping[name][0].atoms]))
        atoms_name_per_res.append([ [a.type for a in r.atoms] for r in topology.molecules.values().mapping[name][0].residues])
        tot_num_atoms.append(np.sum(np.array([len(r.atoms) for r in topology.molecules.values().mapping[name][0].residues])))
    top_df["atoms_name"] = atoms_name
    top_df["residues"] = res
    top_df["N_atoms_per_res"] = atoms
    top_df["Ntot_atoms"] = tot_num_atoms
    top_df["atoms_name_per_res"] = atoms_name_per_res
    
    return topology,  top_df

def atom_element(names):
    return np.array([n for n in names])

def check_inputs(top_df_input, top_df_output, args):
    mat = pd.read_csv(args.input_mat,names=["mi", "ai", "mj", "aj", "c12dist", "p", "cutoff"],  sep="\s+")

    MOL_I = int(os.path.basename(args.input_mat).split(".ndx")[0].split("_")[1]) -1
    MOL_J = int(os.path.basename(args.input_mat).split(".ndx")[0].split("_")[2]) -1

    N_mol_input = len(top_df_input["name"])
    N_mol_output = len(top_df_output["name"])
    # Checks that all input are consistent (correct matrix associated to the system and consitent input and ouput topologies)
    if N_mol_input!=N_mol_output:
        raise ValueError(f'Number of molecules in input and output topology are different: input {N_mol_input}, output {N_mol_output}')
    for i,mol in enumerate(top_df_input["name"]):
        if top_df_input["name"][i] != top_df_output["name"][i]:
            raise ValueError(f"Molecules in input and output topology do not correspond: input {top_df_input['name'][i]}, output {top_df_output['name'][i]}")
        H_indices_in = np.strings.startswith(top_df_input["atoms_name"].to_numpy()[i],"H")
        H_indices_out = np.strings.startswith(top_df_output["atoms_name"].to_numpy()[i],"H")
        if np.any(atom_element(top_df_input["atoms_name"].to_numpy()[i][~H_indices_in])!=atom_element(top_df_output["atoms_name"].to_numpy()[i][~H_indices_out])):
            raise ValueError(f"Found difference in atoms of molecule {mol}")
    
    if(N_mol_input < np.maximum(MOL_I+1, MOL_J+1)):
        raise ValueError(f'Matrix indedes {MOL_I} {MOL_J} are not compatible with the given topology: Not enough molecules in topology')
    if(len(top_df_input["atoms_name"].to_numpy()[MOL_I])*len(top_df_input["atoms_name"].to_numpy()[MOL_J]) != len(mat["ai"])):
        raise ValueError(f'Number of elements in input matrix is not compatible with the number of atoms in topology: matrix : {len(mat["ai"])} , topology mol_{MOL_I+1} {len(top_df_input["atoms_name"].to_numpy()[MOL_I])} - mol_{MOL_J +1 } {len(top_df_input["atoms_name"].to_numpy()[MOL_J])}' )

    return mat, MOL_I, MOL_J


def write_mat(mat_out, output_file):
    #format
    mat_out["mi"] = mat_out["mi"].map("{:}".format)
    mat_out["mj"] = mat_out["mj"].map("{:}".format)
    mat_out["ai"] = mat_out["ai"].map("{:}".format)
    mat_out["aj"] = mat_out["aj"].map("{:}".format)
    mat_out["c12dist"] = mat_out["c12dist"].map("{:,.6f}".format)
    mat_out["p"] = mat_out["p"].map("{:,.6e}".format)
    mat_out["cutoff"] = mat_out["cutoff"].map("{:,.6f}".format)

    out_content = mat_out.to_string(index=False, header=False, columns=COLUMNS)
    out_content = out_content.replace("\n", "<")
    out_content = " ".join(out_content.split())
    out_content = out_content.replace("<", "\n")
    out_content += "\n"
    with gzip.open(output_file, "wt") as f:
        f.write(out_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
This tools modifies an input matrix to change the protonation state accordingly to the input and output topology
        """)
        
    parser.add_argument('--input_mat'   , type=str,  required=True , default=None, help='List of paths associated to the intramat for each domain')
    parser.add_argument('--input_top'   , type=str,  required=True , default=None, help='Path for the matrix to be added on the domain ranges')
    parser.add_argument('--output_top'  , type=str,  required=True , default=None, help='List of residue ranges of the domains')
    parser.add_argument('--out_name'    , type=str,  required=False, default=""  , help='Target topology (the topology of the final full lenght structure)')
    parser.add_argument('--out'         , type=str,  required=False, default="." , help='List of paths of each domain associated to the ranges')
    parser.add_argument('--H_name'         , type=str,  required=False, default="H" , help='List of paths of each domain associated to the ranges')

    args = parser.parse_args()
  


top, top_df = read_topologies(args.input_top)
out_top, out_top_df = read_topologies(args.output_top)

mat, MOL_I, MOL_J = check_inputs(top_df, out_top_df, args)

mat["ai_name"] = np.array(list(itertools.product(top_df["atoms_name"][MOL_I],top_df["atoms_name"][MOL_J]))).T[0]
mat["aj_name"] = np.array(list(itertools.product(top_df["atoms_name"][MOL_I],top_df["atoms_name"][MOL_J]))).T[1]
mat_noH = mat.loc[(~(mat["ai_name"].str.startswith(args.H_name)) | ((mat["ai_name"] == args.H_name))) & ((~(mat["aj_name"].str.startswith(args.H_name))) | ((mat["aj_name"] == args.H_name))) ]


mat_out = pd.DataFrame()
N_atom_out_i = len(out_top_df["atoms_name"][MOL_I])
N_atom_out_j = len(out_top_df["atoms_name"][MOL_J])

# Create output matrix 
mat_out["ai_name"] = np.array(list(itertools.product(out_top_df["atoms_name"][MOL_I],out_top_df["atoms_name"][MOL_J]))).T[0]
mat_out["aj_name"] = np.array(list(itertools.product(out_top_df["atoms_name"][MOL_I],out_top_df["atoms_name"][MOL_J]))).T[1]

mat_out["mi"]  = np.ones(len(out_top_df["atoms_name"][MOL_I])*len(out_top_df["atoms_name"][MOL_J]),dtype = int)*(MOL_I + 1)
mat_out["ai"] = np.array(list(itertools.product(np.arange(1,N_atom_out_i+1),np.arange(1,N_atom_out_j+1)))).T[0]
mat_out["mj"]  = np.ones(len(out_top_df["atoms_name"][MOL_I])*len(out_top_df["atoms_name"][MOL_J]),dtype = int)*(MOL_J + 1)
mat_out["aj"] = np.array(list(itertools.product(np.arange(1,N_atom_out_i+1),np.arange(1,N_atom_out_j+1)))).T[1]

# First define 0 entries and overwrite where atom != H
where_H = (~(mat_out["ai_name"].str.startswith(args.H_name)) | ((mat_out["ai_name"] == args.H_name))) & ((~(mat_out["aj_name"].str.startswith(args.H_name))) | ((mat_out["aj_name"] == args.H_name)))
mat_out["c12dist"] = np.zeros(N_atom_out_i*N_atom_out_j)
mat_out.loc[where_H, "c12dist"] = mat_noH["c12dist"].to_numpy()

mat_out["p"] = np.zeros(N_atom_out_i*N_atom_out_j)
mat_out.loc[where_H, "p"]       = mat_noH["p"].to_numpy()

mat_out["cutoff"] = np.zeros(N_atom_out_i*N_atom_out_j)
mat_out.loc[where_H, "cutoff"]  = mat_noH["cutoff"].to_numpy()


write_mat(mat_out, f"{args.out}/{args.out_name}{os.path.basename(args.input_mat)}.gz")