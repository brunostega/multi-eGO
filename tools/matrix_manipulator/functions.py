import numpy as np 
import sys
import argparse
import os
import parmed as pmd
import warnings
import pandas as pd
import itertools
import gzip

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
        atoms_name.append(np.array([ a.type for a in topology.molecules.values().mapping[name][0].atoms]))
        atoms_name_per_res.append([ [a.type for a in r.atoms] for r in topology.molecules.values().mapping[name][0].residues])
        tot_num_atoms.append(np.sum(np.array([len(r.atoms) for r in topology.molecules.values().mapping[name][0].residues])))
    top_df["atoms_name"] = atoms_name
    top_df["residues"] = res
    top_df["N_atoms_per_res"] = atoms
    top_df["Ntot_atoms"] = tot_num_atoms
    top_df["atoms_name_per_res"] = atoms_name_per_res
    
    return topology,  top_df
