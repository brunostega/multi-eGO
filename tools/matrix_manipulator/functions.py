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
    atoms_type_per_res = []
    atoms_name_per_res = []
    atoms_name = []
    tot_num_atoms = []
    for name in mol_names:
        if name in exclude: continue
        res.append([r.name for r in topology.molecules.values().mapping[name][0].residues])
        atoms.append([len(r.atoms) for r in topology.molecules.values().mapping[name][0].residues])
        atoms_name.append(np.array([ a.name for a in topology.molecules.values().mapping[name][0].atoms]))
        atoms_type_per_res.append([ [a.type for a in r.atoms] for r in topology.molecules.values().mapping[name][0].residues])
        atoms_name_per_res.append([ [a.type[0] for a in r.atoms] for r in topology.molecules.values().mapping[name][0].residues])
        tot_num_atoms.append(np.sum(np.array([len(r.atoms) for r in topology.molecules.values().mapping[name][0].residues])))
    top_df["atoms_name"] = atoms_name
    top_df["residues"] = res
    top_df["N_atoms_per_res"] = atoms
    top_df["N_tot_atoms"] = tot_num_atoms
    top_df["atoms_name_per_res"] = atoms_name_per_res
    top_df["atoms_type_per_res"] = atoms_type_per_res

    return topology,  top_df


def mask_terminals(top_df, i,ranges,intra, n_atoms_start, column):
    # N terminal Hydrogens remove condition
    if ranges[i][0] == 1:
        print("     column res=1 --> not removing N terminal hydrogens")
        mmask_i_base = np.zeros(len(intra[column]))
    else:
        mmask_i_base = np.logical_or(intra[column]==2, intra[column]==3)

    # C terminal Oxygen remove condition
    if ranges[i][1] < len(top_df["residues"][0]): 
        mmask_i = np.where( np.logical_or(mmask_i_base, intra[column]==n_atoms_start))
    else:
        print("     column res=last --> not removing C terminal Oxygen")
        mmask_i = np.where( mmask_i_base )
    return mmask_i


def find_atom_start(top, res_num):
    '''
    Finds the starting atom associated to the residue 
    '''
    atom_idx = 0

    for i in range(res_num-1):
        atom_idx += len(top.residues[i].atoms)

    return atom_idx

def find_atom_end(top, res_num):
    '''
    Finds the ending atom associated to the residue 
    '''
    atom_idx = 0
    n_atoms = len(top.atoms)
    n_res   = len(top.residues)
    if(res_num==n_res):
        return n_atoms
    else:
        for i in range(res_num):
            atom_idx += len(top.residues[i].atoms)

        return atom_idx

def dom_range(ranges_str):
    '''
    Reads the ranges given in input as a string and puts them in output 
    as a list of tuples
    '''
    doms = []
    print("\nReading domain ranges in which inserting intramats")
    for i in range(len(ranges_str)):
       print(ranges_str[i])
       doms.append( (int(ranges_str[i].split("-")[0]), int(ranges_str[i].split("-")[1])) )

    
    for i in range(len(doms) - 1):
        if doms[i][0] >= doms[i + 1][0]:
            print("First numbers are not in order")
            exit()

    # Check if the second numbers are ordered
    for i in range(len(doms) - 1):
        if doms[i][1] >= doms[i + 1][1]:
            print("Second numbers are not in order")
            exit()

    # Check if numbers within each tuple are ordered
    for t in doms:
        if t[0] >= t[1]:
            print("Numbers within tuple are not in order")
            exit()

    return doms


def domain_mask(topology_ref, domain_ranges, dim):
    
    #define domain mask
    domain_mask = np.full((dim, dim), False)
    
    #add partial masks corresponding to the domain ranges
    map_appo = np.array([ True if x >= find_atom_start(topology_ref, domain_ranges[0]) and x < find_atom_end(topology_ref, domain_ranges[1]) else False for x in range(dim)])
    map_appo = map_appo * map_appo[:,np.newaxis]
    
    #join the partial mask to the full  lenght matrix
    domain_mask = np.logical_or(domain_mask, map_appo)
    domain_mask_linear = domain_mask.reshape(dim**2)
    return domain_mask_linear

def domain_mask_inter(topology_ref, domain_ranges, dim, dim_inter):

    #define domain mask
    domain_mask = np.full((dim, dim_inter), False)

    #add partial masks corresponding to the domain ranges
    map_appo = np.array([ True if x >= find_atom_start(topology_ref, domain_ranges[0]) and x < find_atom_end(topology_ref, domain_ranges[1]) else False for x in range(dim)])
    map_appo_2 = np.array([ True for x in range(dim_inter)])
    map_appo = np.outer(map_appo, map_appo_2)

    #join the partial mask to the full  lenght matrix
    domain_mask = np.logical_or(domain_mask, map_appo)
    domain_mask_linear = domain_mask.reshape(dim*dim_inter)

    return domain_mask_linear

def check_domain_topologies(domains_tops, top_df, args, ranges):
    '''
    Check that the domain topologies are compatible with the full system topology:
    - checks that the residues names are the same
    - checks that the number of atoms per residue is the same
    - checks that the atom names are the same
    '''
    print(":::Checking domain topology:::")
    count_err_res     = 0
    count_err_num_at  = 0
    count_err_name_at = 0
    topol_domains = []
    CHECK_ATOM_ELEMENT = args.check_element#;False #this will mostly always give error because atom names will be different. Use it only to check that atom types are the same

    for i,d in enumerate(domains_tops):
        #target topology stuff to be checked
        res_ref_temp = top_df["residues"][0][int(ranges[i][0]-1):int(ranges[i][1])]
        atoms_ref_temp = top_df["N_atoms_per_res"][0][int(ranges[i][0]-1):int(ranges[i][1])]
        atoms_name_target = np.concatenate(top_df["atoms_name_per_res"][0][int(ranges[i][0]-1):int(ranges[i][1])])

        #read domain top 
        topology_dom, top_df_dom= read_topologies(d)

        temp_dom_atoms = np.array(top_df_dom["N_atoms_per_res"][0])
        #remove the excess terminal oxigen
        if ranges[i][1] < len(top_df["residues"][0]): temp_dom_atoms[-1]-=1     

        #remove the terminal excess hydroges to compare full system top and single domain top
        if ranges[i][0] > 1: temp_dom_atoms[0]-=2
        print(f"-Domain {i+1}: between ranges {ranges[i]} of full structure")

        #check residue names
        if np.any(np.array(res_ref_temp)!=np.array(top_df_dom["residues"][0])):
            where = np.argwhere(np.array(res_ref_temp)!=np.array(top_df_dom["residues"][0]))
            print("difference in Residue names!!")
            print(np.array(res_ref_temp))
            print(np.array(top_df_dom["residues"][0]))
            print(f"""Are different here:
    Target {np.array(res_ref_temp)[where]} 
    Domain {np.array(top_df_dom['residues'][0])[where]}
    """)
            count_err_res+=1

        #check size of residues
        if np.any(np.array(atoms_ref_temp)!=temp_dom_atoms):
            where = np.argwhere(np.array(atoms_ref_temp)!=temp_dom_atoms)
            print("difference in number of atoms!!")
            print(np.array(atoms_ref_temp))
            print(top_df_dom["N_atoms_per_res"][0])
            print(f"""Are different here: 
    Target {np.array(atoms_ref_temp)[where]} 
    Domain {temp_dom_atoms[where]}""")

            print(f"At position: {where} \n")
            count_err_num_at+=1
        if CHECK_ATOM_ELEMENT:
            #check atom names of residues
            if ranges[i][0] > 1 : top_df_dom["atoms_name_per_res"][0][0] = np.delete(np.array(top_df_dom["atoms_name_per_res"][0][0]), [1,2])
            if ranges[i][1] < len(top_df["residues"][0]) : top_df_dom["atoms_name_per_res"][0][-1] = np.delete(np.array(top_df_dom["atoms_name_per_res"][0][-1]), [-1])
            if np.any(np.array(atoms_name_target)!=np.concatenate(top_df_dom["atoms_name_per_res"][0])):
                where = np.argwhere(np.array(atoms_name_target)!=np.concatenate(top_df_dom["atoms_name_per_res"][0]))
                print("difference in atoms!!")
                print(np.array(atoms_name_target))
                print(np.concatenate(top_df_dom["atoms_name_per_res"][0]))
                print(f"""Are different here: 
        Target {np.array(atoms_name_target)[where]} 
        Domain {np.concatenate(top_df_dom["atoms_name_per_res"][0])[where]}
        """)
                count_err_name_at +=1
        #print(top_df_dom)
        topol_domains.append(top_df_dom)

    #check that there are no mistake. If any break
    if count_err_res>0:
        print("Some of the residues don't match between full system and single domains")
        exit()
    elif count_err_num_at>0:
        print("Some of the residues have a different number of atoms between full system and single domains")
        exit()
    elif count_err_name_at>0:
        print("Some of the residues in the domains have atom name that don't match the full system")
        exit()
    else:
        print("<<< All checks passed! The single domain topologies are perfectly compatible with the full domain one >>>")

    return topol_domains    
