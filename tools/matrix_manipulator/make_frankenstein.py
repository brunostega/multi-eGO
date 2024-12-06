import os
import re
import sys
import tempfile
import argparse
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import parmed as pmd
import time
import warnings
import gzip
import tarfile
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..",))

from tools.matrix_manipulator.functions import read_topologies
from src.multiego.io import read_config
from src.multiego import io
from src.multiego.arguments import args_dict




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

def check_domain_topologies(domains_tops, top_df, args, ranges):
    print("Check domain topology")
    count_err_res     = 0
    count_err_num_at  = 0
    count_err_name_at = 0
    topol_domains = []
    CHECK_ATOM_NAMES = args.check_names#;False #this will mostly always give error because atom names will be different. Use it only to check that atom types are the same
    for i,d in enumerate(domains_tops):
        #target topology stuff to be checked
        res_ref_temp = top_df["residues"][0][int(ranges[i][0]-1):int(ranges[i][1])]
        atoms_ref_temp = top_df["atoms_per_res"][0][int(ranges[i][0]-1):int(ranges[i][1])]
        atoms_name_target = np.concatenate(top_df["atoms_name"][0][int(ranges[i][0]-1):int(ranges[i][1])])

        #read domain top 
        topology_dom, top_df_dom= read_topologies(d)

        temp_dom_atoms = np.array(top_df_dom["atoms_per_res"][0])
        #remove the excess terminal oxigen
        if ranges[i][1] < len(top_df["residues"][0]): temp_dom_atoms[-1]-=1     
        if not args.skip_H:
            #remove the terminal excess hydroges to compare full system top and single domain top
            if ranges[i][0] > 1: temp_dom_atoms[0]-=2
        print(f"""\nDomain {i+1}: between ranges {ranges[i]} of full structure \n""")
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
            print(top_df_dom["atoms_per_res"][0])
            print(f"""Are different here: 
    Target {np.array(atoms_ref_temp)[where]} 
    Domain {temp_dom_atoms[where]}""")

            print(f"At position: {where} \n")
            count_err_num_at+=1
        if CHECK_ATOM_NAMES:
            #check atom names of residues
            if ranges[i][0] > 1 : top_df_dom["atoms_name"][0][0] = np.delete(np.array(top_df_dom["atoms_name"][0][0]), [1,2])
            if ranges[i][1] < len(top_df["residues"][0]) : top_df_dom["atoms_name"][0][-1] = np.delete(np.array(top_df_dom["atoms_name"][0][-1]), [-1])
            if np.any(np.array(atoms_name_target)!=np.concatenate(top_df_dom["atoms_name"][0])):
                where = np.argwhere(np.array(atoms_name_target)!=np.concatenate(top_df_dom["atoms_name"][0]))
                print("difference in atoms!!")
                print(np.array(atoms_name_target))
                print(np.concatenate(top_df_dom["atoms_name"][0]))
                print(f"""Are different here: 
        Target {np.array(atoms_name_target)[where]} 
        Domain {np.concatenate(top_df_dom["atoms_name"][0])[where]}
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
        print("All checks passed! The single domain topologies are perfectly compatible with the full domain one")

    return topol_domains    


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

########################################################################
# MAIN
########################################################################

# Workflow:
# 1) train full
# 2) rc full
# 3) ***i*** inter_dom rc        --> use train_mat only in domain ranges
# 4) ***ii*** mego inter_dom     --> combine rc and inter_domain_rc in ranges
# 5) ***iii*** combine trainings --> new trains with train full in defined ranges

# Should do:
# 1) apply single domain matrices on top of another one with careful attention on overlay and terminals
#       0 matrix --> inter_domain rc or mg
#       Full lenght matrix --> training combined
# 2) Turn on interactions of a 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
This tool generates a full matrix (needed for multi-ego input) from non-trivial condition. \nExamples: 
        \n1) if one has simulation of single parts of a big protein and wants to concatenate the matrices of each domain in the correct way (use input_intramats and full_lenght_matrix)
        \n2) if one wants to generate the matrix for an inter domain random coil (use input intramats)  
        \n3) if one needs to modify the random coil to include into the full lenght rc the inter-domain random coil (use single_intramat, full_lenght_matrix and skip_H)""")
        
    parser.add_argument('--config'   , type=str,  default=None, help='List of paths associated to the intramat for each domain')
    # parser.add_argument('--input_intramats'   , type=str,  default=None, help='List of paths associated to the intramat for each domain')
    # parser.add_argument('--single_intramat'   , type=str,  default=None, help='Path for the matrix to be added on the domain ranges')
    # parser.add_argument('--input_ranges'      , type=str,  default=None, help='List of residue ranges of the domains')
    # parser.add_argument('--target_top'        , type=str,  help='Target topology (the topology of the final full lenght structure)')
    # parser.add_argument('--domain_tops'       , type=str,  help='List of paths of each domain associated to the ranges')
    # parser.add_argument('--full_lenght_matrix', type=str,  help='full lenght matrix. If passed will be set as the default matrix on top of which the single domains will be added on')
    # parser.add_argument('--skip_H'            , type=bool, default=False, action=argparse.BooleanOptionalAction, help="If true doesn't consider hydrogens (doesn't remove the terminal hydrogens for the checks and the mask)")
    # parser.add_argument('--check_names'       , type=bool, default=False, action=argparse.BooleanOptionalAction, help="If true doesn't consider hydrogens (doesn't remove the terminal hydrogens for the checks and the mask)")
    # parser.add_argument('--check_overlap'     , type=bool, default=True , action=argparse.BooleanOptionalAction, help="If true checks overlap of matrices domains. default true, but memory expensive")

    args = parser.parse_args()

if args.config:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

        # Access the systems
        systems = config["domains"]
        data = [
        {"system": system_name, "range": details["range"], "path_top": details["path_top"], "path_mat": details["path_mat"]}
        for system_name, details in systems.items()
        ]

        df = pd.DataFrame(data)
        print(df)
#     config_yaml = io.read_config(args.config, args_dict)
#     # check if yaml file is empty
#     if not config_yaml:
#         print("WARNING: Configuration file was parsed, but the dictionary is empty")
#     else:
#         args = io.combine_configurations(config_yaml, args, args_dict)

# print(args)
exit()

if args.input_intramats is not None and args.single_intramat is not None:
    print(f"\n***ERROR*** you need to choose either the list of intramat for each domain or the full matrix. Not both\n")
    exit()

if args.input_intramats is  None and args.single_intramat is  None:
    print(f"\n***ERROR*** you need to choose either the list of intramat for each domain or the full matrix. Choose at least one of the two options\n")
    exit()

#read inputs
if args.input_intramats is not None: paths_intramats = np.loadtxt(args.input_intramats, dtype=str, comments=["#"])
ranges_str   = np.loadtxt(args.input_ranges, dtype=str, comments=["#"]) 
domains_tops = np.loadtxt(args.domain_tops, dtype=str, comments=["#"])

#convert input to array of tuples 
ranges = dom_range(ranges_str)

#read the Target topology 
topology, top_df = read_topologies(args.target_top)

print(f"""
Target topology is read: Contains {len(top_df["residues"][0])} residues and a total of {top_df["tot_atoms"][0]}
""")

#check single domains
topol_domains = check_domain_topologies(domains_tops, top_df, args, ranges)

#atoms in full system topology
dim = len(topology.atoms)

#if list of intramats is passed 
if args.input_intramats:
    #read all intramats for each domain
    intramats = []
    for i,intra in enumerate(paths_intramats):
        print(f"---Domain {i+1}")
        print(f"Reading intramat {intra}")
        index_i = int(intra.split("/")[-1].split("_")[1])
        index_j = int(intra.split("/")[-1].split("_")[2].split(".")[0])

        #load intramat
        intra_appo = np.loadtxt(intra, unpack=True)

        #check size of mat compatible with topology
        if topol_domains[i]["tot_atoms"][index_i-1]*topol_domains[i]["tot_atoms"][index_j-1] != len(intra_appo[0]):
            print(f"The number of atoms of molecule {index_i} times molecule {index_j} is different from that of matrix read")
            print(f'{topol_domains[i]["tot_atoms"][index_i-1]*topol_domains[i]["tot_atoms"][index_j-1]} {len(intra_appo[0])}')
            exit()

        #number of atoms of molecule 1 (protein)
        n_atoms_start = topol_domains[i]["tot_atoms"][index_i-1]

        #find start and ending corresponding to the residues in input
        end   = find_atom_end(topology, ranges[i][1])
        start = find_atom_start(topology, ranges[i][0])
        if ranges[i][1] < len(top_df["residues"][0]): 
            end_appo = end
        else:
            end_appo = end -1
        print("  Target:")
        print("     Atoms start, end:", start, end_appo)
        print("     Lenght range target: ", end_appo -start)
        print("     Residues of start and end",topology.atoms[start], topology.atoms[end_appo])
        print(f"  Domain:")
        print("     Number of atoms pre terminal removal :",n_atoms_start)

        #generating the masks to remove the two hydrogens in the N terminal and the last oxigen in the C terminal
        #if the residue of the domain corresponds to the first of the target topology do not remove terminal hydrogen
        if ranges[i][0] == 1:
            print("     i-column res=1 --> not removing N terminal hydrogens")
            mmask_1_base = np.zeros(len(intra_appo[1]))
        else:
            mmask_1_base = np.logical_or(intra_appo[1]==2, intra_appo[1]==3)
        #if not skip H remove also the two terminal hydrogens
        #TODO check this skipH part: seems wrong
        if not args.skip_H: 
            #If residue number is less then the last one f the target dom remove terminal hydrogen
            if ranges[i][1] < len(top_df["residues"][0]): 
                mmask_1 = np.where( np.logical_or(mmask_1_base, intra_appo[1]==n_atoms_start))
            else:
                print("     i-column res=last --> not removing C terminal Oxygen")
                mmask_1 = np.where( mmask_1_base )

        else: mmask_1 = np.where(mmask_1_base)

        #intra-case
        if index_i==index_j and index_i==1:
            #if the residue of the domain corresponds to the first of the target topology do not remove terminal hydrogen
            if ranges[i][0] == 1:
                print("     j-column res=1 --> not removing N terminal hydrogens")
                mmask_2_base = np.zeros(len(intra_appo[3]))
            else:
                mmask_2_base = np.logical_or(intra_appo[3]==2, intra_appo[3]==3)

            #if not skip H remove also the two terminal hydrogens
            if not args.skip_H: 
                if ranges[i][1] < len(top_df["residues"][0]): 
                    mmask_2 = np.where( np.logical_or(mmask_2_base, intra_appo[3]==n_atoms_start))
                else:
                    print("     j-column res=last --> not removing C terminal Oxygen")
                    mmask_2 = np.where( mmask_2_base )
                
            else: mmask_2 = np.where(mmask_2_base)

            mmask = np.concatenate((mmask_1[0], mmask_2[0]))
            intra_appo = np.delete(intra_appo, mmask, axis=1)
            n_atoms_after = np.sqrt(len(intra_appo[0]))
        
        #inter-case
        else: 
            mmask = mmask_1
            intra_appo = np.delete(intra_appo, mmask, axis=1)
            n_atoms_after = len(intra_appo[0])/topol_domains[i]["tot_atoms"][index_j-1] 

#        if ranges[i][1] < len(top_df["residues"][0]): 
#            end_appo = end
#        else:
#            end_appo = end + 1
        if float(end -start)!=n_atoms_after:
            print(f"Number of atoms in full system is not compatible with number of atoms in matrix after terminal removal: {end -start} vs {n_atoms_after}")
            exit()
        print(f"    Number of atoms Target vs Domain: {end -start} vs {n_atoms_after}")

        intramats.append(intra_appo)
        print(f"Loaded intramat {intra}\n")

        #define the indeces
        index_i = int(paths_intramats[0].split("/")[-1].split("_")[1])
        index_j = int(paths_intramats[0].split("/")[-1].split("_")[2].split(".")[0])

if args.single_intramat:
    #read single matrix
    single_intramat = np.loadtxt(args.single_intramat, unpack=True)
    #define the indeces from the matrix passed
    index_i = int(args.single_intramat.split("/")[-1].split("_")[1])
    index_j = int(args.single_intramat.split("/")[-1].split("_")[2].split(".")[0])

#generating the starting final matrix
indices = np.arange(1,dim+1, 1)

dim_inter = topol_domains[0]["tot_atoms"][index_j-1]

if index_i==index_j and index_i==1:
    indices_2 = np.arange(1,dim+1, 1)
else:
    indices_2 = np.arange(1,dim_inter+1, 1)
indices = np.arange(1,dim+1, 1)

row_indices = []
column_indices = []

for i in range(len(indices)):
    for item in indices_2:
        row_indices.append(i + 1)
        column_indices.append(item)

row_indices = np.array(row_indices)
column_indices = np.array(column_indices)

#initializing all 8 columns
dim_matrix = len(indices)*len(indices_2)
final_intramat = np.zeros((8,dim_matrix))

#write indices
final_intramat[0] = np.zeros(dim_matrix)+index_i
final_intramat[1] = row_indices
final_intramat[2] = np.zeros(dim_matrix)+index_j
final_intramat[3] = column_indices

if args.full_lenght_matrix is not None:
    intra_FL = np.loadtxt(args.full_lenght_matrix, unpack=True)
    if len(intra_FL[0])!=dim_matrix:
        print(f"""
***ERROR***: The lenght of the full lenght matrix matrix ({args.full_lenght_matrix}): {len(intra_FL[0])} 
is different from the one obtaied from the reference topology {dim_matrix}. Check one of the two before continuing.""")
        exit()
    else:
        print(f"FULL LENGTH MATRIX PASSED: {args.full_lenght_matrix} will be used as the basic matrix. On top will be added the single domains")
        final_intramat[4] = intra_FL[4]
        final_intramat[5] = intra_FL[5]
        final_intramat[6] = intra_FL[6]

if args.input_intramats:
    for i in range(len(ranges)):

        print(f"adding {paths_intramats[0].split('/')[-1].split('_')[0]} in ranges {ranges[i]}")

        #check weter is intra or inter and define the domain mask properly
        if index_i==index_j and index_i==1:
            domain_mask_linear = domain_mask(topology, ranges[i], dim)
        else:
            domain_mask_linear = domain_mask_inter(topology, ranges[i], dim, dim_inter)

        #check for intersection
        if i > 0 and args.check_overlap:
            new_indeces = ["{}_{}".format(x, y) for x, y in zip(final_intramat[1][np.where(domain_mask_linear)[0]], final_intramat[3][np.where(domain_mask_linear)[0]])]
            #find if there is overlap between consequent domains
            intersect, index_inters_a, index_inters_b = np.intersect1d(new_indeces, old_indices, return_indices=True)

            #if not emtpy compare intramat with previous one and modify new one so that is the previous one where d_prev<d_new
            if intersect.size > 0:
                print(f"****Found an overlap between ranges {ranges[i]} and range {ranges[i-1]}. \n****The contacts will be merged to consider the one with shortest distance or the one with information different from 0")
                #if new intramat probability is 0 use old information
                TEMP_INDEX = 4
                intramats[i][TEMP_INDEX][index_inters_a] = np.where(intramats[i][5][index_inters_a]==0, intramats[i-1][TEMP_INDEX][index_inters_b], intramats[i][TEMP_INDEX][index_inters_a])
                #if are both greater than 0 use the one which has the shortest distance (4 column)
                intramats[i][TEMP_INDEX][index_inters_a] = np.where(np.logical_and( np.logical_and(intramats[i][5][index_inters_a]!=0 , intramats[i-1][5][index_inters_b]!=0 ), intramats[i-1][4][index_inters_b] < intramats[i][4][index_inters_a]), intramats[i-1][TEMP_INDEX][index_inters_b], intramats[i][TEMP_INDEX][index_inters_a])

                TEMP_INDEX = 5
                intramats[i][TEMP_INDEX][index_inters_a] = np.where(intramats[i][5][index_inters_a]==0, intramats[i-1][TEMP_INDEX][index_inters_b], intramats[i][TEMP_INDEX][index_inters_a])
                #if are both greater than 0 use the one which has the shortest distance (4 column)
                intramats[i][TEMP_INDEX][index_inters_a] = np.where(np.logical_and( np.logical_and(intramats[i][5][index_inters_a]!=0 , intramats[i-1][5][index_inters_b]!=0 ), intramats[i-1][4][index_inters_b] < intramats[i][4][index_inters_a]), intramats[i-1][TEMP_INDEX][index_inters_b], intramats[i][TEMP_INDEX][index_inters_a])

                TEMP_INDEX = 6
                if np.any(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b]):
                    print("WARNING: some of the cutoffs in the overlapping region do not match. This should not happend")
                    print(f"{i-1}")
                    print(intramats[i-1][1][index_inters_b][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])
                    print(intramats[i-1][3][index_inters_b][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])
                    print(intramats[i-1][TEMP_INDEX][index_inters_b][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])
                    print(f"\n{i}")
                    print(intramats[i][1][index_inters_a][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])
                    print(intramats[i][3][index_inters_a][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])
                    print(intramats[i][TEMP_INDEX][index_inters_a][np.where(intramats[i][TEMP_INDEX][index_inters_a]!=intramats[i-1][TEMP_INDEX][index_inters_b])[0]])

        #append into final matrix the single domain ones (corrected in case of matrix overlap)
        final_intramat[4][np.where(domain_mask_linear)[0]] = intramats[i][4]
        final_intramat[5][np.where(domain_mask_linear)[0]] = intramats[i][5]
        final_intramat[6][np.where(domain_mask_linear)[0]] = intramats[i][6]
        final_intramat[7][np.where(domain_mask_linear)[0]] = 1

        #save new matrix in the last appended domain region to compare it with new next one (check for overlap)
        if args.check_overlap:
            old_intra = final_intramat[1][np.where(domain_mask_linear)[0]]
            old_intra_5 = final_intramat[5][np.where(domain_mask_linear)[0]]
            old_indices = ["{}_{}".format(x, y) for x, y in zip(final_intramat[1][np.where(domain_mask_linear)[0]], final_intramat[3][np.where(domain_mask_linear)[0]])]

    if index_i==index_j and index_i==1:
        if args.full_lenght_matrix is None:
            np.savetxt(f'frankestein_intra_1_1.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f', '%1i'])
            print(f"     Ouput was saved in frankestein_intra_1_1.ndx")
        else:
            np.savetxt(f'frankestein_intra_1_1_merged.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f', '%1i'])
            print(f"     Ouput was saved in frankestein_intra_1_1_merged.ndx")

    else:
        #remove last column (inter_domain flag) not needed for inter case
        final_intramat= final_intramat[:-1] 
        np.savetxt(f'frankestein_inter_{index_i}_{index_j}.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f'])
        print(f"     Ouput was saved in frankestein_inter_{index_i}_{index_j}.ndx")

if args.single_intramat:
    print("\nSINGLE matrix passed: adding this on the defined ranges")
    for i in range(len(ranges)):

        print(f"Adding {args.single_intramat} in ranges {ranges[i]}")

        #check weter is intra or inter and define the domain mask properly
        if index_i==index_j and index_i==1:
            domain_mask_linear = domain_mask(topology, ranges[i], dim)
        else:
            domain_mask_linear = domain_mask_inter(topology, ranges[i], dim, dim_inter)

        final_intramat[4][np.where(domain_mask_linear)[0]] = single_intramat[4][np.where(domain_mask_linear)[0]]
        final_intramat[5][np.where(domain_mask_linear)[0]] = single_intramat[5][np.where(domain_mask_linear)[0]]
        final_intramat[6][np.where(domain_mask_linear)[0]] = single_intramat[6][np.where(domain_mask_linear)[0]]
        final_intramat[7][np.where(domain_mask_linear)[0]] = 1

    if index_i==index_j and index_i==1:
        if args.full_lenght_matrix is None:
            np.savetxt(f'grouped_intra_1_1.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f', '%1i'])
            print(f"\n    Output was saved in grouped_intra_1_1.ndx")
        else:
            np.savetxt(f'grouped_intra_1_1_merged.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f', '%1i'])
            print(f"\n    Output was saved in grouped_intra_1_1_merged.ndx")

    else:
        #remove last column (inter_domain flag) not needed for inter case
        final_intramat= final_intramat[:-1] 
        np.savetxt(f'grouped_inter_{index_i}_{index_j}.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f'])
        print(f"\n    Output was saved in grouped_inter_{index_i}_{index_j}.ndx")
