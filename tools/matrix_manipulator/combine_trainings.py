import os
import sys
import argparse
import numpy as np
import pandas as pd
# import parmed as pmd
import time
# import warnings
# import gzip
# import tarfile
import h5py
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..",))
from src.multiego.util import mat_modif_functions as functions 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="join domains matrices and a full lenght matrix") 
    parser.add_argument('--config'   , type=str,  default=None, help='Config file with inputs')
    parser.add_argument('--check_overlap'     , type=bool, default=True , action=argparse.BooleanOptionalAction, help="If true checks overlap of matrices domains. default true, but memory expensive")
    parser.add_argument('--check_element'     , type=bool, default=True , action=argparse.BooleanOptionalAction, help="If true checks element differences betweeen domain top and global top. default true")
    args = parser.parse_args()

if args.config is not None:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

        # Access the systems
        systems = config["domains"]
        data = [
        {"system": system_name, "range": details["range"], "path_top": details["path_top"], "path_mat": details["path_mat"]}
        for system_name, details in systems.items()
        ]

        domains = pd.DataFrame(data)
        global_topology_path = config["background_mat"]["path_top"]
        global_matrix_path   = config["background_mat"]["path_mat"]

else:
    raise ValueError("No configuration file was provided")

#check existence of files
if not os.path.exists(global_topology_path):
    raise FileNotFoundError(f"Topology file {global_topology_path} not found")
if not os.path.exists(global_matrix_path):
    raise FileNotFoundError(f"Matrix file {global_matrix_path} not found")
for index, row in domains.iterrows():
    if not os.path.exists(row["path_top"]):
        raise FileNotFoundError(f"Topology file {row['path_top']} not found")
    if not os.path.exists(row["path_mat"]):
        raise FileNotFoundError(f"Matrix file {row['path_mat']} not found")
    
# read domain ranges
ranges_str = domains["range"].values

#convert input to array of tuples 
ranges = functions.dom_range(ranges_str)

#read the Target topology 
topology, top_df = functions.read_topologies(global_topology_path)

print(f'Target topology: \n     Residues:{len(top_df["residues"][0])}\n     Atoms{top_df["N_tot_atoms"][0]}')

#read the domains topologies
domains_tops    = domains["path_top"].values
paths_intramats = domains["path_mat"].values

#check single domains
topol_domains = functions.check_domain_topologies(domains_tops, top_df, args, ranges)

#read all intramats for each domain
print(f"\n:::Reading intramats for each domain:::")
dim = len(topology.atoms)
intramats = []
for i,intra in enumerate(paths_intramats):
    # time counter
    start_time = time.time()

    print(f"  -Domain {i+1}")
    print(f"   Reading intramat {intra}")
    index_mi = int(intra.split("/")[-1].split("_")[1])
    index_mj = int(intra.split("/")[-1].split("_")[2].split(".")[0])

    #load intramat
    intra_appo = np.loadtxt(intra, unpack=True)

    #check size of mat compatible with topology
    if topol_domains[i]["N_tot_atoms"][index_mi-1]*topol_domains[i]["N_tot_atoms"][index_mj-1] != len(intra_appo[0]):
        raise ValueError(f"""The number of atoms of molecule {index_mi} times molecule {index_mj} is different from that of matrix read
{topol_domains[i]['N_tot_atoms'][index_mi-1]*topol_domains[i]['N_tot_atoms'][index_mj-1]} {len(intra_appo[0])}""")

    #number of atoms of molecule 1 (protein)
    n_atoms_start = topol_domains[i]["N_tot_atoms"][index_mi-1]

    #find start and ending corresponding to the residues in input
    end   = functions.find_atom_end(topology, ranges[i][1])
    start = functions.find_atom_start(topology, ranges[i][0])

    print("     Target:")
    print("         Atoms start, end:", start, end)
    print("         Lenght range target: ", end + 1 -start)
    print("         Residues of start and end",topology.atoms[start], topology.atoms[end])
    print(f"    Domain:")
    print("         Number of atoms pre terminal removal :",n_atoms_start)

    # Generate the masks to remove the two hydrogens in the N terminal and the last oxigen in the C terminal
    mmask_i = functions.mask_terminals(top_df, i, ranges, intra_appo, n_atoms_start, 1)

    #intra-case: remove terminal hydrogens and oxygen also for j column
    if index_mi==index_mj and index_mi==1:

        mmask_j = functions.mask_terminals(top_df, i, ranges, intra_appo, n_atoms_start, 3)
        #remove the atoms from the matrix by mask indices
        mmask = np.concatenate((mmask_i[0], mmask_j[0]))
        intra_appo = np.delete(intra_appo, mmask, axis=1)
        n_atoms_after = np.sqrt(len(intra_appo[0]))
    
    #inter-case
    else: 
        # TODO make this more general should take into consideration the number of atoms of the second molecule
        mmask = mmask_i
        intra_appo = np.delete(intra_appo, mmask, axis=1)
        n_atoms_after = len(intra_appo[0])/topol_domains[i]["N_tot_atoms"][index_mj-1] 

    if float(end +1 -start)!=n_atoms_after:
        print(f"    Number of atoms in full system is not compatible with number of atoms in matrix after terminal removal: {end -start} vs {n_atoms_after}")
        exit()

    print(f"    Number of atoms Target vs Domain: {end + 1 -start} vs {n_atoms_after}")
    intramats.append(intra_appo)
    print(f"Loaded intramat {intra} in {time.time()-start_time:.2f} s\n")

print('<<< All matrices loaded >>>\n\n:::Reading global matrix:::')

#generating the starting final matrix
indices = np.arange(1,dim+1, 1)
dim_inter = topol_domains[0]["N_tot_atoms"][index_mj-1]
if index_mi==index_mj and index_mi==1:
    indices_2 = np.arange(1,dim+1, 1)
else:
    indices_2 = np.arange(1,dim_inter+1, 1)
row_indices    = []
column_indices = []
for i in range(len(indices)):
    for item in indices_2:
        row_indices.append(i + 1)
        column_indices.append(item)
row_indices = np.array(row_indices)
column_indices = np.array(column_indices)

#initializing all 8 columns
dim_matrix = len(indices)*len(indices_2)
final_intramat = np.zeros((7,dim_matrix))

#write indices
final_intramat[0] = np.zeros(dim_matrix)+index_mi
final_intramat[1] = row_indices
final_intramat[2] = np.zeros(dim_matrix)+index_mj
final_intramat[3] = column_indices

#free memory
del row_indices
del column_indices

#load the global matrix
start_time = time.time()
intra_FL = np.loadtxt(global_matrix_path, unpack=True)

#check size of mat compatible with topology
if len(intra_FL[0])!=dim_matrix:
    print(f"""
***ERROR***: The lenght of the full lenght matrix matrix ({global_matrix_path}): {len(intra_FL[0])} 
is different from the one obtaied from the reference topology {dim_matrix}. Check one of the two before continuing.""")
    exit()
else:
    print(f"    -FULL LENGTH MATRIX PASSED: {global_matrix_path} will be used as the basic matrix. On top will be added the single domains")
    final_intramat[4] = intra_FL[4]
    final_intramat[5] = intra_FL[5]
    final_intramat[6] = intra_FL[6]
print(f"    Loaded global matrix in {time.time()-start_time:.2f} s")

#Add the single matrices on the defined ranges
print("\n:::Starting to add the single matrices on the defined ranges:::")
for i in range(len(ranges)):
    start_time = time.time()

    #check weter is intra or inter and define the domain mask properly
    if index_mi==index_mj and index_mi==1:
        domain_mask_linear = functions.domain_mask(topology, ranges[i], dim)
    else:
        domain_mask_linear = functions.domain_mask_inter(topology, ranges[i], dim, dim_inter)

    #check intersection
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
    #final_intramat[7][np.where(domain_mask_linear)[0]] = 1

    #save new matrix in the last appended domain region to compare it with new next one (check for overlap)
    if args.check_overlap:
        old_intra = final_intramat[1][np.where(domain_mask_linear)[0]]
        old_intra_5 = final_intramat[5][np.where(domain_mask_linear)[0]]
        old_indices = ["{}_{}".format(x, y) for x, y in zip(final_intramat[1][np.where(domain_mask_linear)[0]], final_intramat[3][np.where(domain_mask_linear)[0]])]

    print(f"    -Added {paths_intramats[0].split('/')[-1].split('_')[0]} in ranges {ranges[i]} in {time.time()-start_time:.2f} s")

#save the final matrix
print("\n:::Saving the final matrix:::")
start_time = time.time()
if index_mi==index_mj and index_mi==1:

    with h5py.File('frankestein_intra_1_1.ndx.h5', 'w') as hf:
        hf.create_dataset('frankestein_intra_1_1.ndx', data=final_intramat.T)
    # np.savetxt(f'frankestein_intra_1_1.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f', '%1i'])
    print(f"    Ouput was saved in frankestein_intra_1_1.ndx in {time.time()-start_time:.2f} s")

else:
    #remove last column (inter_domain flag) not needed for inter case
    final_intramat= final_intramat[:-1] 
    np.savetxt(f'frankestein_inter_{index_mi}_{index_mj}.ndx',final_intramat.T, delimiter=" ", fmt = ['%i', '%i', '%i', '%i', '%2.6f', '%.6e', '%2.6f'])
    print(f"    Ouput was saved in frankestein_inter_{index_mi}_{index_mj}.ndx in {time.time()-start_time:.2f} s")
