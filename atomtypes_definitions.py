import pandas as pd
from read_input import read_gro_atoms

# Script containing the definitions of the atomtypes of every aminoacid used in here. 

# The informations are taken from the atomtypes section of gromos topology

# Gromos atomtypes definition
# Abbiamo tenuto solo le info che ci servivano. Da aggiungere le definizioni dei terminali e quelli che serviranno
# per i prossimi aminoacidi. Abbiamo aggiunto gli H nella massa (e nel c12) - OA ed N -
# per aggiungere C12 "regola del 20".

# Gromos 54a7
# Gromos dictionary definitions of:
    # dict_gro_atomtypes -> number:residue_atom 1:TYR_N
    # gromos_mass -> atom_type:mass N:15.0067
    # res_atom_dict -> residue_atom:chemical_type TYR_N:N

gro_atoms = read_gro_atoms()

# This one will be used in dihedrals
dict_gro_atomtypes = gro_atoms.set_index('; nr')['res_atom'].to_dict()

# This one will be used in pairs (for resnr comparison)
resnr_pairs = gro_atoms.set_index('atom_nmr')['resnr'].to_dict()

# Those ones will be used in make_atomtypes_and_dict
gromos_mass = gro_atoms[['type', 'mass']].drop_duplicates(subset=['type'], keep='first').copy()
gromos_mass_dict = gromos_mass.set_index('type')['mass'].to_dict() 
gromos_res_atom_dict = gro_atoms.set_index('res_atom')['type'].to_dict()


print(' REMEMBER TO CHANGE THE MASSES IN THE ATOMTYPES.ATP AND FFNONBONDED.ITP, THE H ARE EXPLICIT')

# This one will be used for proper dihedrals from SMOG to past to GROMOS topology
gromos_resatom_nmr = gro_atoms[['res_atom', 'resnr', '; nr']].copy()
gromos_resatom_nmr['resnr_2'] = gromos_resatom_nmr['resnr'].astype(str)
gromos_resatom_nmr['res_atom'] = gromos_resatom_nmr['res_atom'] + '_' + gromos_resatom_nmr['resnr_2']
gromos_resatom_nmr_dict = gromos_resatom_nmr.set_index('res_atom')['; nr'].to_dict()



# GRETA TO KEEP
# THE C12 RATIO CHANGED a little bit.atomtypes
gromos_atp = pd.DataFrame(
    {'name': ['O', 'OA', 'N', 'C', 'CH1', 'CH2', 'CH3', 'CH2r', 'NT', 'S', 'NR', 'OM', 'NE', 'NL', 'NZ'],
     'mass': [16, 17, 15, 12, 13, 14, 15, 14, 17, 32, 14, 16, 15, 17, 16],
     'at.num': [8, 8, 7, 6, 6, 6, 6, 6, 7, 16, 7, 8, 7, 7, 7],
     'c12': [1e-06, 3.011e-05, 4.639e-05, 4.937284e-06, 9.70225e-05,
            3.3965584e-05, 2.6646244e-05, 2.8058209e-05, 1.2e-05, 1.3075456e-05,
            3.389281e-06, 7.4149321e-07, 2.319529e-06, 2.319529e-06, 2.319529e-06]
     }
)

gromos_atp.to_dict()
gromos_atp.set_index('name', inplace=True)


# List of atoms which change the charge at pH 2/2.5
# ASP - OD1, OD2, CG
# GLU - OE1, OE2, CD
# HIS - ND1, CE1, NE2, CD2, CG

#print(gro_atoms)

# Selection of the aminoacids and the charged atoms

acid_ASP = gro_atoms[(gro_atoms['residue'] == "ASP") & ((gro_atoms['atom'] == "OD1") | (gro_atoms['atom'] == "OD2") | (gro_atoms['atom'] == "CG"))]
acid_GLU = gro_atoms[(gro_atoms['residue'] == "GLU") & ((gro_atoms['atom'] == "OE1") | (gro_atoms['atom'] == "OE2") | (gro_atoms['atom'] == "CD"))]
acid_HIS = gro_atoms[(gro_atoms['residue'] == "HIS") & ((gro_atoms['atom'] == "ND1") | (gro_atoms['atom'] == "CE1") | (gro_atoms['atom'] == "NE2") | (gro_atoms['atom'] == "CD2") | (gro_atoms['atom'] == "CG"))]

# Append the three dataframes

frames = [acid_ASP, acid_GLU, acid_HIS]
acid_atp = pd.concat(frames, ignore_index = True)

        # ASP 21 +
        # GLU 24 +
        # HIS 20 =
        # TOT 65 
        # acid_aa 65 rows yay!

#acid_atp = acid_atp.drop(['; nr', 'type', 'resnr', 'residue', 'atom', 'cgnr', 'charge', 'mass', 'res_atom'], axis = 1)


# prova passando da df a list

acid_atp = acid_atp['atom_nmr'].tolist()

#print(acid_atp)

#print(acid_ASP.count())
#print(acid_GLU.count())
#print(acid_HIS.count())

#print(acid_atp.to_string())
#print(acid_atp.count())
