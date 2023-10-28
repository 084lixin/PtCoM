import ase.db
from ase.io import read
from ase.neighborlist import NeighborList
import numpy as np


def conversion_base(n, d):
    """
    This function convert the number from 10-base to n-base, 2<n<9
    
    Parameters 
    --------
    n: int
    d: int

    Returns
    ------
    d_n_base: int 

    """
    if n<d:
        d_n_base = n
        return d_n_base
    
    else:
        res_str=''
        while(n//d >= 1):
            res_str += str((n%d))
            n = n//d
        if(n%d != 0):
            res_str += str((n%d))
        d_n_base = int(res_str[::-1])
        return d_n_base

def get_sro(element_a='0', element_b='1', num_shell=3, element_list=[], neighbor_list={}):
    """
    This function extract short range order features based on chemical symbols and neighboring environment of PtCoM.
    Parameters 
    --------
    element_a: str, (1,)
    element_b: str, (1,)
    num_shell: int, length of the short range order
    element_list: list, chemical symbols of PtCoM
    neighbor_list: dict, neighboring environment of PtCoM 

    Returns
    ------
    SRO: list
    """
    # calculate the ratio of element_a in the element list
    ratio_a = element_list.count(element_a)/len(element_list)

    SRO = [] # A list to store SRO values

    for shell in range(0,num_shell):
        # For different atomic shells
        b_neighbor_num = 0 # Total number of neighboring atoms of element b
        b_neighbor_a_num = 0 # Total number of element a in neighboring atoms of element b
        
        # Iterate through the element list to find neighbors of element b
        for i in range(len(element_list)):
            if element_list[i] == element_b:
                b_neighbor_list = neighbor_list[i][shell]
                b_neighbor_symbol = [element_list[j] for j in b_neighbor_list]
                b_neighbor_num += len(b_neighbor_symbol)
                b_neighbor_a_num += b_neighbor_symbol.count(element_a)

        # Calculate the SRO value and append it to the SRO list
        sro = 1 - b_neighbor_a_num/b_neighbor_num/ratio_a
        SRO.append(sro)

    return SRO

def get_pair(element_list=[], neighbor_list={}):
    """
    This function extract atomic pair features based on chemical symbols and neighboring environment of PtCoM.
    Parameters 
    --------
    element_list: list, chemical symbols of PtCoM
    neighbor_list: dict, neighboring environment of PtCoM 

    Returns
    ------
    PAIR: list
    """

    # Initialize a dictionary to count pairs of elements
    pair_dict = {('0','1'):0, ('0','2'):0, ('1','2'):0, ('0','0'):0, ('1','1'):0, ('2','2'):0}
    
    # Calculate the total number of possible pairs
    pair_total = 12 * len(element_list)

    # Iterate through the element list to count pairs
    for i in range(len(element_list)):
        neighbor_i = neighbor_list[i][0]
        for j in range(len(neighbor_i)):
            # Create a sorted tuple for the pair of elements
            pair = tuple(sorted([element_list[i], element_list[neighbor_i[j]]]))
            pair_dict[pair] += 1
    
    # Calculate the pair frequencies and store them in a result list
    PAIR = [value / pair_total for value in pair_dict.values()]
    return PAIR

# Read atomic structure from 'pure.traj'
atoms_initial = read('pure.traj')

# Copy initial structure by 3x3x3 to include periodic image
atoms = atoms_initial * [3,3,3]

# Define radii for neighbor search
radii = 1.4
radius = [1.4]*len(atoms)
nl = NeighborList(radius, self_interaction=False, bothways=True)
nl.update(atoms)

# Create neighbor lists for each atom
neighbor_list = {}
coordination_list = {}
neighbor_multi_list = {}

for index in range(len(atoms)):
    indices, offsets = nl.get_neighbors(index)
    indices = indices.tolist()
    neighbor_list[index] = indices 
    coordination_list[index] = len(indices)

# Generate multi-layer neighbor information for specific atoms
neighbor_multi_list = []
for i in range(13*len(atoms_initial),14*len(atoms_initial)):
    neigh_i_dict = {}
    first_shell_i = neighbor_list[i].copy()
    total_shell_i = first_shell_i.copy()

    second_shell_i = []
    for j in first_shell_i:
        shell_ij = neighbor_list[j].copy()
        for k in shell_ij:
            if (k not in total_shell_i) and (k not in second_shell_i):
                second_shell_i.append(k)
                total_shell_i.append(k)
            elif k not in total_shell_i:
                total_shell_i.append(k)

    third_shell_i = []
    for j in second_shell_i:
        shell_ij = neighbor_list[j].copy()
        for k in shell_ij:
            if (k not in total_shell_i) and (k not in third_shell_i):
                third_shell_i.append(k)
                total_shell_i.append(k)
            elif k not in total_shell_i:
                total_shell_i.append(k)

    neigh_i_dict[0] = [i%108 for i in first_shell_i.copy()]
    neigh_i_dict[1] = [i%108 for i in second_shell_i.copy()]
    neigh_i_dict[2] = [i%108 for i in third_shell_i.copy()]
    neighbor_multi_list.append(neigh_i_dict)

# Connect to the database and extract features
db = ase.db.connect('../1_database/ptcocu.db')
Feature = []

for row in db.select(status='relaxed'):
    # Convert the hexadecimal config_code to config_int in base 3  
    uid = row.uid
    begin_index = uid.find('0x')
    config_code = uid[begin_index:]
    config_str = '{:0108}'.format(conversion_base(int(config_code,16),3))
    config_int = [i for i in config_str]
    
    # Calculate sro for different element pairs
    sro_01 = get_sro(element_a='0', element_b='1',num_shell=3,element_list=config_int,neighbor_list=neighbor_multi_list)
    sro_02 = get_sro(element_a='1', element_b='2',num_shell=3,element_list=config_int,neighbor_list=neighbor_multi_list)
    sro_12 = get_sro(element_a='1', element_b='2',num_shell=3,element_list=config_int,neighbor_list=neighbor_multi_list)

    # Calculate pair for different element pairs
    pairs = get_pair(element_list=config_int,neighbor_list=neighbor_multi_list)

    Feature.append(sro_01 + sro_02 + sro_12 + pairs)

# Save feature data to 'feature.csv'
np.savetxt('feature.csv', Feature, delimiter=',') 

# Extract and save energy data to 'energy.csv'
Energy = []
for row in db.select(status='relaxed'):
    Energy.append(row.energy + 620.253345) # 620.253345: energy of order PtCoCu
np.savetxt('energy.csv', Energy, delimiter=',')