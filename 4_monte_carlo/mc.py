from ase import atoms
from ase.io import read
import numpy as np
from ase import units
from ase import Atom,Atoms
from ase.neighborlist import NeighborList
import random
import pickle
import os

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

def get_feature(config_int=[], neighbor_list={}):
    """
    This function extract short range order feature and atomic pair feature.
    Parameters 
    --------
    config_int: list, chemical symbols of PtCoM
    neighbor_list: dict, neighboring environment of PtCoM 

    Returns
    ------
    PAIR: list
    """
    # SRO
    config_int = [str(i) for i in config_int]
    sro_01 = get_sro(element_a='0', element_b='1',num_shell=3,element_list=config_int,neighbor_list=neighbor_list)
    sro_02 = get_sro(element_a='1', element_b='2',num_shell=3,element_list=config_int,neighbor_list=neighbor_list)
    sro_12 = get_sro(element_a='1', element_b='2',num_shell=3,element_list=config_int,neighbor_list=neighbor_list)
    
    # PAIR
    pairs = get_pair(element_list=config_int,neighbor_list=neighbor_list)

    Feature = sro_01 + sro_02 + sro_12 + pairs
    
    return Feature

# Read atomic structure from 'pure.traj'
atoms_initial = read('pure.traj')

# Copy initial structure by 3x3x3 to include periodic image
atoms = atoms_initial*[3,3,3]

# Define radii for neighbor search
radii = 1.4
radius = [1.4]*len(atoms)
nl = NeighborList(radius, self_interaction=False, bothways=True)
nl.update(atoms)

# Create neighbor lists for each atom
neighbor_list = {}
neighbor_multi_list = {}
coordination_list = {}

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


# Define a class for Monte Carlo simulation
class MenteCarlo:
    def __init__(self, temperature=300, n_step=50000, initial_config=[], out_path=''):
        
        # Initialize the Monte Carlo simulation
        self.out_path = out_path
        self.temperature = temperature
        self.n_step = n_step
        self.config = initial_config

        # Load data preprocessing and GPR model
        self.preprocessing = pickle.load(open("../3_machine_learning/Preprocessing.pkl", "rb"))
        self.model = pickle.load(open("../3_machine_learning/GPRmodel.model", "rb"))
        print('Initializing...')

    def evaluate(self, config_int, neighbor_list):
        # Evaluate the energy of a configuration
        feature = get_feature(config_int=config_int, neighbor_list=neighbor_list)
        feature = np.reshape(feature, (1, -1))
        feature = self.preprocessing.transform(feature)
        # Get uncetrainty given by GPR
        _, sigma_predict = self.model.predict(feature, return_std=True)
        return -sigma_predict

    def get_mutated_config_int(self, config_int=[]):
        # Generate a mutated configuration
        mutated_config_int = config_int.copy()

        # Choose an atom to be exchanged
        change_index_1 = random.choice([i for i in range(len(config_int))])

        # Choose another atom to be exchanged
        avail_change_indices = [i for i in range(len(config_int)) if config_int[change_index_1] != config_int[i]]
        if len(avail_change_indices) != 0:
            change_index_2 = random.choice(avail_change_indices)
        else:
            change_index_2 = change_index_1
        
        # Make change
        mutated_config_int[change_index_2] = config_int[change_index_1]
        mutated_config_int[change_index_1] = config_int[change_index_2]

        return mutated_config_int

    def perform_mc_iterations(self):

        # Perform Monte Carlo iterations
        print('Start MC iteration:')
        Energy = []
        Energy_min = []
        Config_int_mc_steps = []
        Config_int_min = []

        # Initialize
        config_int = self.config.copy()
        energy_prior = self.evaluate(config_int, neighbor_multi_list)
        energy_mutated = energy_prior
        energy_min = energy_prior
        config_int_min = config_int

        Energy.append(energy_prior)
        for _ in range(self.n_step):
            # Generate a mutated configuration
            config_int_mutated = self.get_mutated_config_int(config_int)
            
            # Evluate energy of mutated configuration 
            energy_mutated = self.evaluate(config_int_mutated, neighbor_multi_list)
            
            # Energy difference 
            delta_energy = energy_mutated - energy_prior
            theta = np.random.rand()
            mi = -delta_energy / (units.kB * self.temperature)
            mi = np.float(mi)
            p_a = np.exp(mi)
            prefilter = p_a - theta

            # Accept
            if delta_energy < 0:
                energy_prior = energy_mutated
                config_int = config_int_mutated
                
                # Update energy_min and config_int_min
                if energy_mutated < energy_min:
                    energy_min = energy_mutated
                    config_int_min = config_int_mutated

            # Accept with a probability
            elif prefilter > 0:
                energy_prior = energy_mutated
                config_int = config_int_mutated
            else:
                pass
            
            # Store the config_code and energy in MC iterations 
            Config_int_mc_steps.append(config_int)
            Config_int_min.append(config_int_min)
            Energy.append(energy_prior)
            Energy_min.append(energy_min)
        
        # Store the config_code and energy to csv file
        np.savetxt(self.out_path + '/energy.csv', Energy, delimiter=',')
        np.savetxt(self.out_path + '/energy_min.csv', Energy_min, delimiter=',')
        np.savetxt(self.out_path + '/config.csv', Config_int_mc_steps, delimiter=',')
        np.savetxt(self.out_path + '/config_min.csv', Config_int_min, delimiter=',')
        print('End!')

# Function to run Monte Carlo simulation
def mc_run(temperature=1, n_step=500, initial_config='', out_path=''):
    mc = MenteCarlo(temperature=temperature, n_step=n_step, initial_config=initial_config, out_path=out_path)
    mc.perform_mc_iterations()

# Shuffle initial structure and run Monte Carlo for different configurations
for mc_num in range(10):
    p = read('pure.traj')
    to_alloy_list = [pi.index for pi in p if pi.tag == 1]
    
    # Shuffle initial structure
    for i in range(1):
        random.shuffle(to_alloy_list)

    # Apply doping
    for i in range(int(len(to_alloy_list) / 4)):
        p[to_alloy_list[i]].symbol = 'Co'
    for i in range(int(len(to_alloy_list) / 4), int(len(to_alloy_list) / 2)):
        p[to_alloy_list[i]].symbol = 'Cu'
    for i in range(int(len(to_alloy_list) / 2), int(len(to_alloy_list))):
        p[to_alloy_list[i]].symbol = 'Pt'

    # Convert chemial symbols to config code. Pt:0, Co:1, Cu:2
    initial_config = []
    for atom in p:
        if atom.symbol == 'Pt':
            initial_config.append(0)
        elif atom.symbol == 'Co':
            initial_config.append(1)
        elif atom.symbol == 'Cu':
            initial_config.append(2)

    # Create a directory to store MC results
    folder = os.getcwd() + '/structure' + str(mc_num)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Run MC simulation
    mc_run(temperature=300, n_step=100, initial_config=initial_config, out_path='structure' + str(mc_num))
