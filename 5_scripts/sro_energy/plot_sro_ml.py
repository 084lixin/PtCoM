from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load machine learning predicted data 
data = np.loadtxt("sro_ml.csv", delimiter=",", encoding='utf-8-sig')
rel_energy = np.loadtxt("energy_ml.csv", delimiter=",", encoding='utf-8-sig') 
sro1 = data[:,0]
sro2 = data[:,1]
rel_energy = [i for i in rel_energy]

# Calculte the mean relative energy at each sro_pt_m, sro_co_cu combination
data = {'sro1': sro1,
        'sro2': sro2,
        'rel_energy': rel_energy}
df = pd.DataFrame(data)
result_dict = {}
for name, group in df.groupby(['sro1', 'sro2']):
    min_c_value = group['rel_energy'].mean()
    result_dict[name] = min_c_value

# Save result 
sro1_ml = []
sro2_ml = []
rel_energy_ml = []
for key, value in result_dict.items():
    sro1_ml.append(key[0])
    sro2_ml.append(key[1])
    rel_energy_ml.append(value)

# Load DFT calculated data 
data_dft = np.loadtxt("sro_dft.csv", delimiter=",", encoding='utf-8-sig')
rel_energy_dft = np.loadtxt("energy_dft.csv", delimiter=",", encoding='utf-8-sig') 
sro1_dft = data_dft[:,0]
sro2_dft = data_dft[:,1]
rel_energy_dft = [i/108 for i in rel_energy_dft]

plt.rc('font',family='Arial')
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.xlim(-0.60,0.80)
plt.ylim(-1.1,1.1)
plt.xticks([-0.40,-0.20,0,0.20,0.40,0.60])
plt.yticks([-0.9,-0.6,-0.3,0, 0.3,0.6,0.9])

plt.scatter(sro1_ml, sro2_ml, c=rel_energy_ml, cmap='jet_r',marker='o', s=10,)

cbar=plt.colorbar()
cbar.set_label('Relative energy (eV/Atom)',fontsize=16)
plt.tick_params(labelsize=14)

plt.xlabel(r'$\rm \alpha_{1}^{Pt-(Cu,Co)}$',fontsize=18)
plt.ylabel(r'$\rm \alpha_{1}^{Cu-Co}$',fontsize=18)
plt.legend(['ML'])
plt.savefig('sro_ptcocu_ml.png',bbox_inches='tight')
plt.show()