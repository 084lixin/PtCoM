import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("sro_ml.csv", delimiter=",", encoding='utf-8-sig')


data_dft = np.loadtxt("sro_dft.csv", delimiter=",", encoding='utf-8-sig')
rel_energy_dft = np.loadtxt("energy_dft.csv", delimiter=",", encoding='utf-8-sig') 
sro1_dft = data_dft[:,0]
sro2_dft = data_dft[:,1]
rel_energy_dft = [i/108 for i in rel_energy_dft]

plt.rc('font',family='Arial')
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.xlim(-0.60,0.80)
plt.xticks([-0.40,-0.20,0,0.20,0.40,0.60])
plt.ylim(-1.1,1.1)
plt.yticks([-0.9,-0.6,-0.3,0, 0.3,0.6,0.9])

plt.scatter(sro1_dft,sro2_dft,c=rel_energy_dft,cmap='jet_r',marker='^',s=50)
cbar=plt.colorbar()
cbar.set_label('Relative energy (eV/Atom)',fontsize=16)
plt.tick_params(labelsize=14)

plt.xlabel(r'$\rm \alpha_{1}^{Pt-(Cu,Co)}$',fontsize=18)
plt.ylabel(r'$\rm \alpha_{1}^{Cu-Co}$',fontsize=18)

plt.savefig('sro_energy_dft.png',bbox_inches='tight')
plt.show()