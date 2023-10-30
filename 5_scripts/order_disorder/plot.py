import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(6,5))
plt.rc('font',family='Arial')
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

energy_order = np.loadtxt("energy_order.csv", delimiter=",", encoding='utf-8-sig')[0:800]
energy_random = np.loadtxt("energy_random.csv", delimiter=",", encoding='utf-8-sig')[0:800]

energy_order = pd.Series(energy_order)
energy_random = pd.Series(energy_random)

colors = ['coral','m']
energy_order.plot(kind = 'kde',bw_method=0.1, color=colors[0])
energy_random.plot(kind = 'kde',bw_method=0.2, color=colors[1])
plt.legend(['ordered alloy','random alloy'],fontsize=14, loc=2, frameon=False)

plt.hist(energy_order,color=colors[0],bins=20,density=True, )
plt.hist(energy_random,color=colors[1],bins=9,density=True, )
plt.tick_params(labelsize=16, length=5) #刻度字体大小12,刻度线长度5

plt.xlim([-0.02,0.08])
plt.ylim([0,200])
plt.yticks([0,50,100,150,200])
plt.axvline(np.mean(energy_order),ymin=0.25, ymax=0.75, color='coral',linestyle='--')
plt.axvline(np.mean(energy_random),ymin=0.25, ymax=0.75, color='m',linestyle='--')
 
plt.xlabel('Relative energy (eV/Atom)',fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.savefig('random_order.png', bbox_inches = 'tight')
plt.show()