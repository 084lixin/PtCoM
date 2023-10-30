import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib

kT = 0.025852 # unit eV
k0 = 200
font1 = {
    'family': 'arial',
    'weight': 'normal',
    'size': 16,
}

def activity(eoh, eo):
    # shift energy relative to Pt
    eoh = eoh + 0.91
    eo = eo + 1.39

    g0_u0 = eo - 2.45
    g1_u0 = eoh - eo + 0.97
    g2_u0 = -eoh + 1.48
    ea_diss = 1.8 * eo - 2.89
    k = k0 * np.exp(-np.maximum(g1_u0, g2_u0)/kT)
    vdiss = k0 * np.exp(0.13/kT)
    kdiss = vdiss * np.exp(-ea_diss/kT)
    act = kT * np.log(np.minimum(k, kdiss)/k0)
    return act

N = 1000
x = np.linspace(-0.2, 0.4, N)
y = np.linspace(-0.15, 0.55, N)
X, Y = np.meshgrid(x, y)
Z = activity(X, Y)

fig, ax = plt.subplots(figsize=(6,5))
plt.rc('font',family='Arial')
ax.tick_params(direction="in")

# single color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colors = [(1, 1, 1), (1,0.9,0.9), (1,0.7,0.7), (1, 0, 0)]
n_bin = 100
newcmp = LinearSegmentedColormap.from_list(
        'newcmp', colors, N=n_bin)

cs = ax.contourf(X, Y, Z, levels=np.arange(-0.8, -0.2, 0.05),
                  linewidths=0.2, cmap=newcmp, extend = 'both',)
cs1 = ax.contour(X, Y, Z, levels=np.arange(-0.8, -0.2, 0.05),
                 colors='black', alpha=0.0, linewidths=0.2, linestyles='solid', extend='both')
cbar = plt.colorbar(cs, label='Activity', shrink=0.8, 
                    ticks=np.arange(-0.8, -0.19, 0.2),extend = 'both')
cbar.set_label('Activity', fontdict=font1, fontsize=16)
plt.tick_params(labelsize=14)

# PtCoCu, PtCoNi, PtCoGa, PtCoTi
deltaEO = [1.67, 1.78, 1.54, 1.45]
deltaEOH = [0.94, 1.01, 0.94, 0.91]

# Disorder PtCo
deltaEO +=  [1.67, 1.66, 1.61, 1.64, 1.66] 
deltaEOH += [1.00, 0.94, 0.93, 1.01, 1.04]

# Order PtCo
deltaEO.append(1.82)
deltaEOH.append(1.06)

# Pt
deltaEO.append(1.39)
deltaEOH.append(0.91)

# Shift energy relative to Pt
deltaEO = [i-1.39 for i in deltaEO]
deltaEOH = [i-0.91 for i in deltaEOH]

Labels = [r'$\rm Pt_{2}CoCu$',r'$\rm Pt_{2}CoNi$',r'$\rm Pt_{2}CoGa$',r'$\rm Pt_{2}CoTi$',]

# order Pt2CoM
for i in [0, 1, 2, 3]:
    plt.scatter(deltaEOH[i],deltaEO[i],facecolor='w',edgecolor='k',s=100)
    plt.text(deltaEOH[i]-0.06,deltaEO[i]+0.02, Labels[i])

# random PtCo
for i in [4, 5, 6, 7, 8]:
    plt.scatter(deltaEOH[i],deltaEO[i],facecolor=(0.05,0.52,0.78),edgecolor='k',s=100)

# order PtCo
for i in [9]:
    plt.scatter(deltaEOH[i],deltaEO[i],facecolor=(1,0.64,0.06),edgecolor='k',s=100)
    plt.text(deltaEOH[i]-0.02,deltaEO[i]+0.02,'PtCo')

# Pt
for i in [10]:
    plt.scatter(deltaEOH[i],deltaEO[i],facecolor='w',edgecolor='k',s=100)
    plt.text(deltaEOH[i],deltaEO[i]+0.02,'Pt')

plt.xlabel(r'$\rm \Delta E_{OH} - \Delta E^{Pt}_{OH} $ (eV)', font1)
plt.ylabel(r'$\rm \Delta E_{O} - \Delta E^{Pt}_{O} $ (eV)', font1)
plt.savefig('volcano.png',bbox_inches='tight')
plt.show()