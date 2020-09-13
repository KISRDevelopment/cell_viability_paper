import numpy as np 
import matplotlib.pyplot as plt 

colors = ['#FF0000', '#FFFF00', '#00CC00']
curve_colors = ['#FF0000', '#00CC00']

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

f, ax = plt.subplots(1, 1, figsize=(40, 10))

thetas = [-2, 2]

xstar = np.linspace(-5, 5, 1000)

ref_line = xstar * 0.

for i, theta in enumerate(thetas):
    ystar = 1/(1 + np.exp(-(theta-xstar)))
    ax.plot(xstar, ystar, linewidth=5, color=curve_colors[i], label=r'$\theta = %0.1f$' % theta)
    ax.fill_between(xstar, ref_line, ystar, color=colors[i], alpha=0.25)
    ref_line = ystar

ax.fill_between(xstar, ref_line, np.ones_like(xstar), color=colors[-1], alpha=0.25)
ax.plot(xstar, np.ones_like(xstar) * 0.5, linewidth=2, linestyle='dashed', color='grey')
ax.yaxis.set_tick_params(labelsize=60)
ax.xaxis.set_tick_params(labelsize=60)
ax.set_ylabel('Class Probability', fontsize=80, fontweight='bold')
ax.set_xlabel("$z$", fontsize=80)
ax.set_xlim([-5, 5])
ax.set_ylim([0, 1])
ax.yaxis.set_tick_params(length=10, width=1, which='both')
plt.setp(ax.spines.values(), linewidth=3, color='black')

ax.text(0.1, 0.1, 'Lethal', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=90)
ax.text(0.5, 0.5, 'Reduced Growth', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=90)
ax.text(0.9, 0.9, 'Normal', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=90)

plt.savefig("../figures/orm-logistic.png", bbox_inches='tight', dpi=100)

plt.show()
