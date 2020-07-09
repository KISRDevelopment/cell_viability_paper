import numpy as np 
import matplotlib.pyplot as plt 

colors = ['#FF0000', '#FFFF00', '#00CC00']
curve_colors = ['#FF0000', '#00CC00']
plt.rcParams["font.family"] = "Liberation Serif"
f, ax = plt.subplots(1, 1, figsize=(20, 10))

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
ax.yaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_ylabel(r'$Probability$', fontsize=26)
ax.set_xlabel("$z$", fontsize=26)
ax.set_xlim([-5, 5])
ax.set_ylim([0, 1])
ax.yaxis.set_tick_params(length=10, width=1, which='both')
#ax.legend(fontsize=18)
plt.setp(ax.spines.values(), linewidth=3, color='black')

plt.savefig("../tmp/orm-logistic.png", bbox_inches='tight', dpi=100)

plt.show()
