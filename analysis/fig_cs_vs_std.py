import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy.random as rng 
import sys 

LETHAL_COLOR, SICK_COLOR, HEALTHY_COLOR = ['red', '#978c8c', 'green']  


plt.rcParams["font.family"] = "Liberation Serif"


def main(path, output_path, show=True):
    
    df = pd.read_csv(path)
    bins = np.array(df['bin'])

    cs = df['cs']
    std = df['std']


    healthy_ix = bins == 2
    sick_ix = bins == 1
    lethal_ix = bins == 0

    ratio = np.max(cs) / np.max(std)
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(cs[healthy_ix], std[healthy_ix], 'o', markersize=3, alpha=1, color=HEALTHY_COLOR, label='Normal')
    ax.plot(cs[sick_ix], std[sick_ix], 'o', markersize=3, alpha=1, color=SICK_COLOR, label='Reduced Growth')
    
    ax.plot(0.02 * rng.randn(np.sum(lethal_ix)), 0.02*rng.randn(np.sum(lethal_ix)) / ratio, 'o', color=LETHAL_COLOR, markersize=3, 
        label='Lethal')

    ax.set_xlabel('Colony Size', fontsize=22)
    ax.set_ylabel('Standard Deviation', fontsize=22)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=10, width=1, which='both')
    
    ax.set_xlabel('', fontsize=22)
    ax.set_ylabel('', fontsize=22)

    ax.grid(False)
    plt.setp(ax.spines.values(), linewidth=2, color='black')

    plt.savefig(output_path, bbox_inches='tight', dpi=100)

    if show:
        plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]

    main(path, output_path)
