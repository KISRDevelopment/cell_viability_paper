import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy.random as rng 
import sys 

LETHAL_COLOR, SICK_COLOR, HEALTHY_COLOR = ['red', '#978c8c', 'green']  

plt.rcParams["font.family"] = "Liberation Serif"


def main(task_path, feature_file, fid, output_path, remove_zeros=False):

    d = np.load(feature_file)
    labels = d['feature_labels'].tolist()
    f = d['F'][:, fid]
    f = f * d['std'][fid] + d['mu'][fid]

    df = pd.read_csv(task_path)
    cs = df['cs']
    
    x = f[df['id']]
    if remove_zeros:
        ix = x > 0
        x = x[ix]
        cs = cs[ix]
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(x, cs, 'o', markersize=3)
    
    print(np.max(x))
    z = np.polyfit(x, cs, 1)
    p = np.poly1d(z)
    
    xstar = np.linspace(np.min(x), np.max(x), 1000)
    ax.plot(xstar, p(xstar), 'r--', linewidth=2)

    ax.set_ylabel('Colony Size', fontsize=22)
    ax.set_xlabel(labels[fid], fontsize=22)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=10, width=1, which='both')
    
    ax.grid(False)
    plt.setp(ax.spines.values(), linewidth=2, color='black')
    plt.savefig(output_path, bbox_inches='tight', dpi=100)

if __name__ == "__main__":
    task_path = sys.argv[1]
    feature_file = sys.argv[2]
    fid = int(sys.argv[3])
    output_path = sys.argv[4]

    main(task_path, feature_file, fid, output_path)
