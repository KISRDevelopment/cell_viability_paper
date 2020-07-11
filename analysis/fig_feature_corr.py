import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 


plt.rcParams["font.family"] = "Liberation Serif"

def main(path, output_path, show=True):
    
    d = np.load(path)
    F = d['F']
    feature_labels = [fix_name(l) for l in d['feature_labels'].tolist()]
    
    df = pd.DataFrame(data=F, columns=feature_labels)
    
    corr = df.corr('spearman')

    f, ax = plt.subplots(1,1, figsize=(20, 15))
    cax = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1, origin='lower')
    
    cbar = f.colorbar(cax)
    cbar.ax.tick_params(labelsize=40) 

    ticks = np.arange(0, len(df.columns))
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns, fontsize=40)
    ax.set_yticklabels(df.columns, fontsize=40)

    plt.savefig(output_path,bbox_inches='tight',dpi=100)
    if show:

        plt.show()

def fix_name(s):
    s = s.replace('_centrality','').replace('_cent','').capitalize()
    s = s.replace('_', ' ')
    return s
if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]


    main(path, output_path)