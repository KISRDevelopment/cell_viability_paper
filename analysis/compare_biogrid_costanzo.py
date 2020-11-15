import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

COSTANZO_PATH = '../generated-data/costanzo_gi'
BIOGRID_PATH = '../generated-data/biogrid_yeast'

cdf = pd.read_csv(COSTANZO_PATH)
cdf_pairs = [",".join(sorted((a,b))) for a,b in zip(cdf['a'], cdf['b'])]
cdf_cs = dict(zip(cdf_pairs, cdf['gi']))

bdf = pd.read_csv(BIOGRID_PATH)
bdf_pairs = [",".join(sorted((a,b))) for a,b in zip(bdf['a'], bdf['b'])]
bdf['gi'] = [cdf_cs[p] if p in cdf_cs else np.nan for p in bdf_pairs]

f, axes = plt.subplots(4, 1, figsize=(20, 20))
for b in [0,1,2,3]:

    gis = bdf[bdf['bin'] == b]['gi']
    axes[b].hist(gis, bins=20)
plt.show()