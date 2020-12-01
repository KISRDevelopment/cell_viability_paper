import numpy as np 
import pandas as pd 

hybrid_df = pd.read_csv('../generated-data/task_yeast_gi_hybrid')

print("Hybrid Dataset:")
print(" Observations: %d" % hybrid_df.shape[0])
print(" Interacting: %d (%0.2f %%)" % (np.sum(hybrid_df['bin'] != 1), np.mean(hybrid_df['bin'] != 1)*100))

costanzo_df = pd.read_csv('../generated-data/task_yeast_gi_costanzo')
print("Costanzo Dataset:")
print(" Observations: %d" % costanzo_df.shape[0])
print(" Interacting: %d (%0.2f %%)" % (np.sum(costanzo_df['bin'] != 1), np.mean(costanzo_df['bin'] != 1)*100))

hybrid_interacting_ix = hybrid_df['bin'] != 1
hybrid_interacting_pairs = set([tuple(sorted([r['a'], r['b']])) for i, r in hybrid_df[hybrid_interacting_ix].iterrows()])


costanzo_interacting_ix = costanzo_df['bin'] != 1
costanzo_interacting_pairs = set([tuple(sorted([r['a'], r['b']])) for i, r in costanzo_df[costanzo_interacting_ix].iterrows()])

print()
print("Interacting pairs shared between datasets: %d" % len(hybrid_interacting_pairs.intersection(costanzo_interacting_pairs)))

pairs = hybrid_interacting_pairs - costanzo_interacting_pairs
print("Interacting pairs in hybrid but not in costanzo: %d" % len(pairs))

pairs = costanzo_interacting_pairs - hybrid_interacting_pairs
print("Interacting pairs in costanzo but not in hybrid: %d" % len(pairs))