import pandas as pd
import numpy as np
import sys 

task_path = sys.argv[1]

df = pd.read_csv(task_path)

if 'a_id' in df.columns:
    n_genes = len(set(df['a_id']) | set(df['b_id']))
else:
    n_genes = len(set(df['id']))

print("Number of genes: %d" % n_genes)
print("Output breakdown:")
bins = sorted(set(df['bin']))
for b in bins:
    print("%d: %d" % (b, np.sum(df['bin'] == b)))
