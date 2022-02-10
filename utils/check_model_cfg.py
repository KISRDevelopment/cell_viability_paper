import numpy as np 
import json 
import sys
file_path = sys.argv[1]

d = np.load(file_path, allow_pickle=True)

cfg = d['cfg'].item()

print(json.dumps(cfg, indent=4))

print(d['cm'] / np.sum(d['cm'], axis=1, keepdims=True))
if 'labels' in d:
    print(d['labels'])
    print(len(d['labels']))

print(list(d.keys()))
print(len(d['y_target']))