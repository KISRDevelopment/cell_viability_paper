import numpy as np 
import json 
import sys 

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goids_to_names = json.load(f)

def main(sgo_path):

    d = np.load(sgo_path)

    labels = [goids_to_names[n] for n in d['feature_labels']]

    for i,l in enumerate(labels):
        print("[%d] %s" % (i,l))
        
if __name__ == "__main__":
    main(sys.argv[1])