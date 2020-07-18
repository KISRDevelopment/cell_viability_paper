import numpy as np 
import sys 

def get_union(go_files):

    all_go_ids = set()
    for go_path in go_files:

        d = np.load(go_path, allow_pickle=True)
        print("%s: %d" % (go_path, d['F'].shape[1]))
        go_ids = d['feature_labels']

        for goid in go_ids:
            all_go_ids.add(goid)
    
    print("Union length: %d" % len(all_go_ids))

    return all_go_ids

if __name__ == "__main__":
    go_files = sys.argv[1:]
    print(get_union(go_files))
