import sys 
import numpy as np 
import os 
import subprocess 
import shutil 
import json 

NUM_PROCESSES = 1
SHOW_OUTPUT = True 
OVERWRITE_PROMPT = False

def main():
    script_name = sys.argv[1]
    cfg_path = sys.argv[2]
    output_dir = sys.argv[3]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    train_test_path = cfg['splits_path']

    d = np.load(train_test_path)
    reps, folds,_ = d['train_sets'].shape
    
    if OVERWRITE_PROMPT and os.path.exists(output_dir):
        print("Ok to overwrite %s? " % output_dir)
        r = input().strip().lower()
        if r not in ['yes', 'y']:
            return 
        shutil.rmtree(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_ids = np.arange(reps * folds)
    finished = 0
    for i in range(0, reps*folds, NUM_PROCESSES):
        subset = split_ids[i:(i+NUM_PROCESSES)]
        processes = []
        for j in subset:
            rep = j // folds 
            fold = j % folds 

            output_path = os.path.join(output_dir, 'run_%d_%d.npz' % (rep, fold))
            if os.path.exists(output_path):
                print("Skipping %s" % output_path)
                continue 
            
            print("Spawning rep %d fold %d" % (rep,fold))
            kwargs = {}
            if not SHOW_OUTPUT:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL

            processes.append(subprocess.Popen(['python', "-m", 
                script_name, cfg_path, 
                str(rep), str(fold), 
                output_path
            ],  **kwargs))
        
        for p in processes:
            p.wait()
            finished += 1
            print("Finished %d" % finished)

        
if __name__ == "__main__":
    main()