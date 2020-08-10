import sys 
import numpy as np 
import os 
import subprocess 
import shutil 
import json 
import psutil
import time 

SHOW_OUTPUT = False 
OVERWRITE_PROMPT = False

MEMORY_THRES_PERC = 70
CPU_THRES_PERC = 70
SLEEP_TIME_SEC = 20

def main(script_name, cfg_path, output_dir, num_processes=6, scramble=False):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    if scramble:
        cfg_path = "../tmp/scrambled_cfg"
        cfg['scramble'] = True
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)
    

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

    incr = num_processes
    i = 0
    total = reps*folds
    while i < total:
        subset = split_ids[i:(i+incr)]
        i += incr 

        processes = []
        for j in subset:
            rep = j // folds 
            fold = j % folds 

            output_path = os.path.join(output_dir, 'run_%d_%d.npz' % (rep, fold))
            # if os.path.exists(output_path):
            #     print("Skipping %s" % output_path)
            #     continue 
            
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
        

        # hack sleep for a while to give processes time to initialize
        time.sleep(SLEEP_TIME_SEC)
        mperc = psutil.virtual_memory().percent
        cperc = psutil.cpu_percent()
        print("CPU usage: %0.2f, Memory use: %0.2f" % (cperc, mperc))
        if mperc <= MEMORY_THRES_PERC and cperc <= CPU_THRES_PERC:
            incr += 1 # add one processor
            print("Adding one processor: %d" % incr)
        
        for p in processes:
            p.wait()
            finished += 1
            print("Finished %d" % finished)

        
if __name__ == "__main__":
    script_name = sys.argv[1]
    cfg_path = sys.argv[2]
    output_dir = sys.argv[3]
    num_processes = int(sys.argv[4])

    main(script_name, cfg_path, output_dir, num_processes)