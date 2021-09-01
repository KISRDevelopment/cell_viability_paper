import sys 
import numpy as np 
import os 
import subprocess 
import shutil 
import json 

SHOW_OUTPUT = False 
OVERWRITE_PROMPT = False

def main(script_name, cfg_path, output_dir, num_processes=8, interpreation=False, scramble=False, remove_specs=[], **kwargs):
    orig_cfg_path = cfg_path

    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
    if num_processes > 1 and (CUDA_VISIBLE_DEVICES != "-1"):
        print("Cannot run CV without setting CUDA_VISIBLE_DEVICES=-1")
        return 
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    

    if interpreation:
        cfg['bootstrap_training'] = True 
        cfg['early_stopping'] = False 
        cfg['train_on_full_dataset'] = True
        cfg['train_model'] = True 
        cfg_path = "../tmp/override_cfg"

    if scramble:
        cfg_path = "../tmp/override_cfg"
        cfg['scramble'] = True

    if len(remove_specs) > 0:
        cfg['spec'] = [s for s in cfg['spec'] if s['name'] not in remove_specs]
        cfg_path = "../tmp/override_cfg"

    if kwargs:
        cfg_path = "../tmp/override_cfg"
        cfg.update(kwargs)
    
    print(cfg)
        
    if orig_cfg_path != cfg_path:
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)

    train_test_path = cfg['splits_path']

    d = np.load(train_test_path)
    reps, folds,_ = d['train_sets'].shape
    print("REPS: %d, FOLDS: %d" % (reps, folds))
    if OVERWRITE_PROMPT and os.path.exists(output_dir):
        print("Ok to overwrite %s? " % output_dir)
        r = input().strip().lower()
        if r not in ['yes', 'y']:
            return 
        shutil.rmtree(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    finished = 0

    work_items = []
    for i in range(reps*folds):
        rep = i // folds 
        fold = i % folds 
        output_path = os.path.join(output_dir, 'run_%d_%d.npz' % (rep, fold))
        if os.path.exists(output_path):
            print("Skipping %s" % output_path)
            continue 
        work_items.append(i)
    
    for i in range(0, len(work_items), num_processes):
        subset = work_items[i:(i+num_processes)]
        processes = []
        for j in subset:
            rep = j // folds 
            fold = j % folds 

            output_path = os.path.join(output_dir, 'run_%d_%d.npz' % (rep, fold))
            
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
    script_name = sys.argv[1]
    cfg_path = sys.argv[2]
    output_dir = sys.argv[3]
    num_processes = int(sys.argv[4])

    main(script_name, cfg_path, output_dir, num_processes)