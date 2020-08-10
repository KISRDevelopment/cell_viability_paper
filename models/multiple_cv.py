import json 
import subprocess
import os 
import glob 
import sys 
import time 
import sys, select 

CONTINUE_PROMPT_TIMEOUT_SECS = 5

def main(script_name, cfgs_dir, output_dir, n_processors=6, exclude=lambda s: False, dynamic_dispatch=False, n_runs=40):
    
    files = glob.glob(cfgs_dir + "/*.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in files:
        config_name = os.path.basename(file).replace('.json', '')
        if exclude(config_name):
            continue
        
        config_output_dir = os.path.join(output_dir, config_name)
        if os.path.exists(config_output_dir):
            dir_files = glob.glob(os.path.join(config_output_dir, '*.npz'))
            if len(dir_files) == n_runs:
                print("Ignoring %s" % config_name)
                continue 
        print(config_name)

        module = "models.cv_dynamic_dispatch" if dynamic_dispatch else "models.cv"
        subprocess.call(['python', "-m", module,
            script_name, file, 
            config_output_dir,
            str(n_processors)
        ])
        
        print("Stop? ")
        i, o, e = select.select([sys.stdin], [], [], CONTINUE_PROMPT_TIMEOUT_SECS)
        if i: 
            break 
            print("Stopped.")
        
if __name__ == "__main__":
    script_name = sys.argv[1]
    cfgs_dir = sys.argv[2]
    output_dir = sys.argv[3]
    main(script_name, cfgs_dir, output_dir)
