import json 
import subprocess
import os 
import glob 
import sys 

def main(script_name, cfgs_dir, output_dir, n_processors=6, exclude=lambda s: False):
    
    files = glob.glob(cfgs_dir + "/*.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in files:
        config_name = os.path.basename(file).replace('.json', '')
        if exclude(config_name):
            continue
        print(config_name)

        subprocess.call(['python', "-m", "models.cv",
            script_name, file, 
            os.path.join(output_dir, config_name),
            str(n_processors)
        ])

if __name__ == "__main__":
    script_name = sys.argv[1]
    cfgs_dir = sys.argv[2]
    output_dir = sys.argv[3]
    main(script_name, cfgs_dir, output_dir)
