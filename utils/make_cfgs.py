import json 
import os 
import sys 
import models.feature_loader as feature_loader
import copy 

def main(cfg_path, output_dir):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_cfg = copy.deepcopy(cfg)
    del base_cfg['specs']

    for config_name, spec in cfg['specs']:
        
        base_cfg['spec'] = spec

        with open(os.path.join(output_dir, "%s.json" % config_name), 'w') as f:
            json.dump(base_cfg, f, indent=4)

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    output_dir = sys.argv[2]

    main(cfg_path, output_dir)
