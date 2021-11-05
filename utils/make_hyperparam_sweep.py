import json 
import os 
import sys 
import itertools 

def main(base_cfg_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(base_cfg_path, 'r') as f:
        cfg = json.load(f)
    
    keys = list(cfg['hyperparams'].keys())
    vals = [cfg['hyperparams'][k] for k in keys]

    combinations = [dict(zip(keys, comb)) for comb in itertools.product(*vals)]

    for i, combination in enumerate(combinations):

        cfg.update(combination)

        for fspec in cfg['spec']:
            fspec['layer_sizes'] = combination['layer_sizes']

        print(json.dumps(cfg, indent=4))

        with open(os.path.join(output_dir, "%d.json" % i), 'w') as f:
            json.dump(cfg, f, indent=4)

if __name__ == "__main__":
    base_cfg_path = sys.argv[1]
    output_dir = sys.argv[2]

    main(base_cfg_path, output_dir)
