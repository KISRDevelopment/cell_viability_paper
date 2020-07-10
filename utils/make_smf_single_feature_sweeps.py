import json 
import os 
import sys 
import models.feature_loader as feature_loader

def main(base_cfg_path, feature_name, output_dir):
    
    with open(base_cfg_path, 'r') as f:
        base_cfg = json.load(f)
    
    # get features to sweep
    features_to_sweep = None
    target_spec = None 
    for spec in base_cfg['spec']:
        if spec['name'] == feature_name:
            _, feature_labels = feature_loader.read_paths(spec['paths'])
            features_to_sweep = feature_labels
            target_spec = spec 
            break 
    
    print("To sweep: ")
    print(features_to_sweep)

    main_config_name = os.path.basename(base_cfg_path).replace(".json", "")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fs in features_to_sweep:
        config_name = main_config_name.replace(feature_name, "%s--%s" % (feature_name, fs))
        
        target_spec['selected_features'] = [fs]

        with open(os.path.join(output_dir, "%s.json" % config_name), 'w') as f:
            json.dump(base_cfg, f, indent=4)
if __name__ == "__main__":
    base_cfg_path = sys.argv[1]
    feature_name = sys.argv[2]
    output_dir = sys.argv[3]

    main(base_cfg_path, feature_name, output_dir)
