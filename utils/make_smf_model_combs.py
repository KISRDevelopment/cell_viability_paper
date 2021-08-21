import json 
import subprocess
import os 
import json 
import itertools

feature_groups = {
    "topology" : ["topology"],
    "go" : ["go"],
    "redundancy" : ["redundancy"],
    "phosphorylation" : ["phosphotase", "kinase"],
    "transcription" : ["transcription"],
    "abundance" : ["abundance_wt3", "abundance_rap", "abundance_hu"],
    "localization" : ["localization_wt3", "localization_rap", "localization_hu"]
}

def main(base_cfg_path, comb_output_path, cfg_output_path):
        
    items = feature_groups.keys()
        
    all_group_combs = [comb for r in range(1, len(items)+1) for comb in itertools.combinations(items, r)]
    print("Combinations: %d" % len(all_group_combs))

    combinations = []
    for group_comb in all_group_combs:
        compartments = [comp for group in group_comb for comp in feature_groups[group]]
        combinations.append((group_comb, compartments))
        
    with open(base_cfg_path, 'r') as f:
        base_cfg = json.load(f)

    names_to_comps = dict([(e['name'], e) for e in base_cfg['spec']])

    if not os.path.exists(cfg_output_path):
        os.makedirs(cfg_output_path)
    
    for group_comb, comb in combinations:
        sub_spec = [ names_to_comps[comp_name] for comp_name in comb ]
        base_cfg['spec'] = sub_spec

        cfg_name = '~'.join(group_comb)
        base_cfg['output_path'] = '%s/%s' % (comb_output_path, cfg_name)
        
        with open('%s/%s.json' % (cfg_output_path, cfg_name), 'w') as f:
            json.dump(base_cfg, f, indent=4)

if __name__ == "__main__":
    main()
    