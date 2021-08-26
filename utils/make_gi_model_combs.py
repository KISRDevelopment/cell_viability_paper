import json 
import subprocess
import os 
import json 
import itertools

feature_groups = {
    "topology" : ["topology"],
    "go" : ["go"],
    "redundancy" : ["redundancy"],
    "phosphorylation" : ["phosphotase", "kinase", "transcription"],
    "abundance-and-loc" : ["abundance_wt3", "abundance_rap", "abundance_hu", "localization_wt3", "localization_rap", "localization_hu"],
    "pairwise" : ["pairwise", "pairwise_comm"],
    "smf" : ["smf"]
}

PAIRWISE_CONST = {
    "name": "pairwise_const",
    "group": "pairwise_const",
    "hidden_activation": "tanh",
    "layer_sizes": [
        1
    ],
    "type": "ff",
    "pairwise": True,
    "paths": [
        {
            "normalize": False,
            "path": "../generated-data/pairwise_features/ppc_yeast_const.npy"
        }
    ]
}

SINGLE_CONST = {
    "name": "const",
    "group": "const",
    "hidden_activation": "tanh",
    "layer_sizes": [
        1
    ],
    "type": "ff",
    "pairwise": False,
    "paths": [
        "../generated-data/features/ppc_yeast_const.npz"
    ]
}

def main(base_cfg, output_dir):
        
    items = feature_groups.keys()
        
    all_group_combs = [comb for r in range(1, len(items)+1) for comb in itertools.combinations(items, r)]

    combinations = []
    for group_comb in all_group_combs:
        compartments = [comp for group in group_comb for comp in feature_groups[group]]
        combinations.append((group_comb, compartments))
    
    print("Combinations: %d" % len(combinations))
    names_to_comps = dict([(e['name'], e) for e in base_cfg['spec']])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for group_comb, comb in combinations:
        sub_spec = [ names_to_comps[comp_name] for comp_name in comb ]
        has_pairwise = any([ names_to_comps[comp_name]['pairwise'] and not names_to_comps[comp_name].get('triplet',False) for comp_name in comb ])
        has_single = any([(not names_to_comps[comp_name]['pairwise']) for comp_name in comb])

        if not has_pairwise:
            sub_spec.append(PAIRWISE_CONST)
        if not has_single:
            sub_spec.append(SINGLE_CONST)
        
        base_cfg['spec'] = sub_spec

        cfg_name = '~'.join(group_comb)
        with open('%s/%s.json' % (output_dir,cfg_name), 'w') as f:
            json.dump(base_cfg, f, indent=4)

if __name__ == "__main__":
    cfg_path = sys.argv[1]

    with open(cfg_path, 'r') as f:
        base_cfg = json.load(cfg_path)
    main(base_cfg)
    