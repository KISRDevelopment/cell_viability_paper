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

items = feature_groups.keys()
    
all_group_combs = [comb for r in range(1, len(items)+1) for comb in itertools.combinations(items, r)]
#print(all_group_combs)
#print(len(all_group_combs))

combinations = []
for group_comb in all_group_combs:
    compartments = [comp for group in group_comb for comp in feature_groups[group]]
    combinations.append((group_comb, compartments))
    
with open('cfgs/models/yeast_smf_full_model.json', 'r') as f:
    base_cfg = json.load(f)

names_to_comps = dict([(e['name'], e) for e in base_cfg['spec']])

if not os.path.exists('../tmp/model_cfgs/yeast_smf'):
    os.makedirs('../tmp/model_cfgs/yeast_smf')
for group_comb, comb in combinations:
    sub_spec = [ names_to_comps[comp_name] for comp_name in comb ]
    base_cfg['spec'] = sub_spec

    cfg_name = '_'.join(group_comb)
    base_cfg['output_path'] = '../results/yeast_smf/%s' % cfg_name
    
    with open('../tmp/model_cfgs/yeast_smf/%s.json' % cfg_name, 'w') as f:
        json.dump(base_cfg, f, indent=4)

