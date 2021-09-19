#
# Make Supplementary Table 2: SMF Interpretation Results
#
import numpy as np
import pandas as pd
import analysis.fig_interpretation
import json 

def load_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)

cfg =load_cfg("cfgs/fig_interpretation/tgi.json")
cfg['output_path'] = '../tables/table7_tgi_interpretation.json'
analysis.fig_interpretation.main(cfg, save_plots=False)

