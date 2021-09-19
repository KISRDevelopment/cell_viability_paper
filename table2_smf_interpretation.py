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

smf_cfg = load_cfg("cfgs/fig_interpretation/smf_mn.json")
analysis.fig_interpretation.main(smf_cfg, save_plots=False)

smf_cfg_binary = load_cfg('cfgs/fig_interpretation/smf_binary.json')
analysis.fig_interpretation.main(smf_cfg_binary, save_plots=False)

smf_ca_ma = load_cfg('cfgs/fig_interpretation/smf_ca_ma_v.json')
analysis.fig_interpretation.main(smf_ca_ma, save_plots=False)

# merge everything into one file
writer = pd.ExcelWriter('../tables/table2.xlsx')
for cfg, name in zip([smf_cfg, smf_cfg_binary, smf_ca_ma], ['SMF', 'SMF Binary', 'CA vs MO']):
    results_file = cfg['output_path'] + '.xlsx'
    df = pd.read_excel(results_file, index_col=0, header=[0,1])
    df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
    df.to_excel(writer, sheet_name=name, index=True)
writer.save()
