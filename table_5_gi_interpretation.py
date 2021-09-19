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


gi_cfg_nonbinary = load_cfg('cfgs/fig_interpretation/gi_nonbinary.json')
analysis.fig_interpretation.main(gi_cfg_nonbinary, save_plots=False)

gi_cfg_binary = load_cfg('cfgs/fig_interpretation/gi_binary.json')
analysis.fig_interpretation.main(gi_cfg_binary, save_plots=False)

# merge everything into one file
writer = pd.ExcelWriter('../tables/table5.xlsx')
for cfg, name in zip([gi_cfg_nonbinary, gi_cfg_binary], ['GI 4-way', 'GI Binary']):

    results_file = cfg['output_path'] + '.xlsx'
    df = pd.read_excel(results_file, index_col=0, header=[0,1])
    df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
    print(df.columns)
    df.to_excel(writer, sheet_name=name, index=True)
writer.save()
