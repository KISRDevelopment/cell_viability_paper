#
# Make Supplementary Table 6: Feature Selection Results for TGI
#
import numpy as np
import pandas as pd
import analysis.tbl_model_comp

analysis.tbl_model_comp.main("../results/task_yeast_tgi_fs", 
    "../tables/table6_tgi_feature_selection.xlsx", ['Interacting', 'Neutral'])
