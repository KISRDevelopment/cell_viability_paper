#
# Make Supplementary Table 1: Feature Selection Results
#
import numpy as np
import pandas as pd
import analysis.tbl_model_comp

analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid_fs", 
     "../tables/table3.xlsx", analysis.tbl_model_comp.GI_LABELS)
