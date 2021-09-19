#
# Make Supplementary Table 1: Feature Selection Results
#
import numpy as np
import pandas as pd
import analysis.tbl_model_comp

analysis.tbl_model_comp.main("../results/yeast_smf_30_fs", "../tables/table1.xlsx", analysis.tbl_model_comp.SMF_LABELS)

