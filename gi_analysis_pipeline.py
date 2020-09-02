import os 
import subprocess 
import sys 
import json 
import shlex

import analysis.tbl_model_comp
import analysis.fig_cv_performance

analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid", "../results/task_yeast_gi_hybrid/model_comp.xlsx")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_hybrid.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_costanzo.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_gi.json")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_hybrid_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_gi_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_gi.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_gi.json")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_pombe.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_human.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_dro.json")
