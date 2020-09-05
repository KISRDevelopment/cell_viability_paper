import os 
import subprocess 
import sys 
import json 
import shlex

import analysis.grouped_analysis
import analysis.fig_group_interactions_hist 

task_path = '../generated-data/task_yeast_gi_hybrid'
analysis.grouped_analysis.main(task_path, 'complexes', '../generated-data/complexes.xlsx')
analysis.grouped_analysis.main(task_path, 'pathways', '../generated-data/pathways.xlsx')

analysis.fig_group_interactions_hist.main('../generated-data/complexes.xlsx', '../figures/complexes_hist.png')
analysis.fig_group_interactions_hist.main('../generated-data/pathways.xlsx', '../figures/pathways_hist.png')
