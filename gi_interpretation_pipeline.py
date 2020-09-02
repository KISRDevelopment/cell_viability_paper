import os 
import subprocess 
import sys 
import json 
import shlex
import glob 
import os 
import numpy as np 
import models.cv 
import scipy.stats as stats

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/gi_interpretation/yeast_mn", interpreation=True, 
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
    "../results/gi_interpretation/pombe_mn", interpreation=True, 
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
    "../results/gi_interpretation/human_mn", interpreation=True, num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
    "../results/gi_interpretation/dro_mn", interpreation=True, num_processes=20, epochs=50)
