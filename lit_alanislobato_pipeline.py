import numpy as np 
import pandas as pd 
import os 
import utils.cv_simple_repeated_random
import models.cv 
import feature_preprocessing.pairwise_acdd

# generate the ACDD feature, which was the highest performing in Alanis-Lobato et al 2013 
#feature_preprocessing.pairwise_acdd.main("../generated-data/ppc_yeast")


models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_acdd.json", 
    "../results/lit_alanislobato2013/acdd", 
    num_processes=20,
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_not_negative")

