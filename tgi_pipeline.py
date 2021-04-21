import os 
import subprocess 
import sys 
import json 
import shlex

import models.cv 

import analysis.tbl_model_comp
import utils.make_cfgs


models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_full_model.json", 
    "../results/task_yeast_tgi/full", 
    num_processes=5)

models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
    "../results/task_yeast_tgi/refined", 
    num_processes=20)

models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
    "../results/task_yeast_tgi/mn", 
    num_processes=20)

models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
    "../results/task_yeast_tgi/null", 
    num_processes=20, scramble=True)
