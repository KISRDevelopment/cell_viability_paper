import os 
import subprocess 
import sys 
import json 
import shlex

import feature_preprocessing.pairwise_comms
import feature_preprocessing.pairwise_ig
import feature_preprocessing.pairwise_shortest_path_len
import ppc_creation.find_comms


gpath = "../generated-data/ppc_yeast"
gpath_gml = gpath + '.gml'
comms_path = "../generated-data/communities/ppc_yeast_5steps.json"

if not os.path.exists('../generated-data/features'):
    os.makedirs('../generated-data/features')
if not os.path.exists('../generated-data/pairwise_features'):
    os.makedirs('../generated-data/pairwise_features')
if not os.path.exists('../generated-data/communities'):
    os.makedirs('../generated-data/communities')
if not os.path.exists('../generated-data/splits'):
    os.makedirs('../generated-data/splits')
if not os.path.exists('../generated-data/targets'):
    os.makedirs('../generated-data/targets')

# Find comms
ppc_creation.find_comms.main(gpath_gml, 5)

# create features
feature_preprocessing.pairwise_shortest_path_len.main(gpath)
feature_preprocessing.pairwise_comms.main(gpath, comm_path)
feature_preprocessing.pairwise_ig.main(gpath_gml, "adhesion", 16)
feature_preprocessing.pairwise_ig.main(gpath_gml, "cohesion", 16)
feature_preprocessing.pairwise_ig.main(gpath_gml, "adjacent", 16)
feature_preprocessing.pairwise_ig.main(gpath_gml, "mutual_neighbors", 16)
feature_preprocessing.yeast_smf.main(gpath)
feature_preprocessing.const.main(gpath)
