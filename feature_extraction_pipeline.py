import os 
import subprocess 
import sys 
import json 
import shlex

import utils.map_go_ids_to_names
import utils.map_entrez_fbgn
import feature_preprocessing.topology
import feature_preprocessing.yeast_abundance
import feature_preprocessing.yeast_localization
import feature_preprocessing.yeast_phosphotase
import feature_preprocessing.redundancy
import feature_preprocessing.yeast_sgo
import feature_preprocessing.yeast_transcription
import feature_preprocessing.pairwise_comms_sparse
import feature_preprocessing.pairwise_ig
import feature_preprocessing.pairwise_shortest_path_len
import feature_preprocessing.smf
import feature_preprocessing.yeast_smf
import ppc_creation.find_comms
import feature_preprocessing.common_sgo
import feature_preprocessing.const 
import feature_preprocessing.sgo 

if not os.path.exists('../generated-data/features'):
    os.makedirs('../generated-data/features')

utils.map_go_ids_to_names.main()
utils.map_entrez_fbgn.main()

# yeast features
gpath = "../generated-data/ppc_yeast"
feature_preprocessing.yeast_abundance.main(gpath)
feature_preprocessing.yeast_localization.main(gpath)
feature_preprocessing.yeast_localization.main(gpath, flatten=True)
feature_preprocessing.yeast_phosphotase.main(gpath, "../data-sources/yeast/kinase.txt")
feature_preprocessing.yeast_phosphotase.main(gpath, "../data-sources/yeast/phosphotase.txt")
feature_preprocessing.yeast_sgo.main(gpath)
feature_preprocessing.yeast_transcription.main(gpath)
feature_preprocessing.topology.main(gpath)
feature_preprocessing.redundancy.main("yeast", gpath)
feature_preprocessing.pairwise_shortest_path_len.main(gpath) 
feature_preprocessing.pairwise_ig.main(gpath_gml, "adhesion", 16) # long running
feature_preprocessing.pairwise_ig.main(gpath_gml, "cohesion", 16) # long running
feature_preprocessing.pairwise_ig.main(gpath + '.gml', "adjacent", 16)
feature_preprocessing.pairwise_ig.main(gpath + '.gml', "mutual_neighbors", 16)
feature_preprocessing.const.main(gpath)
ppc_creation.find_comms.main(gpath + '.gml', 5)
feature_preprocessing.pairwise_comms_sparse.main(gpath, '../generated-data/communities/%s_5steps.json' % os.path.basename(gpath))
feature_preprocessing.yeast_smf.main(gpath)

# pombe features
gpath = "../generated-data/ppc_pombe"
subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/pombe/pombase.gaf --map2slim --subset goslim_generic --write-gaf ../tmp/pombase.sgo.gaf"))
feature_preprocessing.sgo.main(gpath, "../tmp/pombase.sgo.gaf", 1)
feature_preprocessing.topology.main(gpath, ['lid'])
feature_preprocessing.redundancy.main("pombe", gpath)
ppc_creation.find_comms.main(gpath + '.gml', 5)
feature_preprocessing.pairwise_comms_sparse.main(gpath, '../generated-data/communities/%s_5steps.json' % os.path.basename(gpath))
feature_preprocessing.smf.main(gpath, "../generated-data/task_pombe_smf")

# human features
gpath = "../generated-data/ppc_human"
subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/human/goa_human.gaf --map2slim --subset goslim_generic --write-gaf ../tmp/human.sgo.gaf"))
feature_preprocessing.sgo.main(gpath, "../tmp/human.sgo.gaf", 2)
feature_preprocessing.topology.main(gpath, ['lid'])
feature_preprocessing.redundancy.main("human", gpath)
ppc_creation.find_comms.main(gpath + '.gml', 5)
feature_preprocessing.pairwise_comms_sparse.main(gpath, '../generated-data/communities/%s_5steps.json' % os.path.basename(gpath))
feature_preprocessing.smf.main(gpath, "../generated-data/task_human_smf")

# dmel features
gpath = "../generated-data/ppc_dro"
subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/dro/fb.gaf --map2slim --subset goslim_drosophila --write-gaf ../tmp/dro.sgo.gaf"))
feature_preprocessing.sgo.main(gpath, "../tmp/dro.sgo.gaf", 1)
feature_preprocessing.topology.main(gpath, ['lid'])
feature_preprocessing.redundancy.main("dro", gpath)
ppc_creation.find_comms.main(gpath + '.gml', 5)
feature_preprocessing.pairwise_comms_sparse.main(gpath, '../generated-data/communities/%s_5steps.json' % os.path.basename(gpath))
feature_preprocessing.smf.main(gpath, "../generated-data/task_dro_smf")

# extract common sgo
gpaths = ["../generated-data/ppc_yeast", 
    "../generated-data/ppc_pombe", 
    "../generated-data/ppc_human", 
    "../generated-data/ppc_dro"]
sgo_files = ["../generated-data/features/%s_sgo.npz" % os.path.basename(gpath) for gpath in gpaths]

for sgo_file, gpath in zip(sgo_files, gpaths):
    target_files = [sgo_file] + [s for s in sgo_files if s != sgo_files]
    feature_preprocessing.common_sgo.main(gpath, target_files)

