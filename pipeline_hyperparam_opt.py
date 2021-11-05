import utils.make_hyperparam_sweep
import models.multiple_cv
import analysis.tbl_model_comp_hyperparam

# utils.make_hyperparam_sweep.main("cfgs/models/yeast_smf_full_model.json", "../tmp/cfgs/hyperparam_smf_full")
# models.multiple_cv.main("models.smf_nn", 
#     "../tmp/cfgs/hyperparam_smf_full", 
#     "../results/hyperparam_smf_full", n_processors=8, n_runs=50)

# analysis.tbl_model_comp_hyperparam.main("../results/hyperparam_smf_full", 
#     "../tmp/hyperparam_smf_full.xlsx", analysis.tbl_model_comp_hyperparam.SMF_LABELS)

utils.make_hyperparam_sweep.main("cfgs/models/yeast_gi_full_model.json", "../tmp/cfgs/hyperparam_gi_full")
models.multiple_cv.main("models.gi_nn", 
    "../tmp/cfgs/hyperparam_gi_full", 
    "../results/hyperparam_gi_full", n_processors=2, n_runs=40)

analysis.tbl_model_comp_hyperparam.main("../results/hyperparam_gi_full", 
    "../tmp/hyperparam_gi_full.xlsx", analysis.tbl_model_comp_hyperparam.GI_LABELS)

