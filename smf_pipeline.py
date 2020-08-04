import os 
import subprocess 
import sys 
import json 
import shlex

import utils.make_smf_single_feature_sweeps
import utils.make_smf_model_combs
import models.multiple_cv
import models.cv 

# create Model Combinations
utils.make_smf_model_combs.main()
utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "redundancy", "../tmp/model_cfgs/yeast_smf")
utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "topology", "../tmp/model_cfgs/yeast_smf")

# # execute CV on all model combinations
models.multiple_cv.main("models.smf_nn", "../tmp/model_cfgs/yeast_smf", "../results/task_yeast_smf_30", 6, exclude=lambda s: 'redundancy' not in s)

# execute CV on refined model
models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_refined_model.json", "../results/task_yeast_smf_30/refined")

# execute CV on null models
models.cv.main("models.null_model", "cfgs/models/yeast_smf_full_model.json", "../results/task_yeast_smf_30/null")
models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_full_model.json", "../results/task_yeast_smf_30/null_scarmbled", scramble=True)

# execute CV on OR model
models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", "../results/task_yeast_smf_30/orm")

# pombe
models.cv.main("models.null_model", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/null_scarmbled", scramble=True)
models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", "../results/task_pombe_smf/orm")

# human
models.cv.main("models.null_model", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf/null_scarmbled", scramble=True)
models.cv.main("models.smf_nn", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", "../results/task_human_smf/orm")

# dro
models.cv.main("models.null_model", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/null_scarmbled", scramble=True)
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", "../results/task_dro_smf/orm")

# dro organsim viability
models.cv.main("models.null_model", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/null")
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/null_scarmbled", scramble=True)
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_org_orm.json", "../results/task_dro_smf_org/orm")

