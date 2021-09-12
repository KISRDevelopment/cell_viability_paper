import models.cv 
import tasks.yeast_smf_alt
import utils.bin_outcomes
import utils.split_train_test
import utils.cv_simple

gpath = "../generated-data/ppc_yeast"
smf_task_path = "../tmp/task_yeast_smf_30_strict"
tasks.yeast_smf_alt.main(gpath, 30, smf_task_path)
utils.bin_outcomes.main(smf_task_path, {
    "is_lethal" : lambda bins: bins == 0,
    "is_viable" : lambda bins: bins > 0,
}, smf_task_path)
utils.split_train_test.main(smf_task_path, 0.2, 
    "../tmp/task_yeast_smf_30_strict_train",
    "../tmp/task_yeast_smf_30_strict_test")
utils.cv_simple.main("../tmp/task_yeast_smf_30_strict_train", 10, 5, 0.2)
# utils.cv_simple.main("../generated-data/task_yeast_smf_30", 10, 5, 0.2, "../generated-data/splits/task_yeast_smf_30_full")



models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
    "../results/task_yeast_smf_30_strict/orm", 
    num_processes = 20,
    target_col = "bin",
    task_path="../tmp/task_yeast_smf_30_strict_train",
    splits_path="../generated-data/splits/task_yeast_smf_30_strict_train_10reps_5folds_0.20valid.npz")

