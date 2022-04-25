import os 
import utils.split_standard
import utils.split_gi

cfgs = [

    {
        "dataset_path" : "../generated-data/task_yeast_smf_30",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : True,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_smf_30_dev_test"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_smf_30",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_smf_30"
    },
    {
        "dataset_path" : "../generated-data/task_pombe_smf",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_pombe_smf"
    },
    {
        "dataset_path" : "../generated-data/task_human_smf",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_human_smf"
    },
    {
        "dataset_path" : "../generated-data/task_human_smf_ca_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_human_smf_ca_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_human_smf_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_human_smf_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_dro_smf"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf_ca_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_dro_smf_ca_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_dro_smf_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_gi_costanzo",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_gi_costanzo"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_gi_hybrid",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : True,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_gi_hybrid_dev_test"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_gi_hybrid",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_gi_hybrid"
    },
    {
        "dataset_path" : "../generated-data/task_pombe_gi",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_pombe_gi"
    },
    {
        "dataset_path" : "../generated-data/task_human_gi",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_human_gi"
    },
    {
        "dataset_path" : "../generated-data/task_dro_gi",
        "function" : utils.split_gi.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_dro_gi"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_tgi",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : True,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_tgi_dev_test"
    },
    {
        "dataset_path" : "../generated-data/task_yeast_tgi",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 4,
        "valid_p" : 0.2,
        "dev_test" : False,
        "test_p" : 0.2,
        "output_path" : "../generated-data/splits/task_yeast_tgi"
    },
]

def main():
    os.makedirs("../generated-data/splits", exist_ok=True)
    for cfg in cfgs:
        print(cfg['dataset_path'], '->', cfg['output_path'])
        cfg['function'](**cfg)


if __name__ == "__main__":
    main()
