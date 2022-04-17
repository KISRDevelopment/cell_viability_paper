import os 
import utils.split_standard

cfgs = [

    # {
    #     "dataset_path" : "../generated-data/task_yeast_smf_30",
    #     "function" : utils.split_standard.main,
    #     "reps" : 10,
    #     "folds" : 5,
    #     "dev_test" : True,
    #     "test_n" : 5,
    #     "output_path" : "../generated-data/splits/task_yeast_smf_30_dev_test"
    # },
    # {
    #     "dataset_path" : "../generated-data/task_yeast_smf_30",
    #     "function" : utils.split_standard.main,
    #     "reps" : 10,
    #     "folds" : 5,
    #     "dev_test" : False,
    #     "test_n" : 5,
    #     "output_path" : "../generated-data/splits/task_yeast_smf_30"
    # },
    # {
    #     "dataset_path" : "../generated-data/task_pombe_smf",
    #     "function" : utils.split_standard.main,
    #     "reps" : 10,
    #     "folds" : 5,
    #     "dev_test" : False,
    #     "test_n" : 5,
    #     "output_path" : "../generated-data/splits/task_pombe_smf"
    # },
    {
        "dataset_path" : "../generated-data/task_human_smf",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_human_smf"
    },
    {
        "dataset_path" : "../generated-data/task_human_smf_ca_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_human_smf_ca_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_human_smf_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_human_smf_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_dro_smf"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf_ca_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_dro_smf_ca_mo_v"
    },
    {
        "dataset_path" : "../generated-data/task_dro_smf_mo_v",
        "function" : utils.split_standard.main,
        "reps" : 10,
        "folds" : 5,
        "dev_test" : False,
        "test_n" : 5,
        "output_path" : "../generated-data/splits/task_dro_smf_mo_v"
    },
]

def main():
    os.makedirs("../generated-data/splits", exist_ok=True)
    for cfg in cfgs:
        print(cfg['dataset_path'], '->', cfg['output_path'])
        cfg['function'](**cfg)


if __name__ == "__main__":
    main()
