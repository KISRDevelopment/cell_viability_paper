# Predicting and explaining the impact of genetic disruptions and interactions on cell and organismal viability 


## File Description

### Dataset Curation

The following files are responsible for extracting features and tasks from raw bioinformatic data.

- `create_ppc.py` creates the protein-protein interaction networks for the budding yeast, fission yeast, human, and fruit fly. 
- `create_tasks.py` generates the single-, double-, and triple-mutant tasks studied in the paper. It assumes that `create_ppc.py` has already been executed.
- `create_features.py` creates the single and pairwise gene features for all four organisms. This requires the `owltool` application from geneontology.org to be present in `../tools`, and requires an NCBI-Blast+ installation (on Ubuntu, this can be installed via `sudo apt install ncbi-blast+`).
- `create_datasets.py` combines features and tasks into one csv file, for each task. GI and Triple GI tasks only include the pairwise features as it would be too much to include the features of individual genes.
- `create_pseudo_triplets_task.py` creates randomly sampled pseudo triplets within- and across-complexes.

This repository includes the code to process source data and run all experiments and analyses in the paper. 

The code relies on Python 3.6+ and requires Tensorflow 2+ for modeling. 

## Required files
- `../generated-data/go_ids_to_names.json`

## Figures need fixing
- Sum LID in pairwise and triple
- 
[**Live GI Prediction Tool**](http://ssdd.kisr.edu.kw/gi_pred/static/search_gi.html)

(link will change once we get assigned a permanant address)
