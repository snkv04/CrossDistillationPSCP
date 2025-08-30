# Leveraging orientation-aware graph neural networks for protein side-chain prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16937575.svg)](https://doi.org/10.5281/zenodo.16937575)

This repository contains the code associated with the paper "On overcoming PDB data scarcity with AlphaFoldDB for protein side-chain packing" by Sriniketh Vangaru and Debswapna Bhattacharya. ([PDF](./CrossDistillationForPSCPPaper.pdf))

### Abstract:

"Protein side-chain packing (PSCP) is the problem of predicting the coordinates of side-chain atoms given fixed backbone coordinates, and it is useful across various tasks in the field of structural biological modeling. While traditional PSCP methods are primarily trained on experimentally-determined structures from the Protein Data Bank (PDB), AlphaFold—a paradigm-shifting, machine learning-based, protein structure prediction tool released by DeepMind—has made available AlphaFoldDB, a database of high-quality synthetic protein structures which massively expands the structural coverage space from the PDB by multiple orders of magnitude. Herein, we aimed to determine whether PSCP methods could benefit from substituting their training data with AlphaFoldDB structures. Using a recent protein encoder named the orientation-aware graph neural network as our testing framework, we find that the high-confidence predicted protein structures from AlphaFoldDB are not suitable direct replacements for native chains in side-chain modeling, with such a replacement causing significant degradations in performance across various evaluation metrics. We also explore an approach of cross-distilling knowledge from AlphaFold's network by using both datasets simultaneously for our model to learn, which displays better results than using either dataset individually, regardless of the backbone type supplied during inference."

## Usage

To set up the conda environment:

```
conda env create -f conda_envs/dw6.yml
conda activate dw6
```

Configurations such as the path to the directory containing the dataset of PDB files to train on can be set in `configs/svp_gnn.yml`. To train the model:

```
python -m scripts.train_allchis
```

Suppose that the directory that your checkpoints were placed into during that training run is `./logs/20250830_022819/checkpoints`. To run inference using the checkpoint with the best validation loss from training, taking as input the structures (in the PDB file format) in `in_dir` while dumping the predictions into `out_dir`, you can run:

```
python -m scripts.inference_allchis \
    --checkpoint_dir 20250830_022819 \
    --checkpoint best_val \
    --data_dir in_dir \
    --output_dir out_dir
```

To compare the predicted structures in `predictions_dir` with the ground-truth structures in `reference_dir`:

```
python -m scripts.assess_two_dirs \
    --target_dir reference_dir \
    --pred_dir predictions_dir
```
