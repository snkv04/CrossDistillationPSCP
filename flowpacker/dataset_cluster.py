import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import multiprocessing
import logging
import numpy as np
import random
from tqdm.contrib.concurrent import process_map

import warnings
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure

from flowpacker.utils.constants import three_to_one_letter, letter_to_num, max_num_heavy_atoms, \
    restype_to_heavyatom_names, heavyatom_to_label, chi_alt_truths, num_to_letter, chi_true_indices, chi_mask, atom_types, atom_type_num

from flowpacker.utils.sidechain_utils import get_bb_dihedral, get_chi_angles

from torch_geometric.data import Data, DataLoader
from torch_cluster import radius_graph, knn_graph
import math
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import pandas as pd


def get_features(path):
    try:
        if path.endswith(".cif"):
            with open(path, "r") as f:
                structure = PDBxFile.read(f)
                structure = get_structure(structure)
        else:
            with open(path, "r") as f:
                structure = PDBFile.read(f)
                structure = structure.get_structure()
    except Exception as e:
        print(f'Error: {e}')
        return None

    # if struc.get_chain_count(structure) > 1: return None # only single chains

    _, aa = struc.get_residues(structure)
    # Replace nonstandard amino acids with X
    for idx, a in enumerate(aa):
        if a not in three_to_one_letter.keys():
            aa[idx] = 'UNK'

    aa_str = [three_to_one_letter.get(i,'X') for i in aa]
    aa_num = [letter_to_num[i] for i in aa_str]

    # if len(aa_str) > self.max_length or len(aa_str) < self.min_length:
    #     return None
    # if len(aa_str) < self.min_length: return None

    aa_mask = np.ones(len(aa))
    atom14_mask = np.zeros((len(aa), max_num_heavy_atoms))
    atom37_mask = np.zeros((len(aa), atom_type_num))
    # Iterate through all residues
    coords, coords37, atom_type, chain_ids = [], [], [], []
    for res_idx, res in enumerate(struc.residue_iter(structure)):
        res_coords = res.coord[0]
        res_name = aa[res_idx]
        chain_ids.append(res.chain_id[0])

        if res_name == "UNK":
            aa_mask[res_idx] = 0

        # Append true coords
        res_crd14 = np.zeros((max_num_heavy_atoms, 3))
        res_crd37 = np.zeros((atom_type_num, 3))
        res_atom_type = []
        for atom14_idx, r in enumerate(restype_to_heavyatom_names[res_name]):
            if r == '':
                res_atom_type.append(4)
                continue
            atom37_idx = atom_types.index(r)
            res_atom_type.append(heavyatom_to_label[r[0]])
            i = np.where(res.atom_name == r)[0]
            if i.size == 0:
                res_crd14[atom14_idx] = 0
                res_crd37[atom37_idx] = 0

            else:
                res_crd14[atom14_idx] = res_coords[i[0]]
                atom14_mask[res_idx, atom14_idx] = 1
                res_crd37[atom37_idx] = res_coords[i[0]]
                atom37_mask[res_idx, atom37_idx] = 1
        coords.append(res_crd14)
        coords37.append(res_crd37)
        atom_type.append(res_atom_type)

    coords = np.array(coords)
    atom_type = np.array(atom_type)
    aa_num = np.array(aa_num)
    chain_ids = np.array(chain_ids)

    assert len(coords) == len(aa_num)

    return {
        "coord": coords,
        "atom_type": atom_type,
        "aa_num": aa_num,
        "aa_str": aa_str,
        "mask": aa_mask,
        "atom14_mask": atom14_mask,
        "chain_id": chain_ids
    }
