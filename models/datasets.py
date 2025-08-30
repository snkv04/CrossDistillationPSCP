import os
import math
import json
import multiprocessing
from types import SimpleNamespace
from joblib import Parallel, delayed
import numpy as np
import random
import shutil
import json
from argparse import Namespace
import time
import argparse
import yaml

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm
from Bio.PDB.Polypeptide import one_to_index, three_to_one, one_to_three
from Bio.PDB import PDBParser
import pydssp

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.protein_utils.sidechains.sidechain_utils import get_sc_dihedral
from protein_learning.protein_utils.sidechains.sidechain_rigid_utils import atom37_to_torsion_angles
from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.default_feature_generator import DefaultFeatureGenerator
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc import train
from protein_learning.models.fbb_design.train import Train as SCPTrain, _augment
from protein_learning.models.utils.dataset_augment_fns import impute_cb
from protein_learning.models.inference_utils import set_canonical_coords_n_masks
from protein_learning.common.helpers import safe_normalize
from protein_learning.assessment.sidechain import debug
from flowpacker.dataset_cluster import get_features
import openfold.np.residue_constants as rc


def _histogram(vals, output_path, xlabel='', ylabel='', title='', num_bins=100):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(vals, torch.Tensor):
        print(f'vals.shape = {vals.shape}')
        vals = vals.flatten().cpu().numpy()
    plt.hist(vals, bins=num_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    vmin, vmax = min(vals), max(vals)
    plt.axvline(vmin, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(vmax, color='red', linestyle='dashed', linewidth=1)
    plt.text(vmin, plt.ylim()[1]*0.9, f"Min: {vmin:.2f}", color='red', ha='right')
    plt.text(vmax, plt.ylim()[1]*0.9, f"Max: {vmax:.2f}", color='red', ha='left')
    plt.savefig(output_path)
    print(f'Saved plot to {output_path}')
    plt.close()


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _dihedrals(pos_N, pos_CA, pos_C, eps=1e-7):
    """
    Args:
        pos_N, pos_CA, pos_C:   (N, 3).
    Returns:
        Dihedral features, (N, 6).
    """
    # print(f"pos n = {pos_N}")
    # print(f"pos ca = {pos_CA}")
    # print(f"pos c = {pos_C}")
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)   # (N, 3, 3)
    # print(f"X.shape before = {X.shape}")
    # print(f"X before = {X}")
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    # print(f"X.shape after = {X.shape}")
    # print(f"X after = {X}")
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    # print(f"cosD before = {cosD}")
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    # print(f"cosD after = {cosD.shape}")
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    # print(f"d features = {D_features.shape}")
    return D_features


def _orientations(pos_CA, resseq=None, mask_out_noncontiguous_residues=False, device=None):
    X = pos_CA
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    if mask_out_noncontiguous_residues:
        assert resseq is not None
        indices = torch.tensor(resseq, device=device)
        assert indices.shape[0] == pos_CA.shape[0]

        forward_mask, backward_mask = (torch.zeros_like(indices, dtype=torch.bool)
                                       for _ in range(2))
        adjacent_is_contiguous = (indices[1:] == indices[:-1] + 1)
        forward_mask[:-1] = adjacent_is_contiguous
        backward_mask[1:] = adjacent_is_contiguous

        forward = forward * forward_mask[:, None]
        backward = backward * backward_mask[:, None]
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _impute_sidechain_vectors(pos_N, pos_CA, pos_C):
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)   # (N, 3, 3)
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 


def _dist(r1, r2):
    r1, r2 = np.array(r1), np.array(r2)
    return np.sqrt(np.sum((r1 - r2) ** 2))


def _remove_unknown_flowpacker_residues(flowpacker_features):
    known_residues = np.array(flowpacker_features['aa_str']) != 'X'
    for key in flowpacker_features:
        if type(flowpacker_features[key]) == np.ndarray:
            flowpacker_features[key] = flowpacker_features[key][known_residues]
        elif type(flowpacker_features[key]) == list:
            flowpacker_features[key] = np.array(flowpacker_features[key])[known_residues].tolist()
    return flowpacker_features


def _process_single_entry(
    data_dir,
    pdb_file,
    allow_nonconsec_res=True,
    limit_CA_dist=False,
    check_bb_atoms=False,
    read_nonstandard_residues=False
):
    try:
        # print(f'allow_nonconsec_res = {allow_nonconsec_res}')
        # print(f'limit_CA_dist = {limit_CA_dist}')
        # print(f'check_bb_atoms = {check_bb_atoms}')
        # print(f'read_nonstandard_residues = {read_nonstandard_residues}')

        # Initializes the PDB parser
        parser = PDBParser(QUIET=True)
        
        # Loads the PDB
        pdb_path = os.path.join(data_dir, pdb_file)
        chain_name, _ = os.path.splitext(pdb_file)
        try:
            structure = parser.get_structure(chain_name, pdb_path)
        except Exception as e:
            raise ValueError(f'Failed to run parser.get_structure() for {chain_name}')
        structure_chains = list(structure.get_chains())
        assert len(structure_chains) == 1, f"Multiple chains somehow found for {chain_name}"

        # Runs some checks on the chain
        chain = structure_chains[0]
        resseq = []
        has_aa_res = False
        resnames = []
        missing_bb_atoms = []
        bb_plddt = []
        read_plddt = pdb_file.startswith("AF") \
            or (
                "data/FINAL/structures" in data_dir \
                and "_af" in data_dir
            )
        for residue in chain:
            if residue.id[0] == ' ': # Skips water molecules and residues with hetero-atoms
                # Gets the amino acid names from Biopython
                try:
                    resnames.append(three_to_one(residue.get_resname()))
                except:
                    if read_nonstandard_residues:
                        resnames.append('X')
                    else:
                        continue

                # Makes sure all backbone atoms are in this residue
                if check_bb_atoms:
                    for bb_atom in ['N', 'CA', 'C', 'O']:
                        try:
                            _ = residue[bb_atom]
                        except KeyError:
                            missing_bb_atoms.append({
                                "seqnum": residue.get_id()[1],
                                "aa": residue.get_resname(),
                                "atom": bb_atom
                            })

                # Gets the backbone pLDDTs
                # Doing it in this loop across residues instead of in a separate loop
                # (1) for efficiency, and
                # (2) because some residues might get thrown out (if they're of
                # nonstandard aa type), so the exact residues iterated through might
                # be different if iterating through them using some other method
                # (though this is rare, as, for instance, it only happens in a
                # single protein in the CASP15 set)
                if not read_plddt:
                    residue_bb_plddt = [100.0] * 4
                else:
                    residue_bb_plddt = []
                    for key in ['N', 'CA', 'C', 'O']:
                        if key in residue:
                            bb_atom = residue[key]
                            residue_bb_plddt.append(bb_atom.get_bfactor())
                        else:
                            residue_bb_plddt.append(0.0) # Will be masked out anyway later
                bb_plddt.append(residue_bb_plddt)

                # Optionally checks for non-contiguous residues
                if allow_nonconsec_res or (not len(resseq)) or residue.id[1] == resseq[-1] + 1:
                    resseq.append(residue.id[1])
                else:
                    raise ValueError(f"Residues {resseq[-1]} and {residue.id[1]} not contiguous for chain {chain_name}")
                
                # Marks that the chain has at least one residue (in case I don't end up
                # using the other lists outputted from this loop)
                has_aa_res = True
        if not has_aa_res:
            raise ValueError(f'No residues found for {chain_name}; skipping...')
        if check_bb_atoms and missing_bb_atoms:
            error_msg = f"Missing the following backbone atoms from chain {chain_name}:"
            for missing_bb_atom in missing_bb_atoms:
                error_msg += "\n" + str(missing_bb_atom)
            raise KeyError(error_msg)
        resnames = "".join(resnames)
        bb_plddt = torch.tensor(bb_plddt)
        assert bb_plddt.shape == (len(resseq), 4)
        
        # Gets backbone coordinates
        protein = Protein.FromPDB(pdb_path)
        bb_coords = protein.bb_atom_coords
        pos_N, pos_CA, pos_C, pos_O = torch.unbind(bb_coords, dim=1)
        sequence = protein.seq
        if len(sequence) != len(resseq):
            # print(f'resseq = \n{resseq}')
            # print(f'resnames = \n{resnames}')
            # print(f'attnpacker seq =\n{sequence}')
            # print(f'flowpacker seq before =\n{"".join(get_features(pdb_path)["aa_str"])}')
            # print(f'flowpacker seq after =\n{"".join(_remove_unknown_flowpacker_residues(get_features(pdb_path))["aa_str"])}', flush=True)
            raise ValueError(f'Different sequence length parsed between Biopython and AttnPacker for {chain_name}')
        if limit_CA_dist:
            for idx in range(1, len(sequence)):
                distance = _dist(pos_CA[idx], pos_CA[idx-1])
                if distance > 4.0:
                    curr_idx, last_idx = resseq[idx], resseq[idx-1]
                    raise ValueError(f"For {chain_name}, distance between {one_to_three(sequence[idx])} {curr_idx}" + \
                        f" and {one_to_three(sequence[idx-1])} {last_idx} was {distance} > 4.0")

        # Gets side-chain dihedral angles
        all_dihedral_info = atom37_to_torsion_angles(
            dict(
                aatype=protein.seq_encoding.unsqueeze(0),
                all_atom_positions=protein.atom_coords.unsqueeze(0),
                all_atom_mask=protein.atom_masks.unsqueeze(0),
            )
        )
        from_sin_cos = lambda x: torch.atan2(*x.unbind(-1))
        chis = from_sin_cos(all_dihedral_info["torsion_angles_sin_cos"][..., -4:, :]).squeeze(0)
        chis_sin_cos = all_dihedral_info["torsion_angles_sin_cos"][..., -4:, :].squeeze(0)
        chis_sin_cos = safe_normalize(chis_sin_cos)
        chi_mask = all_dihedral_info["torsion_angles_mask"][...,-4:].bool().squeeze(0)

        # Imputes the position of the carbon beta atoms per residue
        imputed_pos_CB = impute_cb(protein, protein)[0].get_atom_coords('CB')

        # Computes per-node vector features
        orientations = _orientations(pos_CA=pos_CA)
        imputed_sidechain_vectors = _impute_sidechain_vectors(pos_N=pos_N, pos_CA=pos_CA, pos_C=pos_C)
        node_v = torch.cat([orientations, imputed_sidechain_vectors.unsqueeze(-2)], dim=-2)

        # Gets secondary structure information
        dssp = pydssp.assign(bb_coords, out_type='c3')
        ss_as_str = ''.join(list(dssp)).replace('-', 'C')

        # Gets atom14 representation and mask
        flowpacker_features = _remove_unknown_flowpacker_residues(get_features(pdb_path))
        if len(sequence) != len(flowpacker_features['aa_str']):
            # Removes proteins with alternate location indicators, such as
            # "1SER" and "2SER" in place of a single "SER"
            raise ValueError(f'Different sequence length parsed between AttnPacker and FlowPacker for {chain_name}')
        atom14_coords = torch.tensor(flowpacker_features["coord"]).float()
        atom14_mask = torch.tensor(flowpacker_features["atom14_mask"]).float()

        return Data(
            # Actual attributes
            seq_fasta = sequence,
            seq = torch.LongTensor([rc.restype_order[c] for c in sequence]),
            resseq = resseq,
            pos_N = pos_N,
            pos_CA = pos_CA,
            pos_C = pos_C,
            pos_O = pos_O,
            protein_model = protein,
            chis = chis,
            chis_sin_cos = chis_sin_cos,
            chi_mask = chi_mask,
            node_v = node_v,
            imputed_pos_CB = imputed_pos_CB,
            secondary_structure = ss_as_str,
            atom14_coords = atom14_coords,
            atom14_mask = atom14_mask,
            bb_plddt = bb_plddt,

            # Metadata
            num_chains = 1, # PDBs were split by chains
            name = chain_name,
            # cath = entry['CATH'],
            # mask = mask,
            num_nodes = len(sequence),
        )
    except Exception as e:
        print(f"Discarding target {os.path.splitext(pdb_file)[0]}: {e}")
        return None


class PSCPDataset(Dataset):

    # TODO: Fix edge cases with regard to the subset checkpoints
    def __init__(self, root, subset=""):
        super().__init__()
        self.data_dir = root
        assert subset in ("", "train", "val")
        self.subset = subset
        self.cache_path = os.path.join(
            root,
            'processed_pscp.pt' if not subset else f'processed_pscp_{subset}.pt'
        )

        # Loads PDBs
        self.dataset = None
        start_time = time.time()
        self._load()
        end_time = time.time()
        print(f'Initialized PSCPDataset in {end_time - start_time} seconds from {root}')

    def _load(self):
        if os.path.exists(self.cache_path):
            print(f'Loading from cache: {self.cache_path}')
            self.dataset = torch.load(self.cache_path)
        else:
            self.dataset = self._process()
        self._create_or_update_index_map()

    def _create_or_update_index_map(self):
        self.name_to_idx = {self.dataset[idx].name: idx for idx in range(len(self.dataset))}

    def shuffle_datapoints(self):
        random.shuffle(self.dataset)
        self._create_or_update_index_map()

    def _process(self):
        num_processes = os.cpu_count()
        # num_processes = 16
        dataset = Parallel(n_jobs=num_processes)(
            delayed(_process_single_entry)(self.data_dir, pdb_file)
            for pdb_file in tqdm(os.listdir(self.data_dir), desc='Preprocessing')
            if pdb_file.endswith('.pdb')
        )
        # dataset = []
        # for pdb_file in tqdm(os.listdir(self.data_dir), desc='Preprocessing'):
        #     dataset.append(_process_single_entry(self.data_dir, pdb_file))
        print(f"len before = {len(dataset)}")
        dataset = [item for item in dataset if item is not None]
        print(f"len after = {len(dataset)}")

        torch.save(dataset, self.cache_path)
        print(f'Saved to cache: {self.cache_path}')

        return dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, str):
            index = self.name_to_idx[key]
        else:
            index = key
        data = self.dataset[index].clone()

        return data
    
    def split(self, train_proportion=0.8, overwrite=True):
        """
        Splits the dataset into train and validation PSCPDataset instances.
        Returns:
            train_set (PSCPDataset): dataset with a subset of the original data
            val_set (PSCPDataset): dataset with a different subset of the original data
        """
        assert self.subset == '', 'Can\'t split an already split dataset'

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_proportion)

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_dataset = PSCPDataset.__new__(PSCPDataset)
        val_dataset = PSCPDataset.__new__(PSCPDataset)

        # Manually initialize internals (skip __init__)
        for obj, subset_indices, subset_name in zip([train_dataset, val_dataset],
                                                    [train_indices, val_indices],
                                                    ["train", "val"]):
            obj.data_dir = self.data_dir
            obj.subset = subset_name  
            obj.cache_path = os.path.join(
                self.data_dir,
                f'processed_pscp_{subset_name}.pt'
            )      

            if os.path.isfile(obj.cache_path) and not overwrite:
                print(f'Loading {subset_name} split from cache: {obj.cache_path}')
                obj.dataset = torch.load(obj.cache_path)
            else:
                if os.path.isfile(obj.cache_path):
                    print(f'Overwriting {obj.cache_path}')
                obj.dataset = [self.dataset[i] for i in tqdm(subset_indices, desc=f"Creating {subset_name} split")]
                torch.save(obj.dataset, obj.cache_path)
                print(f'Saved the {subset_name} subset with {len(obj.dataset)} items to {obj.cache_path}')
    
            obj._create_or_update_index_map()

        return train_dataset, val_dataset
    
    def dump_sequences_to_fasta(self, fasta_path: str = None):
        """
        Writes all protein sequences in the dataset into a multi-FASTA file.

        Args:
            fasta_path (str): Path to the output FASTA file
        """
        if not fasta_path:
            fasta_path = os.path.join(self.data_dir, "sequences_from_pscpdataset.fasta")

        with open(fasta_path, "w") as f:
            for data in tqdm(self.dataset):
                # Ensure sequence exists
                if not hasattr(data, "seq_fasta") or data.seq_fasta is None:
                    raise ValueError(f"{data.name} is missing a sequence?")

                header = f">{data.name}"
                sequence = data.seq_fasta
                f.write(header + "\n")

                # Optionally wrap lines at 80 characters (FASTA convention)
                for i in range(0, len(sequence), 80):
                    f.write(sequence[i:i+80] + "\n")
        
        print(f"Wrote sequences to {fasta_path}")


class CombinedDataset(Dataset):
    def __init__(
            self,
            dataset_a: PSCPDataset,
            dataset_b: PSCPDataset,
            num_a: int,
            num_b: int):
        assert ((num_a <= len(dataset_a)) and (num_b <= len(dataset_b)))

        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.num_a = num_a
        self.num_b = num_b
        self.indices_a = []
        self.indices_b = []
        self.current_data = []
        self.resample()

    # TODO: Implement this to work properly for distributed training when
    # resampling is done each epoch (right now, it's done only once before
    # all epochs)
    def resample(self):
        self.indices_a = random.sample(range(len(self.dataset_a)), self.num_a)
        self.indices_b = random.sample(range(len(self.dataset_b)), self.num_b)
        self.current_data = [(0, i) for i in self.indices_a] + [(1, i) for i in self.indices_b]
        random.shuffle(self.current_data)

    def __len__(self):
        return len(self.current_data) # self.num_a + self.num_b

    def __getitem__(self, idx):
        source, local_idx = self.current_data[idx]
        if source == 0:
            return self.dataset_a[local_idx]
        else:
            return self.dataset_b[local_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./casp16_af3_predictions",
        help="Path to data directory"
    )
    args = parser.parse_args()

    # data_dir = '/home/common/proj/side_chain_packing/data/bc40_dataset/by_chains/after_seq_sim_filtering'
    # data_dir = '/home/common/proj/side_chain_packing/data/FINAL/structures/casp14/casp14_native'
    # data_dir = '/home/common/proj/side_chain_packing/data/afdb/6_after_our_filters'
    # data_dir = '/home/common/proj/side_chain_packing/data/bc40_plus_afdb'
    # data_dir = '/home/common/proj/side_chain_packing/data/pdb_s40/5_after_sim_against_casp16'
    # data_dir = '/home/common/proj/side_chain_packing/data/afdb/0_initial_files'

    dataset = PSCPDataset(root=args.data_dir)
    print(f"Loaded PSCP dataset with {len(dataset)} items.")

    # torch.set_printoptions(threshold=1e9)
    data = dataset[9]
    print(f"data = {data}")
    # seq_lengths = [datapoint.pos_CA.shape[0] for datapoint in dataset]
    # _histogram(seq_lengths, 'sequence_lengths.png', 'Chain Length (# of residues)', 'Number of Proteins')

    # train_set, val_set = dataset.split(overwrite=False)

    # dataset.dump_sequences_to_fasta()
