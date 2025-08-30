import functools
import yaml
from types import SimpleNamespace
import os
import json
import math
from typing import *
import argparse
import time

import torch
from torch import nn
import torch_cluster
from torch_geometric.data import Batch
from torch.nn import functional as F

from modules.linear import rotate_apply
from modules.common import ScalarVector
from modules.gconv import SVGraphConvLayer
from modules.geometric import construct_3d_basis
from modules.norm import SVLayerNorm
from modules.perceptron import VectorPerceptron
from models.datasets import PSCPDataset, _remove_unknown_flowpacker_residues, \
    _orientations, _impute_cb_vectors
from utils.misc import BlackHole
from models.loss_fns import huber_loss
from modules.dropout import SVDropout
from modules.geometric import orthogonalize_matrix, apply_rotation

from attnpacker.protein_learning.models.model_abc import train
from attnpacker.protein_learning.features.input_embedding import InputEmbedding
from attnpacker.protein_learning.models.utils.dataset_augment_fns import impute_cb
from attnpacker.protein_learning.models.fbb_design.train import _augment
from attnpacker.protein_learning.models.inference_utils import set_canonical_coords_n_masks
from attnpacker.protein_learning.common.data.data_types.model_input import ModelInput

from flowpacker.utils.sidechain_utils import Idealizer
from flowpacker.dataset_cluster import get_features
from flowpacker.utils.structure_utils import create_structure_from_crds

from pippack.loss import rotamer_recovery_from_coords, nll_chi_loss, offset_mse, \
    BlackHole, sc_rmsd, local_interresidue_sc_clash_loss, unclosed_proline_loss


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _get_ss_tensor(ss_str, device):
    to_indices = {
        'C': 0, # loop / coil
        'H': 1, # alpha helix
        'E': 2, # beta sheet
    }
    as_indices = torch.Tensor([to_indices[c] for c in ss_str]).long().to(device)
    one_hot = F.one_hot(as_indices, num_classes=len(to_indices)).float()
    return one_hot


class PSCPAllChisNetwork(nn.Module):
    def __init__(
        self,
        node_hid_dims=(128, 32), edge_hid_dims=(64, 16),
        num_layers=3, drop_rate=0.1, top_k=30, aa_embed_dim=20,
        num_rbf=16, num_positional_embeddings=16,
        node_out_s_dim=2,
        predict_binned_chis=True, num_chi_bins=72,
        perceptron_mode='svp', conv='gnn', no_vec=False,
        separate_dense_layers=True, separate_offset_layers=True,
        recycle_chi_bin_probs=False, recycle_chi_sincos=False, recycle_sc_coords=False,
        mask_front_and_back_vec=False, unet=True, aa_embeds_on_back_edges=True,
        use_svp_for_dense_and_offset=True
    ):
        super().__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.dummy = nn.Parameter(torch.empty(0))
        self.no_vec = no_vec
        self.predict_binned_chis = predict_binned_chis
        self.num_chi_bins = num_chi_bins
        self.eps = 1e-12
        self.separate_dense_layers = separate_dense_layers
        self.separate_offset_layers = separate_offset_layers
        self.unet = unet
        self.aa_embeds_on_back_edges = aa_embeds_on_back_edges
        self.use_svp_for_dense_and_offset = use_svp_for_dense_and_offset

        # ================ AttnPacker input features ================
        # Setting up feature generation
        self.arg_groups = self._get_arg_groups()
        self.input_feature_config = train.get_input_feature_config(
            self.arg_groups,
            pad_embeddings=True,
            extra_pair_feat_dim=0,
            extra_res_feat_dim=0,
        )
        self.feat_gen = train.get_feature_gen(
            self.arg_groups,
            self.input_feature_config,
            apply_masks=True,
        )

        # For input embedding
        self.input_embedding = InputEmbedding(self.input_feature_config)
        self.node_s_in_dim, self.edge_s_in_dim = self.input_embedding.dims # 115, 205
        self.node_s_in_dim += 3 # for secondary structure encodings
        # ===========================================================

        Perceptron_ = functools.partial(VectorPerceptron, mode=perceptron_mode)

        # Initial embedding of node features
        self.mask_front_and_back_vec = mask_front_and_back_vec
        if not no_vec:
            node_in_dims = (self.node_s_in_dim, 3)
        else:
            node_in_dims = (self.node_s_in_dim, 1)
        self.W_node = nn.Sequential(
            Perceptron_(node_in_dims, node_hid_dims, scalar_act=None, vector_act=None),
            SVLayerNorm(node_hid_dims),
            Perceptron_(node_hid_dims, node_hid_dims, scalar_act=None, vector_act=None),
        )

        # Initial embedding of edge features
        if not no_vec:
            edge_in_dims = (self.edge_s_in_dim, len(self.input_feature_config.rel_dist_atom_pairs))
        else:
            edge_in_dims = (self.edge_s_in_dim, 1)
        self.W_edge = nn.Sequential(
            Perceptron_(edge_in_dims, edge_hid_dims, scalar_act=None, vector_act=None),
            SVLayerNorm(edge_hid_dims),
            Perceptron_(edge_hid_dims, edge_hid_dims, scalar_act=None, vector_act=None),
        )

        # Encoder
        self.encoder_layers = nn.ModuleList(
            SVGraphConvLayer(
                node_hid_dims, 
                edge_hid_dims, 
                mlp_mode=perceptron_mode, 
                drop_rate=drop_rate,
                conv=conv,
            ) for _ in range(num_layers)
        )

        # Embed known amino acids
        self.aa_embed = nn.Embedding(20, embedding_dim=aa_embed_dim) # Each of 20 AAs embedded in length-(aa_embed_dim) dense vectors
        edge_dec_hid_dims = (edge_hid_dims[0]+aa_embed_dim, edge_hid_dims[1]) # Extends scalar portion by aa_embed_dim

        # Decoder
        self.decoder_layers = nn.ModuleList(
            SVGraphConvLayer(
                node_hid_dims, 
                edge_dec_hid_dims, 
                mlp_mode=perceptron_mode, 
                drop_rate=drop_rate,
                conv=conv,
                autoregressive=True,
            ) for _ in range(num_layers)
        )

        # Output layers: convert vector and scalar features to scalar only features
        node_s_dim = node_hid_dims[0]
        if self.use_svp_for_dense_and_offset:
            self.W_out = nn.Sequential(
                SVLayerNorm(node_hid_dims),
                Perceptron_(node_hid_dims, node_hid_dims), # Keep vector information
            )
        else:
            # TODO: Change from 1 unused vector output to actually 0 vector outputs
            # (if DistributedDataParallel doesn't crash out when doing it)
            self.W_out = nn.Sequential(
                SVLayerNorm(node_hid_dims),
                Perceptron_(node_hid_dims, (node_s_dim, 1)),
            )
        if self.predict_binned_chis:
            # Create the dense layer(s) that outputs the bins
            if self.separate_dense_layers:
                if self.use_svp_for_dense_and_offset:
                    # TODO: Change from 1 unused vector output to actually 0 vector outputs
                    # (if DistributedDataParallel doesn't crash out when doing it)
                    self.denses = nn.ModuleList(
                        nn.Sequential(
                            Perceptron_(node_hid_dims, node_hid_dims, scalar_act='relu'),
                            SVDropout(drop_rate=drop_rate),
                            Perceptron_(node_hid_dims, (self.num_chi_bins, 1), scalar_act=None, vector_act=None),
                        ) for _ in range(4)
                    )
                else:
                    self.denses = nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(node_s_dim, node_s_dim), nn.ReLU(),
                            nn.Dropout(p=drop_rate),
                            nn.Linear(node_s_dim, self.num_chi_bins),
                        ) for _ in range(4)
                    )
            else:
                # TODO: Refactor this branch to use the use_svp_for_dense_and_offset flag
                # to determine whether or not to use SVPerceptron
                self.dense = nn.Sequential(
                    nn.Linear(node_s_dim, node_s_dim), nn.ReLU(),
                    nn.Dropout(p=drop_rate),
                    nn.Linear(node_s_dim, 4 * self.num_chi_bins), # All in the same dimension, later gets reshaped
                )

            # Create the offset layer that outputs the offsets from the center of the bins
            if self.separate_offset_layers:
                if self.use_svp_for_dense_and_offset:
                    # TODO: Change from 1 unused vector output to actually 0 vector outputs
                    # (if DistributedDataParallel doesn't crash out when doing it)
                    self.offset_layers = nn.ModuleList(
                        Perceptron_(node_hid_dims, (1, 1), scalar_act=None, vector_act=None) for _ in range(4)
                    )
                else:
                    self.offset_layers = nn.ModuleList(
                        nn.Linear(node_s_dim, 1) for _ in range(4)
                    )
            else:
                # TODO: Refactor this branch to use the use_svp_for_dense_and_offset flag
                # to determine whether or not to use SVPerceptron
                self.offset_layer = nn.Linear(node_s_dim, 4) # Single offset per chi

        else:
            self.node_out_s_dim = node_out_s_dim
            self.dense = nn.Sequential(
                nn.Linear(node_s_dim, node_s_dim), nn.ReLU(),
                nn.Dropout(p=drop_rate),
                nn.Linear(node_s_dim, 4 * node_out_s_dim), # All in the same dimension, later gets reshaped
            )

        # Layers used for recycling previous predictions
        self.recycle_chi_bin_probs = recycle_chi_bin_probs
        self.recycle_chi_sincos = recycle_chi_sincos  # TODO: Implement a layer to recycle these
        self.recycle_sc_coords = recycle_sc_coords  # TODO: Implement a layer to recycle these
        if self.recycle_chi_bin_probs:
            self.W_recycle_chi_bin_probs = nn.Linear(
                4 * self.num_chi_bins, node_hid_dims[0]
            )

        # Used for generating final coordinates
        self.idealizer = Idealizer(use_native_bb_coords=True)

    @property
    def metric_names(self) -> Sequence[str]:
        metrics = [
            "rotamer recovery",
            "rmsd",
        ]

        if self.predict_binned_chis:
            metrics.append(
                "chi nll loss"
            )
            metrics.append(
                "offset mse loss"
            )

        return metrics

    def _get_arg_groups(self):
        arg_groups_path = './models/configs/arg_groups.yaml'
        with open(arg_groups_path, "r") as f:
            arg_groups_dict = yaml.safe_load(f)
        arg_groups = {k: SimpleNamespace(**v) for k, v in arg_groups_dict.items()}
        return arg_groups
    
    def _get_model_input_representation(self, protein, seq_mask=None, dihedral_mask=None,):
        protein, _ = impute_cb(protein, protein)
        extra = _augment(protein, protein)
        protein = set_canonical_coords_n_masks(protein)

        return ModelInput(
            decoy=protein,
            input_features=self.feat_gen.generate_features(
                protein,
                extra=extra,
                seq_mask=seq_mask,
                dihedral_mask=dihedral_mask,
            ),
            extra=extra,
        )

    def _get_node_and_edge_scalars(self, batch, edge_index, device='cuda'):
        batch_size = len(batch.ptr) - 1
        node_s_per_protein, edge_s_per_protein = [], []
        ptr = 0
        for i in range(batch_size):
            protein = batch.protein_model[i]
            model_input_repr = self._get_model_input_representation(protein)
            init_residue_feats, init_pair_feats = map(
                lambda x : x.squeeze(0),
                self.input_embedding(model_input_repr.input_features.to(device))
            )

            # Gets secondary structure features to concatenate onto initial node features
            ss_feats = _get_ss_tensor(batch.secondary_structure[i], device=device)
            init_residue_feats = torch.cat((init_residue_feats, ss_feats), dim=1)

            # Appends node features for individual protein
            node_s_per_protein.append(init_residue_feats)

            # Finishes computing edge features for individual protein
            num_nodes_for_protein = batch.ptr[i+1] - batch.ptr[i]
            num_edges_for_protein = num_nodes_for_protein * min(self.top_k, num_nodes_for_protein - 1)
            edge_index_for_protein = edge_index[:, ptr:ptr+num_edges_for_protein] - batch.ptr[i]
            ptr += num_edges_for_protein
            selected_edge_feats = init_pair_feats[edge_index_for_protein[0], edge_index_for_protein[1], :]
            edge_s_per_protein.append(selected_edge_feats)

        # Combines the features from each individual protein into batch features
        node_s = torch.cat(node_s_per_protein, dim=0)
        edge_s = torch.cat(edge_s_per_protein, dim=0)
        return node_s, edge_s
    
    def _get_node_vectors(self, batch, device='cuda'):
        batch_size = len(batch.ptr) - 1
        node_v_per_protein = []
        for i in range(batch_size):
            protein_start, protein_end = batch.ptr[i], batch.ptr[i+1]
            orientations = _orientations(
                pos_CA=batch.pos_CA[protein_start:protein_end],
                resseq=batch.resseq[i],
                mask_out_noncontiguous_residues=self.mask_front_and_back_vec,
                device=device,
            )
            imputed_sidechain_vectors = _impute_cb_vectors(
                pos_N=batch.pos_N[protein_start:protein_end],
                pos_CA=batch.pos_CA[protein_start:protein_end],
                pos_C=batch.pos_C[protein_start:protein_end],
            )
            concatenated = torch.cat([orientations, imputed_sidechain_vectors.unsqueeze(-2)], dim=-2)
            node_v_per_protein.append(concatenated)
        node_v = torch.cat(node_v_per_protein, dim=0)
        return node_v

    def _impute_cb_batch(self, batch):
        return torch.cat(
            [impute_cb(protein, protein)[0].get_atom_coords('CB') for protein in batch.protein_model],
            dim=0
        )

    def _get_inputs(self, batch):
        device = self.dummy.device

        # Gets scalar features for nodes and edges
        assert batch.batch is not None
        edge_index = torch_cluster.knn_graph(
            batch.pos_CA,
            k=self.top_k,
            batch=batch.batch,
            flow='target_to_source'
        )
        node_s, edge_s = self._get_node_and_edge_scalars(batch, edge_index, device=device)

        # Gets vector features for nodes and edges
        if not self.no_vec:
            # node_v = batch.node_v
            node_v = self._get_node_vectors(batch, device=device)
            pos_CB = batch.imputed_pos_CB if batch.imputed_pos_CB is not None else self._impute_cb_batch(batch).to(device)
            atom_type_to_pos = {
                'N': batch.pos_N,
                'CA': batch.pos_CA,
                'C': batch.pos_C,
                'O': batch.pos_O,
                'CB': pos_CB,
            }
            E_vectors = []
            for pair in self.input_feature_config.rel_dist_atom_pairs:
                E_vectors.append(
                    atom_type_to_pos[pair[1]][edge_index[0]] - atom_type_to_pos[pair[0]][edge_index[1]]
                )
            edge_v = torch.cat(
                list(map(lambda vectors: _normalize(vectors).unsqueeze_(-2), E_vectors)),
                dim=-2
            )
        else:
            # TODO: Change from 1 all-zero vector feature to actually 0 vector features
            node_v = torch.zeros((len(batch.pos_CA), 1, 3), device=self.dummy.device)
            edge_v = torch.zeros((len(edge_index[0]), 1, 3), device=self.dummy.device)

        # Constructs ScalarVector objects for features
        node_in = ScalarVector(s=node_s, v=node_v)
        edge_in = ScalarVector(s=edge_s, v=edge_v)

        # Get rotation matrices
        R_node = construct_3d_basis(batch.pos_CA, batch.pos_C, batch.pos_N)    # (N, 3, 3)
        R_node = R_node.nan_to_num()
        R_edge = R_node[edge_index[0]]  # (E, 3, 3)
        return edge_index, node_in, edge_in, R_node, R_edge

    def forward(self, batch, recycling_iters=0, grads_during_recycling=False):
        # Initializes previous outputs to all zeros
        prev_output = {
            'chi_bin_logits': torch.zeros((batch.pos_CA.shape[0], 4, self.num_chi_bins),
                                          device=self.dummy.device),
            'chi_bin_probs': torch.zeros((batch.pos_CA.shape[0], 4, self.num_chi_bins),
                                         device=self.dummy.device),
            'chi_bin_log_probs': torch.zeros((batch.pos_CA.shape[0], 4, self.num_chi_bins),
                                             device=self.dummy.device),
            'chi_offsets': torch.zeros_like(batch.chis),
        }

        # Runs recycling steps
        if not grads_during_recycling:
            with torch.no_grad():
                for it in range(recycling_iters):
                    output = self.single_forward_pass(batch, prev_output)
                    prev_output = output
        else:
            for it in range(recycling_iters):
                output = self.single_forward_pass(batch, prev_output)
                prev_output = output

        # Runs final forward pass
        output = self.single_forward_pass(batch, prev_output)
        return output

    def single_forward_pass(self, batch, prev_output):
        edge_index, node_in, edge_in, R_node, R_edge = self._get_inputs(batch)

        h_node = rotate_apply(self.W_node, node_in, R_node)
        h_edge = rotate_apply(self.W_edge, edge_in, R_edge)

        # Incorporates previous predictions into hidden node representations
        if self.recycle_chi_bin_probs:
            h_node = h_node + ScalarVector(
                self.W_recycle_chi_bin_probs(prev_output["chi_bin_probs"].view(
                    batch.chis.shape[0], -1
                )),
                0  # Broadcasts to correct shape
            )

        skip_inputs = []
        for layer in self.encoder_layers:
            skip_inputs.append(h_node)
            h_node = layer(h_node, edge_index, h_edge, rot=R_node)     

        # These are just the node features after the encoder layers, so then
        # even though h_node gets updated between the decoder layers, the values
        # at this point get repeatedly get fed to each decoder layer
        encoder_embeddings = h_node

        edge_index_i, edge_index_j = edge_index
        h_seq = self.aa_embed(batch.seq)    # (N, aa_embed_dim)
        h_seq = h_seq[edge_index_j]        # Amino acid embedding of j-nodes.
        if not self.aa_embeds_on_back_edges:
            h_seq[edge_index_j >= edge_index_i] = 0
        h_edge_autoregressive = ScalarVector(
            s = torch.cat([h_edge.s, h_seq], dim=-1),
            v = h_edge.v,
        )

        for layer in self.decoder_layers:
            if self.unet:
                autoregressive_x = skip_inputs.pop()
            else:
                autoregressive_x = encoder_embeddings

            h_node = layer(h_node,
                           edge_index,
                           h_edge_autoregressive,
                           autoregressive_x=autoregressive_x,
                           rot=R_node)

        output = {}
        if self.use_svp_for_dense_and_offset:
            out = rotate_apply(self.W_out, h_node, R_node)
        else:
            out = rotate_apply(self.W_out, h_node, R_node).s  # (N, node_hid_s)
        n = batch.pos_CA.shape[0]
        if self.predict_binned_chis:
            # Gets bin probs
            if self.separate_dense_layers:
                if self.use_svp_for_dense_and_offset:
                    separate_chi_logits = [dense(out).s for dense in self.denses] # Each one is (N, 72)
                else:
                    separate_chi_logits = [dense(out) for dense in self.denses] # Each one is (N, 72)
                chi_bin_logits = torch.stack(separate_chi_logits, dim=1)
            else:
                # TODO: Refactor this branch to use the use_svp_for_dense_and_offset flag
                # to determine whether or not to use SVPerceptron
                chi_bin_logits = self.dense(out).reshape(n, 4, self.num_chi_bins) # (N, 4, 72)
            chi_bin_probs = F.softmax(chi_bin_logits, dim=-1)
            chi_bin_log_probs = F.log_softmax(chi_bin_logits, dim=-1)

            # Gets offsets from middle of bins
            bin_width = 2 * math.pi / self.num_chi_bins
            if self.separate_offset_layers:
                if self.use_svp_for_dense_and_offset:
                    separate_offsets = [
                        (bin_width * (torch.sigmoid(offset_layer(out).s) - 0.5))
                        for offset_layer in self.offset_layers
                    ]
                else:
                    separate_offsets = [
                        (bin_width * (torch.sigmoid(offset_layer(out)) - 0.5))
                        for offset_layer in self.offset_layers
                    ]
                offset = torch.stack(separate_offsets, dim=1)
            else:
                # TODO: Refactor this branch to use the use_svp_for_dense_and_offset flag
                # to determine whether or not to use SVPerceptron
                offset = (bin_width * (torch.sigmoid(self.offset_layer(out)) - 0.5)) \
                    .reshape(n, 4, 1) # (N, 4, 1)

            output['chi_bin_logits'] = chi_bin_logits
            output['chi_bin_probs'] = chi_bin_probs
            output['chi_bin_log_probs'] = chi_bin_log_probs
            output['chi_offsets'] = offset

            chis = self._get_chi_predictions(output, strategy="mode").squeeze(-1)
            gumbel_sampled_chis = self \
                ._get_chi_predictions(output, strategy="gumbel_sample") \
                .squeeze(-1)
            output['chis'] = chis
            output['gumbel_sampled_chis'] = gumbel_sampled_chis
        else:
            trig_dihedrals = self.dense(out).reshape(n, 4, self.node_out_s_dim)   # (N, 4, 2)
            output['trig_dihedrals'] = trig_dihedrals

        return output
    
    def sample_pdb(
        self,
        batch,
        input_path,
        output_path,
    ):
        # Obtains predictions
        assert batch.num_graphs == 1
        n = len(batch.seq)
        if self.predict_binned_chis:
            output = self.forward(batch.to(self.dummy.device))
            chis = output['chis']
        else:
            raise NotImplementedError()

        # Computes FlowPacker features and removes unknown amino acid residues
        flowpacker_features = _remove_unknown_flowpacker_residues(get_features(input_path))
        aa_num = flowpacker_features['aa_num']
        atom_mask = torch.Tensor(flowpacker_features['atom14_mask']).to(self.dummy.device)
        bb_coords = torch.cat([
            batch.pos_N.unsqueeze(1),
            batch.pos_CA.unsqueeze(1),
            batch.pos_C.unsqueeze(1),
            batch.pos_O.unsqueeze(1)
        ], dim=1)
        
        # Calculates and saves the atom coordinates
        all_atom_coords = self.idealizer(aa_num, bb_coords, chis) * atom_mask.unsqueeze(-1)
        aa_str = flowpacker_features['aa_str']
        create_structure_from_crds(aa_str, all_atom_coords, atom_mask, outPath=output_path, save_traj=False)

    def _bin_and_offset_chis(self, chi_angles):
        tol = 1e-5
        assert torch.all(chi_angles >= -math.pi - tol) and torch.all(chi_angles <= math.pi + tol), \
            "All chi angles must be in [-π, π]"

        # Calculates bin indices
        bin_width = 2 * math.pi / self.num_chi_bins
        chi_normalized = torch.remainder(chi_angles + math.pi, 2 * math.pi)
        bin_idx = torch.floor(chi_normalized / bin_width).long().clamp(max=self.num_chi_bins-1)
        chi_one_hot = F.one_hot(bin_idx, num_classes=self.num_chi_bins)

        # Calculates offsets
        bin_center = -math.pi + (bin_idx.float() + 0.5) * bin_width
        chi_offset = (chi_normalized - math.pi) - bin_center

        # Assert offset is within [-bin_width/2, bin_width/2]
        assert torch.all(chi_offset >= (-bin_width / 2) - tol) and \
            torch.all(chi_offset <= (bin_width / 2) + tol), \
            (
                "Chi offsets should lie within [-bin_width/2, bin_width/2], "
                f"but offset was (max={torch.max(chi_offset)}, min={torch.min(chi_offset)}) "
                f"while bin_width/2 is {bin_width/2.0}"
            )

        return bin_idx, chi_one_hot, chi_offset
    
    def _get_chi_predictions(self, output, strategy="gumbel_sample", gumbel_tau=1.0): 
        """Adapted from https://github.com/Kuhlman-Lab/PIPPack"""

        # Extracts outputs
        chi_bin_logits = output['chi_bin_logits']
        chi_bin_probs = output['chi_bin_probs']
        chi_bin_offsets = output['chi_offsets']

        # Gets predicted chi bin centers
        bin_width = 2 * math.pi / self.num_chi_bins
        if strategy in ("mode", "multinomial_sample"):
            if strategy == "mode":
                chi_bin = torch.argmax(chi_bin_probs, dim=-1)
            else:
                # TODO: Adapt this for predicting all chis instead of only 1 like before (if needed?)
                chi_bin = torch.multinomial(
                    chi_bin_probs.view(-1, chi_bin_probs.shape[-1]), num_samples=1
                ).squeeze(-1).view(*chi_bin_probs.shape[:-1])
                raise NotImplementedError()
            pred_bin_center = (-math.pi + ((chi_bin.float() + 0.5) * bin_width)).unsqueeze(-1)

        elif strategy == "gumbel_sample":
            bin_centers = ((torch.arange(self.num_chi_bins, device=self.dummy.device) + 0.5) * 
                           bin_width) - math.pi
            chi_bin_onehot = F.gumbel_softmax(chi_bin_logits, gumbel_tau, hard=True)
            pred_bin_center = torch.sum(
                bin_centers \
                    .reshape(1, 1, self.num_chi_bins) \
                    .expand(chi_bin_onehot.shape[0], 4, self.num_chi_bins) \
                    * chi_bin_onehot,
                dim=-1,
                keepdim=True
            )

        else:
            raise NotImplementedError(
                "Choose an existing logit decoding method "
                "(mode, multinomial_sample, or gumbel_sample)"
            )

        # Determines actual chi value from bin (doesn't need onehot) and offset
        sampled_chi = pred_bin_center + chi_bin_offsets
        return sampled_chi

    def _get_representative_bb_plddt(
        self,
        batch,
        method='avg',
        sharpness=10.0,
        shift=0.7,
        threshold=0.9,
    ):
        if method == 'assume_native':
            return torch.ones(
                batch.pos_CA.shape[0],
                dtype=batch.pos_CA.dtype,
                device=batch.pos_CA.device
            )

        # Compute average backbone pLDDT
        bb_mask = batch.atom14_mask[:, :4]
        masked_sum = ((batch.bb_plddt / 100.0) * bb_mask).sum(dim=-1)
        mask_sum = bb_mask.sum(dim=-1).clamp(min=1e-6)
        avg_bb_plddt = masked_sum / mask_sum

        if method == 'avg':
            return avg_bb_plddt

        elif method == 'square':
            return torch.square(avg_bb_plddt)

        elif method == 'sigmoid':
            return torch.sigmoid(sharpness * (avg_bb_plddt - shift))

        elif method == 'threshold':
            result = (avg_bb_plddt >= threshold).float()
            return result
        
        else:
            raise NotImplementedError(f'Method {method} not implemented')
    
    def compute_loss(
        self,
        output,
        batch,
        _return_breakdown=False,
        _logger=BlackHole(),
        _log_prefix="train",
        loss_weights: Optional[Dict[str, Union[float, bool]]] = {
            "rmsd_loss_weight": 1.0,
            "rotamer_recovery_weight": 1.0,
            "chi_nll_loss_weight": 1.0,
            "offset_mse_loss_weight": 1.0,
            "chi_trig_huber_loss_weight": 1.0,
            "clash_loss_weight": 1.0,
            "proline_loss_weight": 1.0,
        }
    ):
        """Adapted from https://github.com/Kuhlman-Lab/PIPPack"""

        # Masks
        chi_mask = batch.chi_mask
        batch.residue_mask = torch.ones((batch.pos_CA.shape[0]), device=self.dummy.device)

        # Formatting output
        mode_chis = output["chis"]
        gumbel_sampled_chis = output["gumbel_sampled_chis"]
        bb_coords = torch.cat([
            batch.pos_N.unsqueeze(1),
            batch.pos_CA.unsqueeze(1),
            batch.pos_C.unsqueeze(1),
            batch.pos_O.unsqueeze(1)
        ], dim=1)
        output_coords = self.idealizer(batch.seq, bb_coords, gumbel_sampled_chis) * batch.atom14_mask.unsqueeze(-1)

        avg_bb_plddt = self._get_representative_bb_plddt(batch, method='avg')
        loss_fns = {
            # Chi angle-based
            "rotamer_recovery": lambda: rotamer_recovery_from_coords(
                batch.seq, batch.chis, mode_chis, 
                batch.residue_mask, chi_mask, # chi_num=None,
                _metric=_logger.get_metric(_log_prefix + " rotamer recovery")
            ),

            # Coordinate-based (using coords from Gumbel-sampled chis)
            "rmsd_loss": lambda: sc_rmsd(
                output_coords, batch.atom14_coords, batch.seq, 
                batch.atom14_mask, batch.residue_mask,
                avg_bb_plddt=avg_bb_plddt,
                use_sqrt=False,
                _metric=_logger.get_metric(_log_prefix + " rmsd")
            ),
            "clash_loss": lambda: local_interresidue_sc_clash_loss(
                batch, output_coords, 0.6
            )["mean_loss"],
            "proline_loss": lambda: unclosed_proline_loss(
                batch, output_coords
            )
        }
        
        if self.predict_binned_chis:
            # Retrieves outputs and labels
            true_bins, _, true_offsets = self._bin_and_offset_chis(batch.chis)
            chi_bin_log_probs = output["chi_bin_log_probs"]
            chi_offsets = output["chi_offsets"].squeeze(-1)
            
            # Updates loss functions
            loss_fns.update({
                "chi_nll_loss": lambda: nll_chi_loss(
                    chi_bin_log_probs, true_bins,
                    batch.seq, chi_mask.float(),
                    avg_bb_plddt=avg_bb_plddt,
                    _metric=_logger.get_metric(_log_prefix + " chi nll loss"))})
            loss_fns.update({
                "offset_mse_loss": lambda: offset_mse(
                    chi_offsets, true_offsets,
                    chi_mask.float(), self.num_chi_bins, False,
                    avg_bb_plddt=avg_bb_plddt,
                    _metric=_logger.get_metric(_log_prefix + " offset mse loss"))})
        else:
            # TODO: Adapt this whole else branch to work with predicting all chis
            # instead of only one chi
            loss_fns.update({
                "chi_trig_huber_loss": lambda: huber_loss(
                    embedded_chis, batch.chis, batch.chi_mask, chi_num_mask
                )[0]
            })
            raise NotImplementedError()
            
        total_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = loss_weights.get(loss_name + "_weight", 0.0)
            if weight == 0.0:
                continue
            loss = loss_fn()
            if (torch.isnan(loss) or torch.isinf(loss)):
                print(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            total_loss = total_loss + weight * loss
            losses[loss_name] = loss.detach().cpu().clone()
        losses["overall"] = total_loss.detach().cpu().clone()
        
        if not _return_breakdown:
            return total_loss
        
        return total_loss, losses

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./casp16_af3_predictions",
        help="Path to data directory"
    )
    args = parser.parse_args()

    dataset = PSCPDataset(root=args.data_dir)
    model = PSCPAllChisNetwork().to('cuda')

    batch = Batch.from_data_list([dataset[0], dataset[1], dataset[2], dataset[3]]).to('cuda')
    rot_glob = orthogonalize_matrix(torch.randn([1, 3, 3])).to('cuda')
    batch_rot = batch.clone()
    batch_rot.pos_CA = apply_rotation(rot_glob, batch.pos_CA)
    batch_rot.pos_C = apply_rotation(rot_glob, batch.pos_C)
    batch_rot.pos_N = apply_rotation(rot_glob, batch.pos_N)
    batch_rot.pos_O = apply_rotation(rot_glob, batch.pos_O)
    batch_rot.node_v = apply_rotation(rot_glob, batch.node_v)

    print(f'\nRunning inference on: {", ".join(name for name in batch.name)}')
    model.eval()
    output = model(batch)
    total_loss, losses = model.compute_loss(
        output, batch, _return_breakdown=True,
    )
    print(f'total_loss = {total_loss}')
    print(f'losses = {losses}')

    print('\nRunning inference on rotated batch')
    y_ref = output['chi_bin_logits']
    y_rot = model(batch_rot)['chi_bin_logits']

    with open(f'invariance_diffs.json', 'w') as file:
        diffs = list(torch.flatten((y_ref - y_rot).abs()))
        diffs = {'diffs': [float(diff) for diff in diffs]}
        json.dump(diffs, file)

    atol, rtol = 1e-5, 1e-4
    if torch.allclose(y_ref, y_rot, atol=atol, rtol=rtol):
        print('[Model] Passed invariance test.')
    else:
        print(
            '[Model] Failed invariance test: '
            f'{(y_ref - y_rot).abs().max().item()} exceeded abs. tol. {atol} or rel. tol. {rtol}'
        )    

    output_dir = './inference_outputs'
    batch = Batch.from_data_list([dataset[0]]).to('cuda')
    model.eval()
    input_path = os.path.join(args.data_dir, f'{batch.name[0]}.pdb')
    output_path = os.path.join(output_dir, f'{batch.name[0]}_repacked.pdb')
    os.makedirs(output_dir, exist_ok=True)
    model.sample_pdb(batch, input_path, output_path)
    print(f'\nWrote repacked protein to PDB file: {output_path}')
