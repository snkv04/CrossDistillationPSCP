"""
python -m models.inference_allchis \
    --checkpoint_dir 20250829_221150_asdf \
    --checkpoint best_val
"""

import os
import torch
import argparse
import re
import sys

from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.datasets import PSCPDataset
from models.models import PSCPAllChisNetwork
from utils.misc import load_config, seed_all
from utils.train import get_optimizer, get_scheduler

# Parses args
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./models/configs/svp_gnn.yml')
parser.add_argument('--checkpoint_dir', default='svp_gnn_2025_04_14__03_16_36')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--testing_memorization', action='store_true', default=False)
args = parser.parse_args()

# Loads config and checkpoint
config, config_name = load_config(args.config)
seed_all(config.train.seed)

# Loads states from checkpoint
def get_pt(checkpoint_dir, which_ckpt='best_val'):
    folder_path = f'./logs/{checkpoint_dir}/checkpoints'
    assert os.path.isdir(folder_path), f'Invalid checkpoint directory: {folder_path}'

    path = os.path.join(folder_path, which_ckpt)
    if os.path.exists(path):
        # Use provided checkpoint file if given
        ckpt = torch.load(path)
        return path, ckpt
    
    elif which_ckpt == 'latest':
        pattern = re.compile(r"^\d+\.pt$")  # Matches files like 1.pt, 42.pt, etc.
        matching_files = []
        for filename in tqdm(os.listdir(folder_path), desc="Finding a checkpoint"):
            if pattern.match(filename):
                full_path = os.path.join(folder_path, filename)
                matching_files.append(full_path)

        if not matching_files:
            raise Exception(f'Invalid checkpoint directory: {checkpoint_dir}')
        latest_file = max(matching_files, key=os.path.getmtime)
        ckpt = torch.load(latest_file)
        return latest_file, ckpt

    elif which_ckpt == 'best_val':
        curr_val_loss, curr_path, curr_ckpt = 1e9, None, None
        pattern = re.compile(r"^\d+\.pt$")  # Matches files like 1.pt, 42.pt, etc.
        matching_files = []
        for filename in tqdm(os.listdir(folder_path), desc="Finding a checkpoint"):
            if pattern.match(filename):
                full_path = os.path.join(folder_path, filename)
                ckpt = torch.load(full_path)
                if ckpt['avg_val_losses'][-1][1] < curr_val_loss:
                    curr_val_loss = ckpt['avg_val_losses'][-1][1]
                    curr_path = os.path.abspath(full_path)
                    curr_ckpt = ckpt

        if not curr_ckpt:
            raise ValueError(f'No checkpoints found in directory: {folder_path}')
        else:
            return curr_path, curr_ckpt

    else:
        raise ValueError(f'Invalid value for which_ckpt: {which_ckpt}')

# Loads states from checkpoint
if args.testing_memorization:
    args.checkpoint = 'latest'
get_default = lambda a, b: a if a is not None else b
checkpoint_path, checkpoint = get_pt(
    args.checkpoint_dir,
    get_default(args.checkpoint, 'best_val')
)
print(f'Using checkpoint: {checkpoint_path}')
model = PSCPAllChisNetwork(
    top_k=config.model.top_k,
    conv=config.model.convolution_mode,
    separate_dense_layers=True,
    separate_offset_layers=True).to(args.device)
model.load_state_dict(checkpoint['model'])
optimizer = get_optimizer(config.train.optimizer, model)
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler = get_scheduler(config.train.scheduler, optimizer)
scheduler.load_state_dict(checkpoint['scheduler'])

# Constructs dataset
if args.testing_memorization:
    pairs = [(
        config.data.train_val_dir,
        f'./inference_outputs/{os.path.basename(config.data.train_val_dir)}/{args.checkpoint_dir}_{os.path.basename(checkpoint_path).replace(".", "")}'
    )]
else:
    pairs = [
        (f'/home/common/proj/side_chain_packing/data/FINAL/structures/{casp}/{casp}_{input_type}{"_predictions" if input_type != "native" else ""}',
            f'./inference_outputs/{casp}_{input_type}/{args.checkpoint_dir}_{os.path.basename(checkpoint_path).replace(".", "")}')
            for casp in ("casp14", "casp15", "casp16")
            for input_type in ("native", "af2", "af3")
    ]

for data_dir, output_dir in pairs:
    # Loads in the dataset for inference
    if args.testing_memorization:
        dataset = PSCPDataset(root=data_dir, subset='train')
        dataset.dataset = dataset.dataset[:1]
    else:
        dataset = PSCPDataset(root=data_dir)
    data_loader = DataLoader(dataset, batch_size=config.data.test_batch_size)

    # Runs inference
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(data_loader, desc="Testing")):
            input_file = os.path.join(data_dir, f'{batch.name[0]}.pdb')
            output_file = os.path.join(output_dir, f'{batch.name[0]}.pdb')
            model.sample_pdb(batch, input_file, output_file)
    print(f'Wrote predictions into {os.path.abspath(output_dir)}')
