'''
# FOR TESTING MEMORIZATION
export CUDA_VISIBLE_DEVICES=1; nohup \
    python -m models.train_allchis --tag testing_memorization \
    --testing_memorization \
    &> nohup_logs/testing_memorization.log &

# FOR NORMAL TRAINING
export CUDA_VISIBLE_DEVICES=0,1,2,3; nohup \
    python -m torch.distributed.run --nproc_per_node=4 --master_port=29584 \
    -m models.train_allchis --ddp --tag crossdist_finetuningcoords \
    --resume_from /home/common/proj/side_chain_packing/code/OAGNN/logs/crossdist_modifications_sharpsigmoid/checkpoints/144.pt \
    > nohup_logs/crossdist_finetuningcoords.log 2>&1 &
'''

# Deals with command-line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='/home/common/proj/side_chain_packing/code/CrossDistillationPSCP/models/configs/svp_gnn.yml'
)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--testing_memorization', action='store_true', default=False)
parser.add_argument('--no_vec', action='store_true', default=False, help='If set, disables vector features')
# parser.add_argument('--cuda_visible_devices', type=str, default='0')
parser.add_argument('--ddp', action='store_true', help='Enable distributed training')
# parser.add_argument('--local_rank', type=str, default='0') # Only for torch.distributed.launch
args = parser.parse_args()

# Python-native imports
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
print(f'CUDA_VISIBLE_DEVICES = {os.environ["CUDA_VISIBLE_DEVICES"]}')
import shutil
from easydict import EasyDict
import traceback

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# These are imports from this repo
from oagnn_utils.misc import BlackHole, get_logger, get_new_log_dir, load_config, seed_all, Counter
from oagnn_utils.train import get_optimizer, get_scheduler, log_losses
from models.datasets import PSCPDataset, CombinedDataset
from models.models import PSCPAllChisNetwork
from protein_learning.protein_utils.sidechains.sidechain_rigid_utils import atom37_to_torsion_angles
from protein_learning.common.helpers import safe_normalize
from models.loss_fns import trig_loss, huber_loss


def check_cuda_memory():
    torch.cuda.empty_cache()
    print(f'torch.cuda.memory_allocated() = {torch.cuda.memory_allocated()}')
    print(f'torch.cuda.memory_reserved() = {torch.cuda.memory_reserved()}')


def setup_ddp(args):
    if args.ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"]) # Env var is set by torchrun
        # local_rank = int(args.local_rank) # For use with torch.distributed.launch
        torch.cuda.set_device(local_rank)
        print(f'Did set_device() with local rank = {local_rank}')
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        print(f'rank = {args.rank}, local rank = {local_rank}, world size = {args.world_size}')
        args.device = f"cuda:{local_rank}"
        args.is_main_process = args.rank == 0
        args.local_rank = local_rank
    else:
        args.rank = 0
        # args.device is already set through argparse
        args.is_main_process = True
        args.local_rank = 0
    return args


def cleanup_ddp(args):
    if args.ddp:
        dist.destroy_process_group()


# Load configs
args = setup_ddp(args)
config, config_name = load_config(args.config)
seed_all(config.train.seed)


# Logging
if args.debug:
    logger = get_logger(config_name, None)
    writer = BlackHole()
else:
    if args.resume_from is not None and args.overwrite:
        log_dir = os.path.dirname(os.path.dirname(args.resume_from))
    else:
        log_dir = get_new_log_dir(args.logdir, tag=args.tag, name=args.run_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)
        shutil.copytree('./modules', os.path.join(log_dir, 'modules'), dirs_exist_ok=True)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(f'args = {args}')
    logger.info(f'config = {config}')


def get_data_loaders(args):
    if args.is_main_process:
        logger.info('Loading datasets...')

    train_set = PSCPDataset(config.data.train_val_dir, 'train')
    val_set = PSCPDataset(config.data.train_val_dir, 'val')

    if args.testing_memorization:
        train_set.dataset = train_set.dataset[:1]

    # Sets up samplers
    train_sampler = DistributedSampler(train_set, shuffle=True) if args.ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.ddp else None

    # Sets up data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.data.train_batch_size,
        sampler=train_sampler,
        shuffle=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.data.val_batch_size,
        sampler=val_sampler,
        shuffle=False
    )

    if args.is_main_process:
        logger.info('Train: %d | Validation: %d' % (len(train_set), len(val_set)))
    return train_loader, val_loader


def get_model(args):
    if args.is_main_process:
        logger.info('Building model...')
    print(f"[Rank {args.rank}] Creating PSCPAllChisNetwork...")
    print(f"[Rank {args.rank}] Device: {args.device}")
    print(f"[Rank {args.rank}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Rank {args.rank}] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Rank {args.rank}] Current CUDA device: {torch.cuda.current_device()}")
    
    model = PSCPAllChisNetwork(
        no_vec=args.no_vec,
        top_k=config.model.top_k,
        conv=config.model.convolution_mode
    )
    
    print(f"[Rank {args.rank}] Model created, moving to device {args.device}...")
    model = model.to(args.device)
    print(f"[Rank {args.rank}] Model moved to device successfully")
    print(f'model top k = {model.top_k}')
    if args.ddp:
        # for name, param in model.named_parameters():
        #     print(f"[Rank {args.rank}] {name} shape: {tuple(param.shape)}")
        print(f'about to do barrier')
        dist.barrier()

        # Local rank is non-null when ddp is set to true
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    return model


def train(
    it,
    model,
    loss_weights,
    train_loader,
    optimizer, 
    train_losses,
    train_batch_losses,
    global_step
):
    model.train()
    loss_sum_across_batches = 0.0
    batches = enumerate(tqdm(train_loader, desc='Training', position=0, leave=True)) \
        if args.is_main_process \
        else enumerate(train_loader)
    for i, batch in batches:
        batch = batch.to(args.device)
        optimizer.zero_grad()

        try:
            output = model(batch)   # (N, 2)
            if not args.ddp:
                loss, loss_breakdown = model.compute_loss(
                    output, batch, loss_weights=loss_weights, _return_breakdown=True)
            else:
                loss, loss_breakdown = model.module.compute_loss(
                    output, batch, loss_weights=loss_weights, _return_breakdown=True)
            # print(f'loss weight s= {loss_weights}')
            # print(f'loss = {loss}')
            # print(f'loss.requires_grad = {loss.requires_grad}')
            # print(f'type(loss) = {type(loss)}')
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            # Just for recording, not for backprop
            num_residues = batch.chi_mask.float()[:, 0].sum().item()
            loss_sum_across_batches += loss

            batch_loss_tensor = torch.tensor([loss.item(), num_residues], device=args.device)
            batch_idx = len(train_loader) * (it - 1) + i + 1
            train_batch_losses.append((batch_idx, float(batch_loss_tensor[0])))
            
            if args.is_main_process:
                log_others = {
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log_losses(
                    EasyDict(loss_breakdown),
                    global_step.step(), 'train', logger=logger, writer=writer,
                    others=log_others
                )
            
        except Exception as e:
            print(f"Error occurred during training: {e}")
            traceback.print_exc()
            raise
    
    # to_tensor = lambda x : torch.tensor(x, dtype=torch.float64)
    # losses, sum_residues = map(to_tensor, (losses, sum_residues))
    epoch_loss_tensor = torch.tensor([loss_sum_across_batches, len(train_loader)], device=args.device)
    if args.ddp:
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
    avg_train_loss = epoch_loss_tensor[0] / epoch_loss_tensor[1]
    train_losses.append((it, avg_train_loss.item()))
    return avg_train_loss


def validate(
    it,
    model,
    loss_weights,
    val_loader,
    optimizer,
    val_losses,
    global_step
):
    model.eval()
    loss_sum_across_batches = 0.0
    with torch.no_grad():
        batches = enumerate(tqdm(val_loader, desc="Running validation", position=0, leave=True)) \
            if args.is_main_process \
            else enumerate(val_loader)
        for i, batch in batches:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            try:
                output = model(batch)
                if args.ddp:
                    loss, loss_breakdown = model.module.compute_loss(
                        output, batch, loss_weights=loss_weights, _return_breakdown=True)
                else:
                    loss, loss_breakdown = model.compute_loss(
                        output, batch, loss_weights=loss_weights, _return_breakdown=True)

                # Only counts residues with at least chi1
                num_residues = batch.chi_mask.float()[:, 0].sum().item()
                loss_sum_across_batches += loss

                if args.is_main_process:
                    log_losses(
                        EasyDict(loss_breakdown),
                        global_step.step(), 'validation', logger=logger, writer=writer,
                        others={}
                    )

            except Exception as e:
                print(f"Error occurred during validation: {e}")
                traceback.print_exc()
                raise

        val_epoch_loss_tensor = torch.tensor([loss_sum_across_batches, len(val_loader)], device=args.device)
        if args.ddp:
            dist.all_reduce(val_epoch_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_epoch_loss_tensor[0] / val_epoch_loss_tensor[1]
        val_losses.append((it, avg_val_loss.item()))
        return avg_val_loss


def save_plots(iter, train_losses, val_losses, train_batch_losses, train_batches, ckpt_dir):
    if args.testing_memorization:
        val_losses = train_losses

    # First plot
    train_loss_x, train_loss_y = zip(*train_losses)
    plt.scatter(train_loss_x, train_loss_y, marker='.', linestyle='-', color='r',
                zorder=2, label='Training loss')
    val_loss_x, val_loss_y = zip(*val_losses)
    # print(f'val loss x = {val_loss_x}')
    # print(f'val loss y = {val_loss_y}')
    plt.scatter(val_loss_x, val_loss_y, marker='.', linestyle='-', color='g',
                zorder=1, label='Validation loss')

    plt.xlabel('Epoch number')
    plt.ylabel('Per-residue loss across dataset')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(ckpt_dir, f'{iter}_train_val_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Second plot
    train_batch_loss_x, train_batch_loss_y = zip(*train_batch_losses)
    plt.scatter(train_batch_loss_x, train_batch_loss_y, marker='.', linestyle='-', color='b',
                zorder=1, label='Training loss across batch')
    train_dataset_loss_x = [train_batches * x for x in train_loss_x]
    plt.scatter(train_dataset_loss_x, train_loss_y, marker='.', linestyle='-', color='r',
                zorder=3, label='Training loss across dataset')
    val_dataset_loss_x = [train_batches * x for x in val_loss_x]
    plt.scatter(val_dataset_loss_x, val_loss_y, marker='.', linestyle='-', color='g',
                zorder=2, label='Validation loss across dataset')

    plt.xlabel('Batch number')
    plt.ylabel('Per-residue loss on batch')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(ckpt_dir, f'{iter}_train_batch_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()


def linear_anneal_clamped(epoch, start_epoch=1, end_epoch=30, start_weight=0.0002, end_weight=0.002):
    if epoch <= start_epoch:
        return start_weight
    elif epoch >= end_epoch:
        return end_weight
    else:
        slope = (end_weight - start_weight) / (end_epoch - start_epoch)
        return start_weight + slope * (epoch - start_epoch)


def get_loss_weights(epoch=1):
    loss_weights = {
        # Chi angle-based loss terms
        "chi_nll_loss_weight": 1.0,
        "offset_mse_loss_weight": 100.0,
        "rotamer_recovery_weight": 0.0,
        
        # Coordinate-based loss terms
        # "rmsd_loss_weight": linear_anneal_clamped(epoch=epoch),
        "rmsd_loss_weight": 0.01,
        "clash_loss_weight": 0.0, # 1000.0,
        "proline_loss_weight": 0.0, # 1.0,

        # If not doing bin prediction
        "chi_trig_huber_loss_weight": 1.0,
    }
    return loss_weights


def training_loop(
    it_first,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    global_step,
    train_batch_losses=[],
    train_losses=[],
    val_losses=[],
    patience=20,
):
    # Main training loop
    try:
        best_val_loss, epochs_without_improvement = 1e9, 0
        epochs = range(it_first, config.train.max_epochs+1)
        if args.is_main_process:
            epochs = tqdm(epochs, desc="Training epochs")
        for it in epochs:
            if args.ddp:
                train_loader.sampler.set_epoch(it)
                val_loader.sampler.set_epoch(it)

            loss_weights = get_loss_weights(epoch=it)
            avg_train_loss = train(
                it, model, loss_weights, train_loader, optimizer,
                train_losses, train_batch_losses, global_step
            )
            if it % config.train.val_freq == 0:
                # Computes validation loss
                if not args.testing_memorization:
                    avg_val_loss = validate(
                        it, model, loss_weights, val_loader, optimizer,
                        val_losses, global_step
                    )
                    
                # Finishes and logs epoch
                scheduler.step()
                if not args.debug and args.is_main_process:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.module.state_dict() if args.ddp else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'train_batch_losses': train_batch_losses,
                        'avg_train_losses': train_losses,
                        'avg_val_losses': val_losses,
                    }, ckpt_path)
                    save_plots(
                        it, train_losses, val_losses, train_batch_losses,
                        len(train_loader), ckpt_dir
                    )
                    
                # Early stopping logic
                if not args.testing_memorization:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            if args.is_main_process:
                                logger.info(f"Early stopping at epoch {it} (no validation improvement for {patience} epochs).")
                            break
    except KeyboardInterrupt:
        if args.is_main_process:
            logger.info('Terminating...')
    finally:
        if args.is_main_process:
            last_graph_path = os.path.abspath(os.path.join(ckpt_dir, f'{it}_train_val_losses.png'))
            if os.path.exists(last_graph_path):
                logger.info(f'Summary of train/val results at {last_graph_path}')


def main():
    global args
    # args = setup_ddp(args)

    check_cuda_memory()
    train_loader, val_loader = get_data_loaders(args)
    model = get_model(args)
    global_step = Counter()
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # Resume from a checkpoint
    it_first = 1
    train_batch_losses, train_losses, val_losses = [], [], []
    if args.resume_from is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume_from)
        ckpt = torch.load(args.resume_from, map_location=args.device)
        it_first = ckpt['iteration'] + 1
        model.load_state_dict(ckpt['model']) if not args.ddp else model.module.load_state_dict(ckpt['model'])
        # logger.info('Resuming optimizer and scheduler states...')
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        # train_batch_losses = ckpt.get('train_batch_losses', [])
        # train_losses = ckpt.get('train_losses', [])
        # val_losses = ckpt.get('val_losses', [])

    training_loop(
        it_first, model, train_loader, val_loader, optimizer, scheduler, global_step,
        train_batch_losses, train_losses, val_losses
    )
    cleanup_ddp(args)


if __name__ == '__main__':
    main()
