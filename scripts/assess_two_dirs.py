from attnpacker.protein_learning.assessment.sidechain import assess_sidechains, summarize
import os
import json
import torch
from tqdm import tqdm
import glob
import argparse
from functools import partial
from joblib import Parallel, delayed


def results_for_single_target(
    pdb_file_name,
    targets_dir,
    predictions_dir,
    write_per_res_per_target_stats=False,
    write_per_target_stats=False,
    per_target_metrics_dir=None
):
    predicted_pdb = os.path.join(predictions_dir, pdb_file_name)
    target_pdb = os.path.join(targets_dir, pdb_file_name)
    res_level_stats = assess_sidechains(target_pdb_path=target_pdb, decoy_pdb_path=predicted_pdb, steric_tol_fracs = [1,0.9,0.8])
    target_level_stats = summarize(res_level_stats)

    target_name = os.path.splitext(pdb_file_name)[0]

    if write_per_res_per_target_stats:
        assert per_target_metrics_dir is not None, "Must provide per_target_metrics_dir to get per res stats"
        os.makedirs(per_target_metrics_dir, exist_ok=True)

        per_residue_stats_file_name = os.path.join(
            per_target_metrics_dir,
            f"{target_name}_per_residue.json"
        )
        with open(per_residue_stats_file_name, "w") as file:
            json.dump(tensor_to_python(res_level_stats), file, indent=2)

    if write_per_target_stats:
        assert per_target_metrics_dir is not None, "Must provide per_target_metrics_dir to get per target stats"
        os.makedirs(per_target_metrics_dir, exist_ok=True)

        per_target_stats_file_name = os.path.join(
            per_target_metrics_dir,
            f"{target_name}.json"
        )
        with open(per_target_stats_file_name, "w") as file:
            json.dump(tensor_to_python(target_level_stats), file, indent=2)

    return target_level_stats


def results_per_each_target(targets_dir, predictions_dir, per_target_metrics_dir=None):
    target_files = glob.glob(os.path.join(targets_dir, "*.pdb"))
    target_files = set([os.path.basename(filename) for filename in target_files])
    prediction_files = glob.glob(os.path.join(predictions_dir, "*.pdb"))
    prediction_files = set([os.path.basename(filename) for filename in prediction_files])
    assert prediction_files - target_files == set()

    process_target = partial(
        results_for_single_target,
        targets_dir=targets_dir,
        predictions_dir=predictions_dir,
        per_target_metrics_dir=per_target_metrics_dir,
        write_per_res_per_target_stats=True,
    )
    num_processes = os.cpu_count() // 2
    target_stats_list = Parallel(n_jobs=num_processes)(
        delayed(process_target)(target)
        for target in tqdm(prediction_files)
    )
        
    return target_stats_list


def aggregate_dataset_stats(target_stats_list):
    overall_stats = {}
    first_target_stats = target_stats_list[0]

    for centrality in ["all", "core", "surface"]:
        overall_stats[centrality] = dict()

        tensor_keys = ['rmsd', 'mae_sr', 'mean_mae']
        for key in tensor_keys:
            stacked = torch.stack([target[centrality][key] for target in target_stats_list]).float()
            overall_stats[centrality][key] = torch.mean(stacked, dim=0)
        
        overall_stats[centrality]['dihedral_counts'] = torch.sum(
            torch.stack([target[centrality]['dihedral_counts'] for target in target_stats_list]),
            dim=0
        )

        overall_stats[centrality]['num_sc'] = torch.sum(torch.tensor([target[centrality]['num_sc'] for target in target_stats_list]))
        overall_stats[centrality]['mean_seq_len'] = torch.mean(
            torch.tensor([target[centrality]['seq_len'] for target in target_stats_list]).float()
        )

    # These metrics are only given across all residues.
    overall_stats["all"]["ca_rmsd"] = torch.mean(
        torch.stack([target["all"]["ca_rmsd"] for target in target_stats_list]).float(),
        dim=0
    )
    overall_stats["all"]['clash_info'] = {
        threshold: {
            'energy': torch.mean(
                torch.stack([target["all"]['clash_info'][threshold]['energy'] for target in target_stats_list]).float(),
                dim=0
            ),
            'num_atom_pairs': torch.mean(
                torch.tensor([target["all"]['clash_info'][threshold]['num_atom_pairs'] for target in target_stats_list]).float()
            ),
            'num_clashes': torch.mean(
                torch.tensor([target["all"]['clash_info'][threshold]['num_clashes'] for target in target_stats_list]).float()
            ),
        } for threshold in first_target_stats["all"]['clash_info'].keys()
    }
    overall_stats["all"]["num_targets"] = len(target_stats_list)
    
    return overall_stats


def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim > 0 else obj.item()
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    
    return obj


def run_assess_dir(targets_dir, predictions_dir):
    # Computes metrics
    per_target = results_per_each_target(
        targets_dir=targets_dir,
        predictions_dir=predictions_dir,
        per_target_metrics_dir=predictions_dir
    )
    across_all_targets = aggregate_dataset_stats(target_stats_list=per_target)

    # Saves results
    converted = tensor_to_python(across_all_targets)
    stats_dir = predictions_dir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file_name = os.path.join(stats_dir, "stats.json")
    with open(stats_file_name, "w") as file:
        json.dump(converted, file, indent=4)
    print(f"Saved assessment results to {stats_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_dir',
        type=str,
        default='/home/common/proj/side_chain_packing/data/FINAL/structures/casp16/casp16_native'
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        default='/home/common/proj/side_chain_packing/code/CrossDistillationPSCP/inference_outputs/casp16_native/20250830_022819_3pt_rosetta_min'
    )
    args = parser.parse_args()

    run_assess_dir(args.target_dir, args.pred_dir)
