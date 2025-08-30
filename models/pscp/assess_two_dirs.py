from protein_learning.assessment.sidechain import assess_sidechains, summarize, debug
import pprint
import os
import json
import torch
from tqdm import tqdm
import glob
import argparse
import multiprocessing
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

    # if "T119" in pdb_file_name:
    #     debug(res_level_stats, "res_level_stats", False)

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
    
    # print(f'overall_stats = {pprint.pformat(overall_stats)}')
    return overall_stats

def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim > 0 else obj.item()
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    
    return obj

def run_assess_dir(targets_dir, predictions_dir):
    stats_dir = predictions_dir
    stats_file_name = "stats.json"

    per_target = results_per_each_target(
        targets_dir=targets_dir,
        predictions_dir=predictions_dir,
        per_target_metrics_dir=predictions_dir
    )
    across_all_targets = aggregate_dataset_stats(target_stats_list=per_target)

    converted = tensor_to_python(across_all_targets)
    os.makedirs(stats_dir, exist_ok=True)
    stats_file_name = os.path.join(stats_dir, stats_file_name)
    with open(stats_file_name, "w") as file:
        json.dump(converted, file, indent=4)
    print(f"Saved to {stats_file_name} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default=None)
    parser.add_argument('--pred_dir', type=str, default=None)
    args = parser.parse_args()

    targets_dir = args.target_dir if args.target_dir is not None \
        else f'/home/common/proj/side_chain_packing/data/bc40_dataset/by_chains' 
    predictions_dir = args.pred_dir if args.pred_dir is not None \
        else f'/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/bc40_dataset_train/20250426_195939_200pt'
    run_assess_dir(targets_dir, predictions_dir)

    # for targets_dir, predictions_dir in (
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_native/20250429_234711_chi1_novec_24pt'),
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_af2/20250429_234711_chi1_novec_24pt'),
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_af3/20250429_234711_chi1_novec_24pt'),
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_native/20250429_235249_chi1_27pt'),
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_af2/20250429_235249_chi1_27pt'),
    #     ('/home/common/proj/side_chain_packing/data/FINAL/casp15/casp15_native','/home/common/proj/side_chain_packing/code/OAGNN/inference_outputs/casp15_af3/20250429_235249_chi1_27pt'),
    # ):
    #     run_assess_dir(targets_dir, predictions_dir)

    pass
