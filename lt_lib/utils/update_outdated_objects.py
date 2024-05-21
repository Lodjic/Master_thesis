# Author: Loïc Thiriet

import copy
from pathlib import Path

from lt_lib.utils.load_and_save import load_pytorch_checkpoint, save_pytorch_checkpoint


def update_checkpoint_metrics_dict(checkpoint_path: Path) -> None:
    # Loads metrics from saved model checkpoint
    chkpt = load_pytorch_checkpoint(checkpoint_path)
    metrics = copy.deepcopy(chkpt["metrics"])

    # Process phases 1 by 1 (usually train and val)
    for phase in metrics.keys():
        # If there is no key named 'level1' it means the chkpt is outdated
        if "level1" not in metrics[phase].keys():
            # Constructs a value dict of the new level1 key
            level1_metric_dict = {
                "tp": copy.deepcopy(metrics[phase]["tp"]),
                "fp": copy.deepcopy(metrics[phase]["fp"]),
                "fn": copy.deepcopy(metrics[phase]["fn"]),
                "recall": copy.deepcopy(metrics[phase]["recall_lvl1"]),
                "precision": copy.deepcopy(metrics[phase]["precision_lvl1"]),
            }

            # Removes metrics from root of phase
            for metric in ["tp", "fp", "fn", "recall_lvl1", "precision_lvl1"]:
                del metrics[phase][metric]

            # Adds the new level1 metrics
            metrics[phase]["level1"] = level1_metric_dict

        if "comparable_loss" not in metrics[phase].keys():
            metrics[phase]["comparable_loss"] = copy.deepcopy(metrics[phase]["loss"])

        if "level2" not in metrics[phase].keys():
            metrics[phase]["level2"] = {
                "mAP50": [0] * len(copy.deepcopy(metrics[phase]["loss"])),
                "mAP": [0] * len(copy.deepcopy(metrics[phase]["loss"])),
            }

    chkpt["metrics"] = metrics
    save_pytorch_checkpoint(chkpt, checkpoint_path)
