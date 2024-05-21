# Author : Loïc Thiriet

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from ruamel.yaml import YAML

################################ Common files ################################


def load_json_as_dict(file_path: Path) -> dict[Any, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_dict_as_json(file_path: Path, dictionary: dict) -> None:
    with open(file_path, "w") as f:
        json.dump(dictionary, f)


def load_yaml_as_dict(file_path: Path):
    with file_path.open() as yaml_file:
        content_dict = YAML(typ="safe", pure=True).load(yaml_file)
    return content_dict


def save_dict_as_yaml(dict_to_save: dict[str, Any], file_path: Path):
    with file_path.open("wb") as yaml_file:
        YAML(typ="safe", pure=True).dump(dict_to_save, yaml_file)


################################ Model checkpoints ################################


def load_pytorch_checkpoint(file_path: Path) -> dict[str, Any]:
    # Loads saved model checkpoint
    checkpoint = torch.load(file_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return checkpoint


def save_pytorch_checkpoint(checkpoint_dict: dict[str, Any], checkpoint_path: Path):
    torch.save(checkpoint_dict, checkpoint_path)


def save_pytorch_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, dict[str, list[float]]] | None,
    model_saving_type: str,
    save_optimizer: bool,
    saving_path: Path,
) -> None:
    # Constructs checkpoint dict with epoch, model, metrics and optimizer info
    saving_dict = {"epoch": copy.deepcopy(epoch)}

    if model_saving_type == "model":
        saving_dict["model"] = copy.deepcopy(model)
        saving_dict["model_state_dict"] = None
    elif model_saving_type == "model_state_dict":
        saving_dict["model"] = None
        saving_dict["model_state_dict"] = copy.deepcopy(model.state_dict())
    else:
        saving_dict["model"] = copy.deepcopy(model)
        saving_dict["model_state_dict"] = copy.deepcopy(model.state_dict())

    saving_dict["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict()) if save_optimizer else None
    saving_dict["metrics"] = copy.deepcopy(metrics) if metrics else None

    # Saves the checkpoint
    if not os.path.isdir(saving_path.parent):
        os.makedirs(saving_path.parent)
    save_pytorch_checkpoint(saving_dict, saving_path)
