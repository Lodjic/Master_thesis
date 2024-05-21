# Author : Loïc Thiriet

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import wandb
from loguru import logger

################################ Loguru logging functions ################################


def initialize_logger(saving_dir: Path, console_log_level: str = "DEBUG"):
    logger.remove()
    logger.add(sys.stderr, level=console_log_level, colorize=True, backtrace=True, diagnose=True, enqueue=True)
    logger.add(saving_dir / "log_file.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True, enqueue=True)


def log_task_begin_or_end(task_name: str, begin_or_end: str, lower_or_upper: Literal["lower", "upper"]):
    if lower_or_upper == "upper":
        logger.info(f">>>>>>>>>> {begin_or_end.upper()} {task_name.upper()} <<<<<<<<<<\n")
    if lower_or_upper == "lower":
        logger.info(f">>>>>>>>>> {begin_or_end.capitalize()} {task_name.lower()} <<<<<<<<<<\n")


def logging_end_of_training(optimizer, num_epochs, metrics):
    optimizer_params = optimizer.state_dict()["param_groups"][0]
    del optimizer_params["params"]
    logger.info(
        f"""The model was trained on {num_epochs} epochs with params {optimizer_params} :
    - min_train_loss = {min(metrics["train"]["loss"]):.5f} at epoch {np.argmin(metrics["train"]["loss"])}
    - min_val_loss = {min(metrics["val"]["loss"]):.5f} at epoch {np.argmin(metrics["val"]["loss"])}
    - max_train_recall = {max(metrics["train"]["level1"]["recall"]) * 100:.1f}% at epoch {np.argmax(metrics["train"]["level1"]["recall"])}
    - max_train_precision = {max(metrics["train"]["level1"]["precision"]) * 100:.1f}% at epoch {np.argmax(metrics["train"]["level1"]["precision"])}
    - max_val_recall = {max(metrics["val"]["level1"]["recall"]) * 100:.1f}% at epoch {np.argmax(metrics["val"]["level1"]["recall"])}
    - max_val_precision = {max(metrics["val"]["level1"]["precision"]) * 100:.1f}% at epoch {np.argmax(metrics["val"]["level1"]["precision"])}\n"""
    )


################################ Wandb logging functions ################################


def wandb_log_config_files(config_path: Path, model_config_path: Path):
    configs_artifact = wandb.Artifact(name="config_files", type="configs")
    configs_artifact.add_file(config_path)
    configs_artifact.add_file(model_config_path)
    wandb.log_artifact(configs_artifact)


def wandb_log_model(chkpt_file_path: str | Path, artifact_type: str, artifact_name: str):
    epoch_model_artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    epoch_model_artifact.add_file(chkpt_file_path)
    wandb.log_artifact(epoch_model_artifact)
