# Author: Loïc Thiriet

import os
import shutil
from pathlib import Path
from typing import Any

import ray.train as ray_train
import wandb
from loguru import logger
from ray.air.integrations.wandb import setup_wandb

from lt_lib.optimization.update_configs import update_config_files
from lt_lib.orchestration.task_orchestrator import TaskOrchestrator
from lt_lib.schemas.config_files_schemas import ModelConfig, RunConfig
from lt_lib.utils.dict_utils import extend_flattened_dict, flatten_dict
from lt_lib.utils.load_and_save import (
    load_pytorch_checkpoint,
    load_yaml_as_dict,
    save_dict_as_yaml,
)
from lt_lib.utils.log import initialize_logger, log_task_begin_or_end
from lt_lib.utils.regex_matcher import get_elements_with_regex


def modify_trial_model_config_to_restore_from_checkpoint(trial_dir: Path) -> None:
    """
    Modifies the current trial model config file to restore training from the last saved checkpoint.

    Args:
        trial_dir: The directory containing the trial's info, notably the checkpoints and configuration files.

    Raises:
        ValueError: If no checkpoint is found in the trial directory.
    """
    ### Gets last saved checkpoint
    # Finds the last checkpoint in alphabetic order if there are more than 1 checkpoint
    checkpoint_list = list(trial_dir.rglob("*.tar"))
    if len(checkpoint_list) == 0:
        logger.error(f"Did not found any checkpoint in the trial directory: {trial_dir}.")
        raise ValueError(f"Did not found any checkpoint in the trial directory: {trial_dir}.")
    elif len(checkpoint_list) > 1:
        checkpoint_path = checkpoint_list[-1]
        logger.error(
            f"Found more than 1 checkpoint in the trial directory '{trial_dir}', taking the last one in "
            + f"alphabetic order: '{checkpoint_path}'."
        )
    else:
        checkpoint_path = checkpoint_list[0]

    # Gets list of yaml config path
    trial_configs = list(trial_dir.rglob("*yaml"))

    # For all config yaml found (because we cannot infer the model config name) check if there is a parameter
    # called 'saved_checkpoint_path'. If yes replace its value with checkpoint_path and if not do nothing.
    for trial_config_path in trial_configs:
        config = load_yaml_as_dict(trial_config_path)
        flattened_dict = flatten_dict(config)
        param_saved_checkpoint_path = get_elements_with_regex(
            f"^.*saved_checkpoint_path.*$", flattened_dict.keys(), unique=True, authorize_no_match=True
        )
        if param_saved_checkpoint_path:
            # Updates trial saved checkpoint path
            flattened_dict[param_saved_checkpoint_path] = str(checkpoint_path)
            # Loads checkpoint for verification purposes
            param_model_loading_type = get_elements_with_regex(
                f"^.*model_loading_type.*$", flattened_dict.keys(), unique=True, authorize_no_match=False
            )
            chkpt = load_pytorch_checkpoint(checkpoint_path)
            # Updates trial model loading type depending if a model_state_dict or model is available
            if chkpt["model_state_dict"]:
                flattened_dict[param_model_loading_type] = "model_state_dict"
            else:
                flattened_dict[param_model_loading_type] = "model"
            # Extend flattened config dict to get back to its previous shape and save it
            config = extend_flattened_dict(flattened_dict)
            save_dict_as_yaml(config, trial_config_path)


def creates_trial_configs_from_sampled_parameters(
    base_config_path: Path,
    base_model_config_path: Path | None,
    sampled_params: dict[str, Any],
    trial_dir: Path,
    trial_output_dir: Path,
):
    """
    Creates the trial configuration files from the default configuration files and sampled parameters.

    Args:
        base_config_path: The path to the base/default configuration file.
        base_model_config_path: The path to the base model configuration file.
        sampled_params: A dictionary containing the sampled parameters.
        trial_dir: The trial directory path.
        trial_output_dir: The trial output directory in which to also copy the trial configurations.

    Returns:
        A tuple containing the paths of the trial configuration file and model configuration file.
    """
    ### Creates updated version of the trial configs for the ray_results directory
    # Creates a trial config and model config path in the ray_results directory
    trial_config_dir = trial_dir / "configs"
    trial_config_path = trial_config_dir / base_config_path.name
    if base_model_config_path:
        trial_model_config_path = trial_config_dir / base_model_config_path.name
    else:
        trial_model_config_path = base_model_config_path

    # If the trial config directory does not exists, then creates it and updates the trial config files
    # If it already exists, it means the the optimization is being restored
    if not os.path.isdir(trial_config_dir):
        os.mkdir(trial_config_dir)

        # Copies the base config file to the ray_results directory
        shutil.copy2(base_config_path, trial_config_path)

        # Copies the base model config file to the ray_results directory
        if base_model_config_path:
            shutil.copy2(base_model_config_path, trial_model_config_path)

        # Updates config files in the ray_results directory with sampled params
        update_config_files(sampled_params, trial_config_path, trial_model_config_path)

    else:
        logger.info("Trial seems to be restoring from a previous run because trial config files were found.")
        modify_trial_model_config_to_restore_from_checkpoint(trial_dir)
        logger.info(
            "Trial saved_checkpoint_path from model_config was changed to the checkpoint found in the trial directory."
        )

    ### Copies the  newly updated config files from the ray_results directory to the outputs directory (local folder)
    # Creates a trial configs directory in the outputs directory
    trial_output_config_dir = trial_output_dir / "configs"
    if not os.path.isdir(trial_output_config_dir):
        os.makedirs(trial_output_config_dir)

    # Copies config file from the ray_results directory to the outputs directory
    shutil.copy2(trial_config_path, trial_output_config_dir / trial_config_path.name)

    # Copies model config file from the ray_results directory to the outputs directory
    if base_model_config_path:
        shutil.copy2(trial_model_config_path, trial_output_config_dir / trial_model_config_path.name)

    return trial_config_path, trial_model_config_path


def objective(sampled_params: dict[str, Any], use_wandb_ray_integration: bool, cli_args: dict) -> None:
    """
    Defines the objective function to be optimized.

    Args:
        sampled_params: Sampled parameters for the current trial.
        use_wandb_ray_integration: Whether to use Weights & Biases integration within Ray.
        cli_args: Dictionary of the command line arguments.
    """
    # Get the train context
    trial_id = ray_train.get_context().get_trial_id()
    trial_name = ray_train.get_context().get_trial_name()
    experiment_name = ray_train.get_context().get_experiment_name()
    trial_dir = Path(ray_train.get_context().get_trial_dir())

    # Intitalizes the logger
    initialize_logger(Path(cli_args.outputs_directory), cli_args.console_log_level)
    # Logging beginning of trial
    log_task_begin_or_end(f"Trial-{trial_id}", "begin", "upper")

    # Get configs path
    base_config_path = Path(cli_args.config_path)
    base_model_config_path = Path(cli_args.model_config_path) if cli_args.model_config_path != "" else None
    trial_outputs_dir = Path(cli_args.outputs_directory) / f"outputs_trial-{trial_id}"

    # Creates trial configs based on the sampled parameters
    trial_config_path, trial_model_config_path = creates_trial_configs_from_sampled_parameters(
        base_config_path,
        base_model_config_path,
        sampled_params,
        trial_dir,
        trial_outputs_dir,
    )

    # Initializes wandb
    if use_wandb_ray_integration:
        # Gets parent dir of the experiment dir (ie: the ray_results_dir)
        ray_results_dir = trial_dir.parent.parent
        # Gets parameters for wandb.init() fct
        wandb_init_params = RunConfig.from_yaml(trial_config_path).wandb_init_params
        setup_wandb(
            config={k.split(".")[-1]: v for k, v in sampled_params.items()},
            rank_zero_only=False,
            dir=ray_results_dir,
            id=trial_name,
            name=trial_name,
            group=wandb_init_params["group"] if "group" in wandb_init_params.keys() else None,
            project=wandb_init_params["project"] if "project" in wandb_init_params.keys() else experiment_name,
            resume=wandb_init_params["resume"] if "resume" in wandb_init_params.keys() else "auto",
        )

    # Instantiates TaskOrchestrator and launch it
    task_orchestrator = TaskOrchestrator(
        inputs_dir=Path(cli_args.inputs_directory),
        outputs_dir=trial_outputs_dir,
        config_path=trial_config_path,
        model_config_path=trial_model_config_path,
        manually_init_wandb=False,
    )
    task_orchestrator.run()

    # Reading, metrics extraction and reporting to ray.train is done in the at the end of the tasks. Especially, only
    # train and eval tasks report to ray.train
    if wandb.run:
        wandb.finish()

    # Loggin end of trial
    log_task_begin_or_end(f"Trial-{trial_id}", "end", "upper")
