# Author: Loïc Thiriet

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lt_lib.core.initialize_model import initialize_model
from lt_lib.data.datasets import generate_dataloaders
from lt_lib.schemas.config_files_schemas import ModelConfig
from lt_lib.schemas.task_schemas import PredictTaskConfig, TaskSchema, TrainTaskConfig
from lt_lib.utils.dict_utils import flatten_dict
from lt_lib.utils.load_and_save import (
    load_pytorch_checkpoint,
    save_dict_as_json,
    save_pytorch_model_checkpoint,
)

######################################
### Parameters extraction function ###
######################################


def extract_task_init_params(
    task_config: TrainTaskConfig | PredictTaskConfig, model_config: ModelConfig | None
) -> dict[str, Any]:
    """
    Extracts the different initialization parameters from a task config and eventually a model config.

    Args:
        task_config: Configuration object for the task, either `TrainTaskConfig` or `PredictTaskConfig`.
        model_config: Configuration object for the model, defaults to None.

    Returns:
        init_params_dict: A dictionary containing the initialization parameters for the task, eventually the model
            initialization parameters, the dataloaders initialization parameters, the optimizer initialization
            parameters and the scheduler initialization parameters.

    Note:
        The returned dictionary includes the following keys:
        - 'model_init_params': Initialization parameters for the model.
        - 'dataloaders_init_params': Initialization parameters for dataloaders.
        - 'optimizer_init_params': Initialization parameters for the optimizer.
        - 'lr_scheduler_init_params': Initialization parameters for the learning rate scheduler.
        - 'task_params': Task-specific parameters excluding parameters related to model, optimizer, and scheduler.
    """
    init_params_dict = {}

    # Get model initialization params if a model is passed
    if model_config:
        init_params_dict["model_init_params"] = model_config.model_dump(exclude={"config_type", "model"})
        init_params_dict["model_init_params"].update(
            {"model": model_config.model.model_type, "model_parameters": model_config.model.parameters.model_dump()}
        )
    else:
        init_params_dict["model_init_params"] = {}

    # Get dataloaders initialization params
    init_params_dict["dataloaders_init_params"] = task_config.model_dump(
        include={
            "images_extension",
            "batch_size",
            "manual_seed",
        }
    )

    # Get optimizer initialization params
    init_params_dict["optimizer_init_params"] = task_config.model_dump(
        include={
            "optimizer",
            "optimizer_params",
        }
    )

    # Get scheduler
    init_params_dict["lr_scheduler_init_params"] = task_config.model_dump(
        include={
            "lr_scheduler",
            "lr_scheduler_params",
        }
    )

    # Get the task params
    init_params_dict["task_params"] = task_config.model_dump(
        exclude={
            "images_extension",
            "batch_size",
            "optimizer",
            "optimizer_params",
            "lr_scheduler",
            "lr_scheduler_params",
        }
    )

    return init_params_dict


#######################################
### Parameters overriding functions ###
#######################################


def override_saved_model_path(
    inputs_dir: Path, root_outputs_dir: Path, model_input: str, model_init_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Overrides the saved checkpoint path depending on model inputs model intialization parameters.

    Args:
        inputs_dir: Directory path containing the input data.
        root_outputs_dir: Outputs directory root path.
        model_input: Input string specifying model loading type, saved model directory, and model suffix.
            Format: "<model_loading_type>:<saved_model_directory>:<model_suffix>".
        model_init_params: Dictionary containing the model initialization parameters.

    Returns:
        dict[str, Any]: Updated model initialization parameters after overriding the saved model path.

    Notes:
        - If the model_input indicates that the model is in 'inputs_directory', the function looks for the
            'saved_checkpoint_path' key in the 'model_init_params'. If it is empty, it looks for a saved model in
            the 'inputs_dir'.
        - If the model_input does not indicates that the model is in 'inputs_directory', it assumes the model is saved
            in the output of another task and search for it.
    """
    model_input = model_input.split(":")
    saved_model_dir = model_input[1]
    model_suffix = model_input[2] if len(model_input) == 3 else "best"

    # If model is saved in the inputs_dir we override the saved_model_path only if it was not passed in the config
    if saved_model_dir == "inputs_directory":
        if model_init_params["saved_checkpoint_path"] is None:
            model_init_params["saved_checkpoint_path"] = str(list(inputs_dir.glob("*.tar"))[0])
    # Else we suppose it's saved in the output of another task
    else:
        model_init_params["saved_checkpoint_path"] = str(
            list((root_outputs_dir / saved_model_dir).rglob(f"*{model_suffix}.tar"))[0]
        )

    model_init_params["model_loading_type"] = model_input[0]
    model_init_params["weights_backbone"] = None

    return model_init_params


def override_dataframe_path(
    inputs_dir: Path, root_outputs_dir: Path, df_input: str, task_params: dict[str, Any]
) -> dict[str, Path]:
    """
    Overrides the dataframe path based on input parameters.

    Args:
        inputs_dir: Directory path containing the input data.
        root_outputs_dir: Outputs directory root path.
        df_input: A string representing the type, the directory name and the function (train, val or test) of the
            dataframe. Format: "type:directory:function".
        task_params: A dictionary containing the task parameters.

    Returns:
        task_params: A dictionary with the updated dataframe path.
    """
    df_input = df_input.split(":")
    df_type = df_input[0]
    df_dir = df_input[1]

    # We overrride the dataframe path only if it was not passed in the task params
    if task_params[f"{df_type}_path"] is None:

        if df_dir == "inputs_directory":
            if df_type == "gts":
                df_fct = df_input[2] if len(df_input) == 3 else "test"
                task_params[f"{df_type}_path"] = str(
                    list((inputs_dir / f"{df_fct}/annotations").glob(f"*{df_type}*.csv"))[0]
                )
            else:
                task_params[f"{df_type}_path"] = str(list(inputs_dir.glob(f"*{df_type}*.csv"))[0])
        else:
            task_params[f"{df_type}_path"] = str(list((root_outputs_dir / df_dir).glob(f"*{df_type}*.csv"))[0])

    return task_params


def override_config_paths_with_inputs(
    inputs_dir: Path, root_outputs_dir: Path, inputs: str, init_params_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Overrides configuration paths (both model and dataframe ones) based on inputs specification.

    Args:
        inputs_dir: Directory path containing the input data.
        root_outputs_dir: Outputs directory root path.
        input: Comma-separated string of input types.
        init_params_dict: A dictionary containing the all initialization parameters necesarry for a task. It can include
            model, dataloaders, optimizer and scheduler initialization parameters.

    Returns:
        init_params_dict: Updated initialization parameters dictionary with overridden paths.
    """
    for input in inputs.split(","):
        # If 'model' keyword is in the task inputs the saved_checkpoint_path needs to be overriden
        if "model" in input.split(":")[0]:
            init_params_dict["model_init_params"] = override_saved_model_path(
                inputs_dir, root_outputs_dir, input, init_params_dict["model_init_params"]
            )

        # If 'predictions' or 'gts' keyword is in the task inputs the detections_path or gts_path needs to be overriden
        elif input.split(":")[0] in ["predictions", "gts"]:
            init_params_dict["task_params"] = override_dataframe_path(
                inputs_dir, root_outputs_dir, input, init_params_dict["task_params"]
            )

    return init_params_dict


########################################
### Objects initialization functions ###
########################################


def initialize_optimizer(
    model: torch.nn.Module,
    optimizer: str | None = None,
    optimizer_params: dict[str, Any] = {},
    checkpoint_path: Path | None = None,
):
    """
    Initialize optimizer for training a neural network model.

    Args:
        model: A torch.nn.Module instance representing the neural network model.
        optimizer: Name of the optimizer to be used. Default to None.
        optimizer_params: Dictionary containing parameters for the optimizer. Default is an empty dictionary.
        checkpoint_path: Path to a checkpoint file to resume training from. Default to None.

    Returns:
        optimizer: Initialized optimizer.
        optimizer_resumed: Boolean indicating whether the optimizer was resumed from a checkpoint.

    Raises:
        ValueError: If the specified optimizer is not implemented.
    """
    optimizer_name = optimizer

    if optimizer == "Adam":
        optimizer = Adam(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Parameter optimizer='{optimizer_name}' is not yet implemented!")

    # If a checkpoint_path is passed, then loads it and gets the optimizer_state_dict
    optimizer_resumed = False
    if checkpoint_path:
        optimizer_state_dict = load_pytorch_checkpoint(checkpoint_path)["optimizer_state_dict"]
        # If it's not None, loads state_dict in the optimizer
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)
            optimizer_resumed = True
            logger.info(
                f"Optimizer '{optimizer_name}' resumed from the state_dict saved in checkpoint file '{checkpoint_path}'."
            )

    if not optimizer_resumed:
        logger.info(f"Optimizer '{optimizer_name}' freshly initiated from scratch.")

    return optimizer, optimizer_resumed


def initialize_lr_scheduler(
    optimizer: str | None = None, lr_scheduler: str | None = None, lr_scheduler_params: dict[str, Any] = {}
):
    """
    Initializes a learning rate scheduler.

    Args:
        optimizer: The optimizer for which the scheduler is to be initialized. Default to None.
        lr_scheduler: The name of learning rate scheduler to be initialized. Default to None.
        lr_scheduler_params: Dictionary containing the parameters for the learning rate scheduler. Default is an empty
            dictionary.

    Returns:
        lr_scheduler: The initialized learning rate scheduler, or None if no scheduler is specified.
    """
    if lr_scheduler:
        if lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_params)
        else:
            raise ValueError(f"The `lr_scheduler` value '{lr_scheduler}' has not been implemented yet.")
        return lr_scheduler
    else:
        return None


def initialize_training_objects(
    checkpoint_path: Path | None = None,
    checkpoint_scoring_metric: str | None = None,
    checkpoint_scoring_order: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Initializes training objects for deep learning model training. If a `checkpoint_path` is passed, it loads metrics
    from the checkpoint file. If not, it initializes metrics from scratch.

    Args:
        checkpoint_path: Path to the checkpoint file. Default to None.
        checkpoint_scoring_metric: Metric used for scoring and ranking different epoch checkpoints. Default to None.
        checkpoint_scoring_order: The order (min or max) indicating the ranking direction for the checkpoint ranking.
            Default to None.

    Returns:
        training_objects: Dictionary containing the objects and information necessary to the training.
        metrics_resumed: Indicates whether metrics were resumed from the checkpoint or not.
    """
    ORDER_DISPATCHER = {"min": [min, np.inf], "max": [max, 0]}

    # Boolean to know if metrics dict was resumed
    metrics_resumed = False

    # If a checkpoint_path is passed, then loads it
    if checkpoint_path:
        ckpt = load_pytorch_checkpoint(checkpoint_path)
        if ckpt["metrics"]:
            training_objects = {
                "metrics": ckpt["metrics"],
                "epoch": ckpt["epoch"] + 1,  # number of first epoch when resuming
                "checkpoint_scoring_metric_value": (
                    ORDER_DISPATCHER[checkpoint_scoring_order][0](
                        flatten_dict(ckpt["metrics"])[checkpoint_scoring_metric]
                    )
                    if checkpoint_scoring_metric
                    else None
                ),
            }
            metrics_resumed = True
            logger.info(f"Training metrics resumed from the checkpoint file '{checkpoint_path}'.")

    # If metrics were not resumed, then initiates metrics dict from scratch
    if not metrics_resumed:
        metrics = {
            phase: {
                "loss": [],
                "comparable_loss": [],
                "level1": {"tp": [], "fp": [], "fn": [], "recall": [], "precision": []},
                "level2": {"mAP50": [], "mAP": []},
            }
            for phase in ["train", "val"]
        }
        training_objects = {
            "metrics": metrics,
            "epoch": 0,
            "checkpoint_scoring_metric_value": (
                ORDER_DISPATCHER[checkpoint_scoring_order][1] if checkpoint_scoring_order else None
            ),
        }
        logger.info(f"Training metrics freshly initiated from scratch.")

    return training_objects, metrics_resumed


####################################
### Task initialization function ###
####################################


def initialize_task(
    inputs_dir: Path,
    outputs_dir: Path,
    task_schema: TaskSchema,
    model_config: ModelConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Initializes a task with the provided configuration (input and output directories, task schema, and optionally a
    model config).

    Args:
        inputs_dir: Directory path containing the input data.
        outputs_dir: Directory path to save the output data.
        task_schema: TaskSchema defining the complete task configuration.
        model_config: ModelConfig configuration for the model. Default to None.

    Returns:
        initialized_objects: Dictionary of initialized objects (e.g. the model, the optimizer etc.).
        initialized_params: Dictionary of initialized parameters (e.g. task parameters etc.).

    Raises:
        ValueError: If the model config file is missing and needed for a 'train' or 'predict' task.

    Note:
        This function initializes various objects and parameters based on the provided inputs.
        It also saves the task initialization parameters for debugging purposes.
        Specific actions are performed for 'train' or 'predict' tasks, including model instantiation,
        optimizer initialization, and optionally loading metrics and a checkpoint file.

    """
    # Gets the initialization parameters
    init_params_dict = extract_task_init_params(task_schema.config, model_config)
    init_params_dict = override_config_paths_with_inputs(
        inputs_dir, outputs_dir.parent, task_schema.inputs, init_params_dict
    )

    # Initilizes the returned dict with initialized objects and parameters
    initialized_objects = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    initialized_params = {"task_params": init_params_dict["task_params"]}

    # Saves the task initialization parameters for debug
    save_dict_as_json(outputs_dir / f"{task_schema.task_type}_init_params.json", init_params_dict)

    ########## SPECIFIC TO TRAIN AND PREDICT TASKS ##########
    # If the task is of type 'train' or 'predict', instantiates a model
    if task_schema.task_type in ["train", "predict"]:
        if not init_params_dict["model_init_params"]:
            raise ValueError(f"The model config file is missing and needed for a 'train' or 'predict' task! ")

        # Instantiate model
        initialized_objects["model"], saved_checkpoint_path = initialize_model(
            root_data_dir=inputs_dir,
            **init_params_dict["model_init_params"],
        )
        # Sends the model to the correct device
        initialized_objects["model"].to(initialized_objects["device"])

        # Instantiates optimizer if the parameter is passed. This if statement is equivalent to if "train" task.
        if init_params_dict["optimizer_init_params"]:
            initialized_objects["optimizer"], optimizer_resumed = initialize_optimizer(
                model=initialized_objects["model"],
                checkpoint_path=saved_checkpoint_path,
                **init_params_dict["optimizer_init_params"],
            )

            # Instantiates lr_scheduler if one is passed
            if init_params_dict["lr_scheduler_init_params"]:
                initialized_objects["lr_scheduler"] = initialize_lr_scheduler(
                    optimizer=initialized_objects["optimizer"], **init_params_dict["lr_scheduler_init_params"]
                )

            # Instantiates metrics, epoch number and checkpoint_scoring_metric_value
            initialized_objects["training_extra_objects"], metrics_resumed = initialize_training_objects(
                saved_checkpoint_path,
                initialized_params["task_params"]["checkpoint_scoring_metric"],
                initialized_params["task_params"]["checkpoint_scoring_order"],
            )

            # Raise some warning or error if optimizer and metrics are not resumed together
            if metrics_resumed and not optimizer_resumed:
                logger.warning(
                    f"Metrics were resumed from epoch {initialized_objects['training_extra_objects']['epoch']}, but "
                    + "optimizer was not resumed! Are you sure your checkpoint file is not corrupted?"
                )
            if optimizer_resumed and not metrics_resumed:
                logger.error("Optimizer was resumed but not the metrics there must be something wrong.")
                raise ValueError("Optimizer was resumed but not the metrics there must be something wrong.")

            # If the we resume the training we copy the saved model in the outputs as it is the current best model
            if metrics_resumed:
                save_pytorch_model_checkpoint(
                    model=initialized_objects["model"],
                    optimizer=initialized_objects["optimizer"],
                    epoch=initialized_objects["training_extra_objects"]["epoch"],
                    metrics=initialized_objects["training_extra_objects"]["metrics"],
                    model_saving_type=init_params_dict["task_params"]["model_saving_type"],
                    save_optimizer=init_params_dict["task_params"]["save_optimizer"],
                    saving_path=outputs_dir / f"{init_params_dict['task_params']['checkpoint_file_name']}-best.tar",
                )

    # Istantiates dataloaders if dataloaders parameters are passed
    if init_params_dict["dataloaders_init_params"]:
        ds_fct = [input.split(":") for input in task_schema.inputs.split(",") if "dataset" in input][0]
        ds_fct = ds_fct[2] if len(ds_fct) == 3 else "test"
        initialized_objects["dataloaders"] = generate_dataloaders(
            root_data_dir=inputs_dir,
            task_type=task_schema.task_type,
            dataset_fct=ds_fct,
            **init_params_dict["dataloaders_init_params"],
        )

    return initialized_objects, initialized_params
