# Author : Loïc Thiriet

from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import wandb
from ray import train as ray_train
from tqdm.auto import tqdm

from lt_lib.core.initialize_task import initialize_task
from lt_lib.core.matching import batch_matching_during_training
from lt_lib.core.metrics import (
    compute_level1_metrics_from_AP_df,
    compute_mAP_from_AP_df,
)
from lt_lib.data.datasets import send_inputs_and_targets_to_device
from lt_lib.schemas.config_files_schemas import BaseYamlConfig, ModelConfig
from lt_lib.utils.dict_utils import flatten_dict, keep_only_last_value_for_all_keys
from lt_lib.utils.load_and_save import save_dict_as_json, save_pytorch_model_checkpoint
from lt_lib.utils.log import (
    log_task_begin_or_end,
    logging_end_of_training,
    wandb_log_model,
)


def save_level1_metrics_values_to_metrics(
    metrics: dict, phase: Literal["train", "val"], tp: int, fp: int, fn: int, recall: float, precision: float
) -> dict:
    """
    Saves the values of level 1 metrics (True Positives, False Positives, False Negatives, recall and precision) to the
    metrics dictionary for the specified phase.

    Args:
        metrics: Dictionary containing the metrics values of past epochs.
        phase: The phase for which the metrics are being saved, either "train" or "val".
        tp: Number of True Positives.
        fp: Number of False Positives.
        fn: Number of False Negatives.
        recall: Recall value.
        precision: Precision value.

    Returns:
        metrics: The updated metrics dictionary.
    """
    metrics[phase]["level1"]["tp"].append(tp)
    metrics[phase]["level1"]["fp"].append(fp)
    metrics[phase]["level1"]["fn"].append(fn)
    metrics[phase]["level1"]["recall"].append(recall)
    metrics[phase]["level1"]["precision"].append(precision)
    return metrics


def save_level2_metrics_values_to_metrics(
    metrics: dict, phase: Literal["train", "val"], mAP: float, mAP50: float
) -> dict:
    """
    Saves the values of level 2 metrics (mAP and mAP50) to the metrics dictionary for the specific phase.

    Args:
        metrics: Dictionary containing the metrics values of past epochs.
        phase: The phase for which the metrics are being saved, either "train" or "val".
        mAP (float): The mean Average Precision value.
        mAP50 (float): The mean Average Precision at the 0.5 matching IoU threshold.

    Returns:
        metrics: The updated metrics dictionary.
    """
    metrics[phase]["level2"]["mAP50"].append(mAP50)
    metrics[phase]["level2"]["mAP"].append(mAP)
    return metrics


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    extra_objects: dict[str, Any],
    # Training params
    n_epochs: int,
    loss_reduction_factor: float,
    loss_reduction_sign_indicator: int,
    # Matching params
    matching_iou_threshold: float,
    # Inputs directory
    inputs_dir: Path,
    # Saving params
    model_saving_type: str,
    save_optimizer: bool,
    saving_frequency: int,
    checkpoint_scoring_metric: str | None,
    checkpoint_scoring_order: Literal["min", "max"] | None,
    checkpoint_file_name: str,
    outputs_dir: Path,
    # Wandb model logging param
    wandb_log_model_checkpoint: bool,
    # Seed
    manual_seed: int | float | None,
    # Tqdm disabling params
    disable_epoch_tqdm: bool | None,
    disable_batch_tqdm: bool | None,
) -> None:
    """
    Trains a given model on a certain number of epochs, using the provided data and configurations.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimizer used for updating the model's parameters.
        lr_scheduler: Learning rate scheduler for scheduling the learning rate during training.
        dataloaders: A dictionary containing  the dataloaders for training and validation phases.
        device: The PyTorch device on which the model and data should be processed.
        extra_objects: Additional objects required for training, such as epoch metrics.
        n_epochs: Number of epochs on which to train the model.
        loss_reduction_factor: Factor to reduce one of the two losses by.
        loss_reduction_sign_indicator: Sign indicating which loss to reduce.
        matching_iou_threshold: The threshold IoU value above which to consider a valid match.
        inputs_dir: Directory path containing the input data.
        model_saving_type: Method specifying how to save the model weights.
        save_optimizer: Whether to save the optimizer state along with the model.
        saving_frequency: Epoch frequency at which to save the model.
        checkpoint_scoring_metric: Metric used for scoring and ranking different epoch checkpoints.
        checkpoint_scoring_order: The order (min or max) indicating the ranking direction for the checkpoint ranking.
        checkpoint_file_name: Name of the checkpoint file.
        outputs_dir: Directory path to save outputs, metrics and checkpoints.
        wandb_log_model_checkpoint: Whether to log model checkpoints to Weights & Biases.
        manual_seed: Seed for random number generation.
        disable_epoch_tqdm: Whether to disable the tqdm progress bar for epochs.
        disable_batch_tqdm: Whether to disable the tqdm progress bar for batches.
    """
    # Sets manual_seed if provided
    if manual_seed:
        torch.manual_seed(manual_seed)

    # Initialize metrics
    metrics = extra_objects["metrics"]
    checkpoint_scoring_metric_value = extra_objects["checkpoint_scoring_metric_value"]

    epoch_progress_bar = tqdm(
        range(extra_objects["epoch"], n_epochs), desc="Epoch loop", leave=True, position=0, disable=disable_epoch_tqdm
    )
    for epoch in epoch_progress_bar:

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Initiates epoch metrics. they will be updated during the epoch.
            running_loss = 0.0
            comparable_loss = 0.0
            epoch_AP_df = pl.DataFrame()
            # for key in ["tp", "fp", "fn"]:
            #     metrics[phase]["level1"][key].append(0)

            # Iterate over data
            for inputs, targets, imgs_path in tqdm(
                dataloaders[phase],
                desc=f"Mini-batch loop ({phase})",
                leave=False,
                position=1,
                disable=disable_batch_tqdm,
            ):
                # Sends inputs and labels to device
                inputs, targets = send_inputs_and_targets_to_device(inputs, targets, device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Tracks history if only in train phase
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculates the loss
                    head_losses, detections = model(inputs, targets)

                    # Computes the loss weights from the ratio
                    weight_classification_loss = 1 / loss_reduction_factor if loss_reduction_sign_indicator < 0 else 1
                    weight_regression_loss = 1 / loss_reduction_factor if loss_reduction_sign_indicator > 0 else 1

                    batch_loss = (
                        weight_classification_loss * head_losses["classification"]
                        + weight_regression_loss * head_losses["bbox_regression"]
                    )
                    batch_comparable_loss = head_losses["classification"] + head_losses["bbox_regression"]

                    if phase == "train":
                        batch_loss.backward()
                        optimizer.step()

                # Adds batch loss to running epoch loss
                running_loss += batch_loss.item() * inputs.size(0)
                comparable_loss += batch_comparable_loss.item() * inputs.size(0)

                if model.process_detections_during_training:
                    # batch_predictions_df = export_predictions_to_df(imgs_path, detections)
                    batch_AP_df = batch_matching_during_training(imgs_path, detections, targets, False, "max", "label")
                    batch_AP_df.drop_in_place("id")
                    epoch_AP_df = pl.concat([epoch_AP_df, batch_AP_df], how="vertical")

            # Stores loss value
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            metrics[phase]["loss"].append(epoch_loss)
            metrics[phase]["comparable_loss"].append(comparable_loss / len(dataloaders[phase].dataset))

            if model.process_detections_during_training:
                # Computes mAP
                # Note: using with `with_row_count` instead of `with_row_index` because does not exist on Colab
                epoch_AP_df = epoch_AP_df.with_row_count("id")
                AP_metrics_values = compute_mAP_from_AP_df(epoch_AP_df, labels_to_consider=[1, 2, 3])
                metrics = save_level2_metrics_values_to_metrics(metrics, phase, *AP_metrics_values)
                level1_metrics_values = compute_level1_metrics_from_AP_df(epoch_AP_df, matching_iou_threshold)
                metrics = save_level1_metrics_values_to_metrics(metrics, phase, *level1_metrics_values)

            # If we are in training phase and there is a lr_scheduler then we should increment its step
            if phase == "train":
                if lr_scheduler:
                    lr_scheduler.step(epoch_loss)

            # If we are in the validation phase, we might have to save and log the model
            else:
                checkpoint_scoring_metric_value = save_and_log_current_epoch_model_during_training(
                    epoch,
                    checkpoint_scoring_metric,
                    checkpoint_scoring_order,
                    checkpoint_scoring_metric_value,
                    model,
                    optimizer,
                    metrics,
                    model_saving_type,
                    save_optimizer,
                    saving_frequency,
                    checkpoint_file_name,
                    outputs_dir,
                    wandb_log_model_checkpoint,
                )

    # Saving metrics to a json file
    save_dict_as_json(outputs_dir / "metrics.json", metrics)

    # Log end of training
    logging_end_of_training(optimizer, n_epochs, metrics)


def save_and_log_current_epoch_model_during_training(
    epoch: int,
    checkpoint_scoring_metric: str | None,
    checkpoint_scoring_order: Literal["min", "max"] | None,
    checkpoint_scoring_metric_value: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    model_saving_type: str,
    save_optimizer: bool,
    saving_frequency: int,
    checkpoint_file_name: str,
    outputs_dir: Path,
    wandb_log_model_checkpoint: bool,
) -> float:
    ORDER_DISPATCHER = {"min": np.argmin, "max": np.argmax}

    # Get the wandb.run if was initialized and is running
    wandb_run = wandb.run

    current_epoch_metrics = keep_only_last_value_for_all_keys(flatten_dict(metrics))

    # If a wandb_run was initialized, then logs metrics to wandb
    if wandb_run:
        wandb.log(current_epoch_metrics)

    # Saving path of the current epoch model checkpoint
    saving_chkpt_path = None

    # If the epoch saving period is the current epoch, then saves and logs current epoch model
    if saving_frequency > 0 and epoch % saving_frequency == 0:
        # Saves the model in a tar file
        saving_chkpt_path = outputs_dir / f"{checkpoint_file_name}-E{epoch}/{checkpoint_file_name}-E{epoch}.tar"
        save_pytorch_model_checkpoint(
            model, optimizer, epoch, metrics, model_saving_type, save_optimizer, saving_chkpt_path
        )
        # If a wandb_run was initialized, then logs the best model to wandb
        if wandb_run and wandb_log_model_checkpoint:
            wandb_log_model(
                chkpt_file_path=saving_chkpt_path,
                artifact_type=f"epoch_{model_saving_type}",
                artifact_name=f"{model_saving_type}_E{epoch}",
            )

    # If the metric use for best model evaluation of the current epoch is below saved best value and we do not already
    # save the model at every epoch, then saves and logs the new best model
    if (
        ORDER_DISPATCHER[checkpoint_scoring_order](
            [checkpoint_scoring_metric_value, current_epoch_metrics[checkpoint_scoring_metric]]
        )
        == 1
        and saving_frequency != 1
    ):
        checkpoint_scoring_metric_value = current_epoch_metrics[checkpoint_scoring_metric]
        # Saves the model
        saving_chkpt_path = outputs_dir / f"{checkpoint_file_name}-best/{checkpoint_file_name}-best.tar"
        save_pytorch_model_checkpoint(
            model, optimizer, epoch, metrics, model_saving_type, save_optimizer, saving_chkpt_path
        )
        # If a wandb_run was initialized, then logs the best model to wandb
        if wandb_run and wandb_log_model_checkpoint:
            wandb_log_model(
                chkpt_file_path=saving_chkpt_path,
                artifact_type=f"best_{model_saving_type}",
                artifact_name=f"best_{model_saving_type}",
            )

    # If Ray-Tune is operating, then reports epoch metrics and eventually checkpoint to it
    if ray_train.context.session.get_session():
        # Adds epoch to logged metrics
        current_epoch_metrics["epoch"] = epoch

        # If the model was saved during this epoch, then reports it to Ray-Tune
        if saving_chkpt_path:
            ray_train.report(
                current_epoch_metrics,
                checkpoint=ray_train.Checkpoint.from_directory(saving_chkpt_path.parent),
            )
        # If no model checkpoint was saved, then only reports the metrics
        else:
            ray_train.report(current_epoch_metrics)

    return checkpoint_scoring_metric_value


def train(
    inputs_dir: Path,
    outputs_dir: Path,
    configs: dict[str, BaseYamlConfig | ModelConfig],
) -> None:
    """
    Task function training a model using provided inputs and configurations.

    Args:
        inputs_dir: Directory path containing the input data.
        outputs_dir: Directory path to store the output data.
        configs: Dictionary containing kwargs for the task. It should contain:
            - "task_schema": Schema defining the parameters of the task.
            - "model_config": Configuration for the model.
    """
    # Initialize model, optimizer, lr_scheduler, dataloaders and train_params
    initialized_objects, initialized_params = initialize_task(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        task_schema=configs["task_schema"],
        model_config=configs["model_config"],
    )

    # Log beginning of training
    log_task_begin_or_end("train", "begin", "lower")

    # Launch training
    train_model(
        model=initialized_objects["model"],
        optimizer=initialized_objects["optimizer"],
        lr_scheduler=initialized_objects["lr_scheduler"],
        dataloaders=initialized_objects["dataloaders"],
        device=initialized_objects["device"],
        extra_objects=initialized_objects["training_extra_objects"],
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        **initialized_params["task_params"],
    )

    # Log end of training
    log_task_begin_or_end("train", "end", "lower")
