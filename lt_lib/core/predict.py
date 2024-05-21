# Author : Loïc Thiriet

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel
from torch import Tensor
from tqdm.auto import tqdm

from lt_lib.core.initialize_task import initialize_task
from lt_lib.data.datasets import send_inputs_and_targets_to_device
from lt_lib.schemas.config_files_schemas import ModelConfig
from lt_lib.utils.log import log_task_begin_or_end


def export_predictions_to_df(
    imgs_path: list[str],
    batch_predictions: list[dict[str, np.ndarray | Tensor]],
    with_bboxes: bool = True,
    confidence_columns: Literal["all", "max"] = "all",
    label_column_name: str = "probable_label",
):
    """
    Exports predictions to a Polars DataFrame.

    Args:
        imgs_path: List of image paths.
        batch_predictions: List of dictionaries the predictions on one image.
        with_bboxes: Whether to include bounding boxes within the dataframe. Defaults to True.
        confidence_columns: Determines how to handle confidence columns, either keep them 'all' or just the 'max'.
            Defaults to "max".
        label_column_name: Name to give to the label column. Defaults to "probable_label".

    Returns:
        batch_predictions_df: A Polars DataFrame containing the predictions.
    """
    # Instantiate a polars df to save the predictions
    batch_predictions_df = pl.DataFrame()

    # Process images 1 by 1
    for img_idx, img_path in enumerate(imgs_path):
        # Detaches bboxes, scores and probable label
        if with_bboxes:
            img_bboxes = batch_predictions[img_idx]["boxes"].detach().to("cpu").numpy().astype(np.int16)
        img_scores = batch_predictions[img_idx]["scores"].detach().to("cpu").numpy().astype(np.float32)
        img_probable_labels = batch_predictions[img_idx]["probable_labels"].detach().to("cpu").numpy().astype(np.uint8)

        # If there are some detections on the image then appends those to the batch_predictions_df
        if len(img_probable_labels) > 0:
            # Creates a dict of columns from the bboxes, scores and probable_labels matrices
            cols = {"img_name": [str(img_path.name)] * len(img_probable_labels)}

            # Adds the bboxes columns
            if with_bboxes:
                cols.update(
                    {
                        "bbox_xmin": img_bboxes[:, 0],
                        "bbox_ymin": img_bboxes[:, 1],
                        "bbox_xmax": img_bboxes[:, 2],
                        "bbox_ymax": img_bboxes[:, 3],
                    }
                )

            # Adds the confidence columns based on the 'confidence_columns' provided
            if confidence_columns == "all":
                # class i+1 because the background class is index 0 in the RetinaNet
                cols.update({f"confidence_label_{i+1}": img_scores[:, i] for i in range(img_scores.shape[1])})
            elif confidence_columns == "max":
                cols.update({f"confidence": np.max(img_scores[:, :], axis=1)})
            else:
                raise ValueError(
                    f"Value for parameter 'confidence_columns' must be 'all' or 'max', not '{confidence_columns}'."
                )

            # Adds the label column with the provided name
            cols.update({label_column_name: img_probable_labels})

            # Transforms to a pandas Dataframe and concatenate with previous images
            img_df = pl.from_dict(cols)
            batch_predictions_df = pl.concat([batch_predictions_df, img_df], how="vertical")

    return batch_predictions_df


def model_predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    outputs_dir: Path,
) -> None:
    """
    Makes predictions using the provided model on data from the given dataloader and saves the results to a CSV file.

    Args:
        model: The model used for prediction.
        dataloader: The dataloader containing the data to predict on.
        device: The PyTorch device to perform computations on.
        outputs_dir: The directory path to save the predictions CSV file.
    """
    # Set model to evaluate mode
    model.eval()

    # Instantiate a polars df to save the predictions
    predictions_df = pl.DataFrame()

    # Iterate over data
    for inputs, _, imgs_path in tqdm(dataloader, desc=f"Mini-batch loop "):
        # Sends inputs and labels to device
        inputs, _ = send_inputs_and_targets_to_device(inputs, [], device)

        # forward
        with torch.no_grad():
            # Get model outputs
            _, batch_predictions = model(inputs)

        # Saves predictions image by image
        batch_predictions_df = export_predictions_to_df(
            imgs_path, batch_predictions, with_bboxes=True, confidence_columns="all", label_column_name="probable_label"
        )
        predictions_df = pl.concat([predictions_df, batch_predictions_df], how="vertical")

    # Adding an index column (using with `with_row_count` instead of `with_row_index` even if deprecated because
    # `with_row_index` does not exist on Colab)
    predictions_df = predictions_df.with_row_count("id")

    # Save predictions in csv file
    predictions_df.write_csv(outputs_dir / "predictions.csv", include_header=True)


def predict(
    inputs_dir: Path,
    outputs_dir: Path,
    configs: dict[str, BaseModel | ModelConfig],
) -> None:
    """
    Task function performing predictions on data using a model.

    Args:
        inputs_dir: Directory path containing the input data.
        outputs_dir: Directory path to store the output data.
        configs: Dictionary containing kwargs for the task. It should contain:
            - "task_schema": Schema defining the parameters of the task.
            - "model_config": Configuration for the model.
    """
    # Initialize model and dataloader
    initialized_objects, _ = initialize_task(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        task_schema=configs["task_schema"],
        model_config=configs["model_config"],
    )

    # Log beginning of predictions
    log_task_begin_or_end("predict", "begin", "lower")

    # Launch prediction
    model_predict(
        model=initialized_objects["model"],
        dataloader=initialized_objects["dataloaders"],
        device=initialized_objects["device"],
        outputs_dir=outputs_dir,
    )

    # Log end of predictions
    log_task_begin_or_end("predict", "end", "lower")
