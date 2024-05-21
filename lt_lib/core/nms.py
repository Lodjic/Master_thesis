from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel
from tqdm.auto import tqdm

from lt_lib.core.initialize_task import initialize_task
from lt_lib.utils.log import log_task_begin_or_end
from lt_lib.utils.regex_matcher import get_elements_with_regex


def apply_nms_to_predictions(
    predictions_path: Path, nms_iou_threshold: float, outputs_dir: Path, disable_tqdm: bool | None = None
) -> None:
    """
    Apply non-maximum suppression (NMS) algorithm to predictions.

    Args:
        predictions_path: Path to the prediction file.
        nms_iou_threshold: IoU threshold used in the NMS algorithm.
        outputs_dir: Directory path to save the filtered predictions.
        disable_tqdm: Whether to disable the tqdm progress bar. Default is None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = pl.read_csv(predictions_path)

    filtered_predictions = pl.DataFrame(schema=predictions.schema)
    # Process images 1 by 1
    for img_name in tqdm(predictions["img_name"].unique(), disable=disable_tqdm):
        # Get bboxes of the current image processed
        img_predictions = predictions.filter(pl.col("img_name") == img_name)
        # Exports the bboxes in a torch.Tensor
        bboxes = torch.Tensor(
            img_predictions[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy(), device=device
        )
        # Exports the scores max in a torch.Tensor
        confidence_columns = get_elements_with_regex("^.*confidence.*$", img_predictions.columns, unique=False)
        img_confidences = img_predictions[confidence_columns].to_numpy()
        probable_labels = img_predictions["probable_label"].to_numpy() - 1
        scores = torch.Tensor(img_confidences[np.arange(len(img_confidences)), probable_labels], device=device)

        kept_idx = torch.ops.torchvision.nms(bboxes, scores, nms_iou_threshold).detach().cpu().numpy()

        filtered_predictions = pl.concat([filtered_predictions, img_predictions[kept_idx, :]])

    filtered_predictions.write_csv(outputs_dir / "predictions.csv", include_header=True)


def apply_nms(
    inputs_dir: Path,
    outputs_dir: Path,
    configs: dict[str, BaseModel],
):
    """
    Task function applying the non-maximum suppression (NMS) algorithm to predictions.

    Args:
        inputs_dir: Directory path containing the input data.
        outputs_dir: Directory path to store the output data.
        configs: Dictionary containing kwargs for the task. It should contain:
            - "task_schema": Schema defining the parameters of the task.
    """
    # Initialize model and dataloader
    _, initialized_params = initialize_task(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        task_schema=configs["task_schema"],
    )

    # Log end of eval
    log_task_begin_or_end("nms", "begin", "lower")

    # Launch prediction
    apply_nms_to_predictions(
        outputs_dir=outputs_dir,
        **initialized_params["task_params"],
    )

    # Log end of eval
    log_task_begin_or_end("nms", "end", "lower")
