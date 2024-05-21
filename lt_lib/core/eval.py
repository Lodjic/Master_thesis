from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import ray.train as ray_train
import wandb
from pydantic import BaseModel
from tqdm.auto import tqdm

from lt_lib.core.initialize_task import initialize_task
from lt_lib.core.matching import (
    filter_ids_with_low_iou,
    match_predictions_and_gts_for_one_img,
)
from lt_lib.core.metrics import *
from lt_lib.data.datasets import POLARS_GTS_SCHEMA
from lt_lib.utils.dict_utils import flatten_dict
from lt_lib.utils.load_and_save import load_json_as_dict, save_dict_as_json
from lt_lib.utils.log import log_task_begin_or_end
from lt_lib.utils.regex_matcher import get_elements_with_regex


def match_and_compute_eval_one_img(
    img_name: str | Path,
    gts: pl.DataFrame,
    predictions: pl.DataFrame,
    label_to_label_name_dict: dict[str, str],
    matching_iou_threshold: float = 0.4,
) -> tuple[str, dict[str, dict[str, float | dict[str, float]]]]:
    """
    Computes evaluation metrics for a single image.

    Args:
        img_name: Name of the image.
        gts: DataFrame containing the ground truth bounding boxes.
        predictions: DataFrame containing predicted bounding boxes.
        label_to_label_name_dict: Dictionary mapping class ids to class label names.
        matching_iou_threshold: Threshold for matching predicted and ground truth bounding boxes based on the IoU.
            Defaults to 0.4.

    Returns:
        img_name, img_metrics: The image name and the computed metrics in the form of a dictionary.
    """
    # Filter image gts and predictions
    img_gts = gts.filter(pl.col("img_name") == img_name)
    img_predictions = predictions.filter(pl.col("img_name") == img_name)

    # Get image gts and predictions bboxes
    img_gts_bboxes = img_gts[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()
    img_predictions_bboxes = img_predictions[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()

    # Computes matching
    img_predictions_matched_idx, img_gts_matched_idx, img_matched_ious = match_predictions_and_gts_for_one_img(
        img_predictions_bboxes,
        img_gts_bboxes,
    )
    # Filters low iou matches
    img_predictions_matched_idx, img_gts_matched_idx, _ = filter_ids_with_low_iou(
        img_predictions_matched_idx, img_gts_matched_idx, img_matched_ious, matching_iou_threshold
    )

    # Compute img metrics
    img_metrics = compute_img_metrics(
        img_gts, img_predictions, img_gts_matched_idx, img_predictions_matched_idx, label_to_label_name_dict
    )
    return img_name, img_metrics


def match_and_compute_eval(
    predictions_path: str | Path,
    gts_path: str | Path,
    outputs_dir: Path,
    matching_iou_threshold: float = 0.4,
    custom_metrics: dict[str, str] | None = None,
    disable_tqdm: bool | None = None,
) -> None:
    """
    Computes the evaluation metrics for all the images based on the prediction CSV file and save it to a JSON file.

    Args:
        predictions_path: Path to the prediction CSV file.
        gts_path: Path to the ground truth CSV file.
        outputs_dir: Directory path to save the evaluation results.
        matching_iou_threshold: IoU threshold for matching predictions to ground truths. Defaults to  0.4.
        custom_metrics: Dictionary of custom metrics to compute. Defaults to None.
        disable_tqdm: Whether to disable the tqdm progress bar. Defaults to None.
    """
    # Load gts and predictions csv files
    predictions = pl.read_csv(predictions_path)
    gts = pl.read_csv(gts_path).cast(POLARS_GTS_SCHEMA, strict=True)
    label_to_label_name_dict = load_json_as_dict(Path(gts_path).parent / "label_to_label_name.json")

    # Get the image names list
    img_name_list = gts["img_name"].unique().to_list()

    # Initiate results metrics dict
    metrics = {"global_results": {}, "per_img_results": {}}

    # Processes images 1 by 1
    for img_name in tqdm(img_name_list, disable=disable_tqdm):
        _, img_metrics = match_and_compute_eval_one_img(
            img_name, gts, predictions, label_to_label_name_dict, matching_iou_threshold
        )
        metrics["per_img_results"][img_name] = img_metrics

    # Computes global metrics
    global_results = compute_global_results(metrics["per_img_results"])
    # Adds custom metrics if there are some
    if custom_metrics:
        global_results = add_custom_metrics(global_results, custom_metrics)
    # Saves global_metrics to metrics dict
    metrics["global_results"] = global_results

    # Saves the evaluation metrics computed to a json file
    save_dict_as_json(file_path=outputs_dir / "eval.json", dictionary=metrics)

    ### Ray and wandb logging
    # Flattens global_metrics dict for reporting if necessary
    if ray_train.context.session.get_session() or wandb.run:
        flattened_global_results = flatten_dict(metrics["global_results"])
    # If wandb is running, then logs interesting metrics
    if wandb.run:
        metrics_to_keep = get_elements_with_regex(
            "^(level1.recall|level1.precision|.*civil.*\.recall.*|.*civil.*\.precision.*).*$",
            flattened_global_results.keys(),
            unique=False,
        )
        wandb.log({m: flattened_global_results[m] for m in metrics_to_keep})
    # If Ray-tune is operating, then logs the metrics to it
    if ray_train.context.session.get_session():
        ray_train.report(flattened_global_results)


def match_and_compute_eval_multi_process(
    predictions_path: str | Path,
    gts_path: str | Path,
    outputs_dir: Path,
    matching_iou_threshold: float = 0.4,
    processes: int = 6,
    chunksize: int = 20,
    disable_tqdm: bool | None = None,
) -> None:
    """
    Computes the evaluation metrics for all the images based on the prediction CSV file across multiple processes and
    save it to a JSON file. It is the same function as the function of the same name but in a multi-processed version.

    Args:
        predictions_path: Path to the prediction CSV file.
        gts_path: Path to the ground truth CSV file.
        outputs_dir: Directory path to save evaluation results.
        matching_iou_threshold: IoU threshold for matching predictions to ground truths. Defaults to 0.4.
        processes: Number of processes to use. Defaults to 6.
        chunksize: Chunk size for multiprocessing. Defaults to 20.
        disable_tqdm: Whether to disable the tqdm progress bar. Defaults to None.
    """
    # Load gts and predictions csv files
    predictions = pl.read_csv(predictions_path)
    gts = pl.read_csv(gts_path).cast(POLARS_GTS_SCHEMA, strict=True)

    label_to_label_name_dict = load_json_as_dict(Path(gts_path).parent / "label_to_label_name.json")

    # Get the image names list
    img_name_list = gts["img_name"].unique().to_list()

    # Initiate results metrics dict
    metrics = {"global_results": {}, "per_img_results": {}}

    # Distribute files accross processes file by file
    with Pool(processes=processes) as pool:
        with tqdm(total=len(img_name_list), disable=disable_tqdm) as pbar:
            for img_metrics in pool.imap(
                partial(
                    match_and_compute_eval_one_img,
                    gts=gts,
                    predictions=predictions,
                    label_to_label_name_dict=label_to_label_name_dict,
                    matching_iou_threshold=matching_iou_threshold,
                ),
                img_name_list,
                chunksize=chunksize,
            ):
                metrics["per_img_results"][img_metrics[0]] = img_metrics[1]
                pbar.update()

    # Compute global metrics
    metrics["global_results"] = compute_global_results(metrics["per_img_results"])

    save_dict_as_json(file_path=outputs_dir / "eval.json", dictionary=metrics)


def evaluation(
    inputs_dir: Path,
    outputs_dir: Path,
    configs: dict[str, BaseModel],
):
    """
    Task function evaluating a model based on a prediction and ground truth file.

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
    log_task_begin_or_end("eval", "begin", "lower")

    # Launch prediction
    match_and_compute_eval(
        outputs_dir=outputs_dir,
        **initialized_params["task_params"],
    )

    # Log end of eval
    log_task_begin_or_end("eval", "end", "lower")
