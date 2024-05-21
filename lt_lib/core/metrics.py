import copy

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from lt_lib.utils.dict_utils import add_nested_dict, extend_flattened_dict, flatten_dict
from lt_lib.utils.shuting_yard_algorithm import shunting_yard_custom

##################################
### Function for training step ###
##################################


def filter_low_iou_matches_of_AP_df(AP_df: pl.DataFrame, matching_iou_threshold: float) -> pl.DataFrame:
    """
    Adjusts attributes of matched predictions with an Intersection over Union (IoU) below a given threshold.

    Args:
        AP_df: AP dataframe, a dataframe containing predictions and associated specific attribute columns (e.g. 'iou',
        'match', 'correct_match' etc.).
        matching_iou_threshold: The threshold IoU value above which to consider a valid match.

    Returns:
        A dataframe with low IoU matches attributes ('match' and 'correct_match') adjusted.
    """
    # Finds matched predictions with iou below threshold
    too_low_iou_ids = AP_df.filter(
        pl.col("match") == 1, pl.col("iou").is_between(0, matching_iou_threshold, closed="none")
    )["id"]

    # Passes 'match' and 'correct_match' attributes of low ious predictions to 0 -> makes TPs become FPs
    AP_df = AP_df.with_columns(match=AP_df["match"].scatter(too_low_iou_ids, 0))
    AP_df = AP_df.with_columns(correct_match=AP_df["correct_match"].scatter(too_low_iou_ids, 0))

    # Adds newly created FNs because some TPs that have become FPs
    # Note: FNs do not change anything in the AP calculus but it does for any other metric
    fn_df = AP_df.filter(pl.col("id").is_in(too_low_iou_ids.to_list()))
    fn_df = fn_df.with_columns(confidence=pl.lit(0.0), iou=pl.lit(0.0))
    fn_df.drop_in_place("id")
    # Note: using with `with_row_count` instead of `with_row_index` because does not exist on Colab
    fn_df = fn_df.with_row_count("id", offset=len(AP_df))

    AP_df = pl.concat([AP_df, fn_df], how="vertical_relaxed")

    return AP_df


def compute_mAP_from_AP_df(
    AP_df: pl.DataFrame, labels_to_consider: list[int] | list[str], label_to_label_name: dict[int:str] | None = None
) -> tuple[float]:
    """
    Computes the mean Average Precision (mAP) of a dataset from an AP dataframe, a specific dataframe containing
    specific attribute columns.

    Args:
        AP_df: AP dataframe, a dataframe containing predictions and associated specific attribute columns (e.g. 'match',
            'correct_match' etc.).
        labels_to_consider: List of labels (either integers or strings) to consider for the mAP computation.
        label_to_label_name: Dictionary mapping class ids to class label names. Defaults to None.

    Returns:
        The mean Average Precision (mAP)
        The mean Average Precision value for the IoU threshold of 0.5 (mAP50).
    """
    if isinstance(labels_to_consider[0], str):
        label_name_to_label = {v: k for (k, v) in label_to_label_name.items()}
        labels_to_consider = [label_name_to_label[label_name] for label_name in labels_to_consider]

    # Filters only considered labels
    AP_df_filtered = AP_df.filter(pl.col("label").is_in(labels_to_consider))
    AP_df_filtered.drop_in_place("id")
    # Note: using with `with_row_count` instead of `with_row_index` because does not exist on Colab
    AP_df_filtered = AP_df_filtered.with_row_count("id")

    AP_per_thresh = []

    # Compute APs for different iou thresholds
    for iou_thresh in np.linspace(0.5, 0.95, 10):
        # Filters low iou matches
        AP_df_filtered = filter_low_iou_matches_of_AP_df(AP_df_filtered, iou_thresh)

        AP_per_label = []

        # For each label computes its AP
        for label in AP_df_filtered["label"].unique():
            AP_df_filtered_per_label = AP_df_filtered.filter(pl.col("label") == label)
            AP_per_label.append(
                average_precision_score(
                    AP_df_filtered_per_label["correct_match"], AP_df_filtered_per_label["confidence"]
                )
            )

        AP_per_thresh.append(np.mean(AP_per_label))

    return np.mean(AP_per_thresh), AP_per_thresh[0]


def compute_level1_metrics_from_AP_df(
    AP_df: pl.DataFrame,
    matching_iou_threshold: float,
):
    """
    Computes level 1 metrics from the provided AP dataframe, a specific dataframe containing specific attribute columns.
    Matches are considered correct match only if their IoU is superior to the matching_iou_threshold.

    Args:
        AP_df: AP_df: AP dataframe, a dataframe containing predictions and associated specific attribute columns (e.g. 'match',
            'correct_match' etc.).
        matching_iou_threshold: The threshold IoU value above which to consider a valid match.

    Returns:
        tp: True positives.
        fp: False positives
        fn: False negatives.
        recall: Recall.
        precision: Precision.
    """
    # Filters low iou matches
    AP_df = filter_low_iou_matches_of_AP_df(AP_df, matching_iou_threshold)

    # Counts TPs, FPs and FNs
    tp = AP_df.filter(pl.col("match") == 1)["match"].count()
    fp = AP_df.filter(pl.col("match") == 0, pl.col("confidence") > 0.0)["match"].count()
    fn = AP_df.filter(pl.col("match") == 0, pl.col("confidence") == 0.0)["match"].count()

    # Computes recall and precision
    recall = tp / (tp + fn) if tp != 0 else 0
    precision = tp / (tp + fp) if tp != 0 else 0

    return tp, fp, fn, recall, precision


#####################################
### Functions for evaluation step ###
#####################################


def correct_and_add_metrics_to_global_results_nested_dict(nested_dict: dict) -> dict:
    """
    Recusrive function correcting metric values and adding new metrics to a global results nested dictionary.

    Args:
        nested_dict: The global results nested dictionary.

    Returns:
        nested_dict: The updated global results nested dictionary with added metrics.
    """
    if "recall" in nested_dict.keys():
        # Gets label_name tps, fps and fns
        tp = nested_dict["tp"]
        fp = nested_dict["fp"]
        fn = nested_dict["fn"]

        # Updates recall and precision
        recall = tp / (tp + fn) if tp != 0 else 0
        nested_dict["recall"] = recall
        precision = tp / (tp + fp) if tp != 0 else 0
        nested_dict["precision"] = precision
        # Updates f1-score
        nested_dict["f1"] = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        # If nested_dict is not a level 1 metrics dict (so level 2 or more), then computes soft and classif metrics
        if len(list(nested_dict.keys())) > 6:
            # Gets missed count
            missed = nested_dict["missed"]
            nonexistent = nested_dict["nonexistent"]

            # Computes soft_recall
            nested_dict["soft_recall"] = (tp + fn - missed) / (tp + fn) if tp + fn - missed != 0 else 0
            # Computes soft_precision
            nested_dict["soft_precision"] = (
                (tp + fn - missed) / (tp + fn - missed + nonexistent) if tp + fn - missed != 0 else 0
            )

            # Computes classif_recall
            nested_dict["classif_recall"] = tp / (tp + fn - missed) if tp != 0 else 0
            # Computes classif_precision
            nested_dict["classif_precision"] = tp / (tp + fp - nonexistent) if tp != 0 else 0

        return nested_dict

    else:
        for key in nested_dict.keys():
            nested_dict[key] = correct_and_add_metrics_to_global_results_nested_dict(nested_dict[key])

        return nested_dict


def compute_global_results(per_img_results: dict) -> dict:
    """
    Computes global results metrics from the aggregation of per-image results.

    Args:
        per_img_results: A dictionary containing per-image results.

    Returns:
        metrics: A dictionary containing the aggregated global results.
    """
    img_name_list = list(per_img_results.keys())
    metrics = copy.deepcopy(per_img_results[img_name_list[0]])

    for img_name in img_name_list[1:]:
        metrics = add_nested_dict(metrics, per_img_results[img_name])

    metrics = correct_and_add_metrics_to_global_results_nested_dict(metrics)

    return metrics


def add_custom_metrics(global_results: dict[str, dict[str, dict[str, float]]], custom_metrics: dict[str, str]):
    """
    Calculates custom metrics based on provided expressions and add them to global results.

    Args:
        global_results: The global results nested dictionary.
        custom_metrics: A dictionary where keys are custom metric names and values are expressions in string format
            that can include existing metric names.

    Returns:
        dict: The global results nested dictionary updated with the custom metrics.
    """
    flattened_global_results = flatten_dict(global_results)

    for metric_name, expression in custom_metrics.items():
        flattened_global_results[f"custom_metrics.{metric_name}"] = shunting_yard_custom(
            expression, flattened_global_results
        )

    return extend_flattened_dict(flattened_global_results)


def compute_img_metrics(
    img_gts: pl.DataFrame,
    img_predictions: pl.DataFrame,
    img_gts_matched_idx: list,
    img_predictions_matched_idx: list,
    label_to_label_name_dict: dict[str, str],
) -> dict[str, dict[str, float | dict[str, float]]]:
    """
    Computes level1 and level2 metrics for one image based on the matched predictions and ground truths.

    Args:
        img_gts: Ground truth of the image.
        img_predictions: Predictions made by a model on the image.
        img_gts_matched_idx: List of the ground truth indices that were matched to a prediction.
        img_predictions_matched_idx: List of the prediction indices that were matched to a ground truth.
        label_to_label_name_dict: Dictionary mapping class ids to class label names.

    Returns:
        metrics: A dictionary containing the computed metrics.
    """
    # Keeps only 'id' and 'probable_label' columns. Casting is important to not get an error in case the
    # img_predictions dataframe is empty.
    img_predictions = img_predictions[["id", "probable_label"]].cast({"id": pl.UInt16, "probable_label": pl.Int32})

    # Initiate image metrics dict that will be returned
    metrics = {}

    # Level 1
    metrics["level1"] = {}
    tp = len(img_gts_matched_idx)
    # Computes true positives, false positive and false negatives
    metrics["level1"]["tp"] = tp
    metrics["level1"]["fp"] = len(img_predictions) - len(img_predictions_matched_idx)
    metrics["level1"]["fn"] = len(img_gts) - len(img_gts_matched_idx)
    # Computes recall and precision
    recall = tp / (tp + metrics["level1"]["fn"]) if tp > 0 else 0
    metrics["level1"]["recall"] = recall
    precision = tp / (tp + metrics["level1"]["fp"]) if tp > 0 else 0
    metrics["level1"]["precision"] = precision
    # Computes f1-score
    metrics["level1"]["f1"] = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    label_list = list(label_to_label_name_dict.keys())

    # Level 2
    metrics["level2"] = {}
    for label_pred in label_list:
        metrics["level2"][label_to_label_name_dict[label_pred]] = {}

        # Among predictions that where matched, gets only the indices for which probable_label=label_pred
        idx_label_pred = np.where(img_predictions[img_predictions_matched_idx, "probable_label"] == int(label_pred))[0]
        # Among indices of predictions and gts matched, gets only the indices for which probable_label=label_pred
        img_predictions_idx_per_label = img_predictions_matched_idx[idx_label_pred]
        img_gts_idx_per_label = img_gts_matched_idx[idx_label_pred]

        img_matched_predictions_labels = img_predictions[img_predictions_idx_per_label, "probable_label"]
        img_matched_gts_labels = img_gts[img_gts_idx_per_label, "label"]

        # Computes confusions between classes
        count_confusions_other_labels_predicted_as_label_pred = 0
        for label_gt in label_list:
            count = (img_matched_gts_labels == int(label_gt)).sum()
            metrics["level2"][label_to_label_name_dict[label_pred]][label_to_label_name_dict[label_gt]] = count
            count_confusions_other_labels_predicted_as_label_pred += count

        # Computes true positives
        tp = (img_matched_gts_labels == int(label_pred)).sum()
        metrics["level2"][label_to_label_name_dict[label_pred]]["tp"] = tp
        count_confusions_other_labels_predicted_as_label_pred -= tp

        # Computes false positives
        fp = (
            len(img_predictions.filter(pl.col("probable_label") == int(label_pred)))
            - metrics["level2"][label_to_label_name_dict[label_pred]]["tp"]
        )
        metrics["level2"][label_to_label_name_dict[label_pred]]["fp"] = fp

        # Computes false negatives: gts labeled with a certain label minus gts labeled with this same label and matched
        fn = (
            len(img_gts.filter(pl.col("label") == int(label_pred)))
            - metrics["level2"][label_to_label_name_dict[label_pred]]["tp"]
        )
        metrics["level2"][label_to_label_name_dict[label_pred]]["fn"] = fn

        # Computes recall
        recall = tp / (tp + fn) if tp > 0 else 0
        metrics["level2"][label_to_label_name_dict[label_pred]]["recall"] = recall

        # Computes precision
        precision = tp / (tp + fp) if tp > 0 else 0
        metrics["level2"][label_to_label_name_dict[label_pred]]["precision"] = precision

        # Computes f1-score
        metrics["level2"][label_to_label_name_dict[label_pred]]["f1"] = (
            2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        )

        # Computes nonexistent
        metrics["level2"][label_to_label_name_dict[label_pred]]["nonexistent"] = (
            fp - count_confusions_other_labels_predicted_as_label_pred
        )

    # Computes soft and classif metrics
    label_name_list = list(label_to_label_name_dict.values())
    for label_name in label_name_list:
        count_confusions_label_predicted_as_other_labels = 0

        # Counts the number of confusions of label predicted as other labels
        other_label_name_list = copy.deepcopy(label_name_list)
        other_label_name_list.remove(label_name)
        for other_label_name in other_label_name_list:
            count_confusions_label_predicted_as_other_labels += metrics["level2"][other_label_name][label_name]

        # Gets label_name tps, fps and fns
        fn = metrics["level2"][label_name]["fn"]

        # Computes missed
        missed = fn - count_confusions_label_predicted_as_other_labels
        metrics["level2"][label_name]["missed"] = missed

        ### Usually useless because the images are tiles (small images) so no need to compute complex metrics
        # # Gets tp, fp and nonexistent counts
        # tp = metrics["level2"][label_name]["tp"]
        # fp = metrics["level2"][label_name]["fp"]
        # nonexistent = metrics["level2"][label_name]["nonexistent"]
        # # Computes soft_recall
        # metrics["level2"][label_name]["soft_recall"] = (tp + fn - missed) / (tp + fn) if tp + fn - missed != 0 else 0
        # # Computes soft_precision
        # metrics["level2"][label_name]["soft_precision"] = (
        #     (tp + fn - missed) / (tp + fn - missed + nonexistent) if tp + fn - missed != 0 else 0
        # )
        # # Computes classif_recall
        # metrics["level2"][label_name]["classif_recall"] = tp / (tp + fn - missed) if tp != 0 else 0
        # # Computes classif_precision
        # metrics["level2"][label_name]["classif_precision"] = tp / (tp + fp - nonexistent) if tp != 0 else 0

    return metrics
