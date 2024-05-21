# Author: Loïc Thiriet

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
import scipy.stats as st

# from dash import Dash, Input, Output, callback, dash_table, dcc, html
from plotly.colors import n_colors
from ray import tune
from tqdm.auto import tqdm

from lt_lib.optimization.objective import objective
from lt_lib.schemas.config_files_schemas import OptimizationConfig
from lt_lib.utils.regex_matcher import get_elements_with_regex


def get_result_grid(experiment_path: Path, verbose: bool = True):
    if verbose:
        print(
            f"Can the experiment directory be restored for results analysis: {tune.Tuner.can_restore(experiment_path)}."
        )
    param_space = OptimizationConfig().from_py_file(experiment_path / "optimization_config.py")[0]
    tuner_restored = tune.Tuner.restore(str(experiment_path), objective, param_space=param_space)
    return tuner_restored.get_results()


def rename_config_params_column_name(df: pl.DataFrame):
    param_column_names = get_elements_with_regex("^.*config.*$", df.columns, unique=False)
    param_columns_new_name_mapping = {col_name: f"config/{col_name.rsplit('.')[-1]}" for col_name in param_column_names}
    df = df.rename(param_columns_new_name_mapping)
    return df


def filter_df_with_dict(df: pl.DataFrame, filtering_dict: dict[str, float]):
    for column_name, min_value in filtering_dict.items():
        df = df.filter(pl.col(column_name) >= min_value)
    return df


def keep_non_dominated_points(df: pl.DataFrame, on_columns_to_filter: list[str]):
    points_to_filter = df[on_columns_to_filter].to_numpy()
    num_points = len(points_to_filter)

    # Create a boolean array to track whether each point is dominated
    is_dominated = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        # Check dominance for all other points
        is_dominated[i] = np.any(
            np.any(
                np.all(points_to_filter[i] <= points_to_filter, axis=1)[:, np.newaxis]
                & (points_to_filter[i] < points_to_filter),
                axis=1,
            )
        )

    # Filter non-dominated points
    non_dominated_indices = np.where(~is_dominated)[0]
    return df[non_dominated_indices, :]


def get_best_results_from_ray_results_root_dir(
    ray_results_root_dir: Path, metric: str = "custom_metrics.super_metric", mode: str = "max"
):
    # Intializes the resulting dataframe
    best_results = pl.DataFrame()

    # Processes each sub_dir 1 by 1
    for algo_dir_path in tqdm(list(ray_results_root_dir.glob("ray*"))):
        # Gets the algorithm name and the nb of samples from directory name
        algo, n_samples = algo_dir_path.name.split("_")[1].split("-")
        # Initializes the subirectory dataframe
        experiments_best_results = pl.DataFrame()

        # Processes each experiment 1 by 1
        for experiment_path in algo_dir_path.glob("experiment*"):
            # Gets the experiment dataframe
            experiment_df = (
                get_result_grid(experiment_path, verbose=False)
                .get_best_result(metric=metric, mode=mode)
                .metrics_dataframe
            )

            # Gets the seed value from the experiment name
            if "seed" in experiment_path.name:
                seed = int(experiment_path.name[15:].split("_")[0])
            else:
                seed = None

            # Converts to pandas df to polars df
            experiment_df = pl.from_pandas(experiment_df)
            # Adds a column for the seed number used in the current experiment
            experiment_df = experiment_df.with_columns(seed=pl.lit(seed))
            # Concatenate current experiment df with other already extracted
            experiments_best_results = pl.concat([experiments_best_results, experiment_df])

        # Adds a column for the number of trials (= samples) run in each experiment
        experiments_best_results = experiments_best_results.with_columns(
            algorithm=pl.lit(algo), n_trials=pl.lit(int(n_samples))
        )
        # Concatenate current algorithm best results with other already extracted
        best_results = pl.concat([best_results, experiments_best_results])

    best_results = rename_config_params_column_name(best_results)

    return best_results


def print_best_results_config_params_mean_value(best_results_df: pl.DataFrame, algorithm: str, n_trials: int):
    print(f"The values of the config parameters of the best results are (mean ± std):")
    config_params = get_elements_with_regex("^.*config.*$", list(best_results_df.schema.keys()), unique=False)

    for config_param in config_params:
        param_mean = best_results_df.filter(pl.col("algorithm") == algorithm, pl.col("n_trials") == n_trials)[
            config_param
        ].mean()
        param_std = best_results_df.filter(pl.col("algorithm") == algorithm, pl.col("n_trials") == n_trials)[
            config_param
        ].std()
        print(f"\t- {config_param}: {param_mean:.4f} ± {(param_std if param_std else 0):.4f}")


def compute_mean_and_error_bars(
    df: pl.DataFrame,
    group_by: str | list[str] = ["algorithm", "n_trials"],
    column_to_aggregate: str = "custom_metrics.super_metric",
    confidence_interval_threshold: float = 0.95,
):
    agg_df = (
        df.group_by(group_by)
        .agg(
            n_samples=pl.col(column_to_aggregate).count(),
            mean=pl.col(column_to_aggregate).mean(),
            std=pl.col(column_to_aggregate).std(ddof=1),
        )
        .sort(group_by)
    )

    agg_df = agg_df.with_columns(sem=pl.col("std") / pl.col("n_samples").sqrt())

    agg_df = agg_df.with_columns(
        lower_conf=pl.struct(["n_samples", "mean", "sem"]).map_batches(
            lambda x: st.t.interval(
                confidence=confidence_interval_threshold,
                df=x.struct.field("n_samples") - 1,
                loc=x.struct.field("mean"),
                scale=x.struct.field("sem"),
            )[0]
        )
    )

    agg_df = agg_df.with_columns(
        upper_conf=pl.struct(["n_samples", "mean", "sem"]).map_batches(
            lambda x: st.t.interval(
                confidence=confidence_interval_threshold,
                df=x.struct.field("n_samples") - 1,
                loc=x.struct.field("mean"),
                scale=x.struct.field("sem"),
            )[1]
        )
    )

    agg_df = agg_df.with_columns(half_conf=(pl.col("upper_conf") - pl.col("lower_conf")) / 2)

    return agg_df


### Legacy


def table_degraded_color(df):
    colors = n_colors("rgb(144, 237, 248)", "rgb(10, 152, 168)", 50, colortype="rgb")

    color_df = copy.deepcopy(df)
    # If the column 'number' is present we don't want it to interfer with the coloration of cells
    if "number" == df.columns[0]:
        color_df.drop("number", axis=1, inplace=True)

    # Computes delta per columns
    df_max_by_column = color_df.max()
    df_min_by_column = color_df.min()
    delta = df_max_by_column - df_min_by_column
    delta_max = np.max(delta)

    # Creating a dataframe with int values scaled between 0 and 99 (100 indexes) for coloration purposes
    # This dataframe is scaled according to the biggest column delta value (max - min)
    # This allows to reduce bias between columns with big and small deltas
    color_df = np.round(((color_df - (df_max_by_column - delta_max)) / delta_max) * 49).astype(int)
    cells_color = [np.array(colors)[color_df[col]] for col in color_df.columns]

    # If the column 'number' is present we want to put it in grey and not change its values
    if "number" == df.columns[0]:
        cells_color = [["lightgrey"] * cells_color[0].size] + cells_color
        df.iloc[:, 1:] = df.iloc[:, 1:] * 100
    else:
        df *= 100

    # Round decimals
    df = df.round(decimals=2)

    # Plot the table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{' '.join(col.split('_'))}</b>" for col in df.columns],
                    line_color="darkslategray",
                    fill_color="lightgrey",
                    align="center",
                    font=dict(color="rgb(70, 70, 70)", size=12),
                ),
                columnwidth=50,
                cells=dict(
                    values=[df[col] for col in df.columns],
                    line_color="darkslategray",
                    fill_color=cells_color,
                    align="center",
                    font=dict(color="white", size=11),
                ),
            ),
        ],
    )

    fig.show()
