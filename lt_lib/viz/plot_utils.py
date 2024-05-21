from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import scienceplots


def plot_px_hist(
    object_to_plot: pl.DataFrame | pd.DataFrame | np.ndarray | list[Any],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    nbins: int = 100,
    bargap: int = 0,
    **kwargs,
) -> None:
    """Function to plot histograms with plotly.

    Note: should be used with kwargs for more personnalisation. Example: x='column_name', color='column_name' or
    histnorm='percent'.

    Args:
        object_to_plot: dataframe or array or list of values to plot as a histogram.
        x_label: label of the x-axis. If None will be determined automatically by plotly. Defaults to None.
        y_label: label of the y-axis. If None will be determined automatically by plotly.Defaults to None.
        title: title of the graph. If None will be determined automatically by plotly. Defaults to None.
        nbins: number of bins to plot. Note that plotly search for the "best" value below this nb. Defaults to 100.
        bargap: values of the gap between bars for an increase readability. Default to 0.
        **kwargs: kwargs to be passed to the px.histogram() function.
    """
    fig = px.histogram(object_to_plot, nbins=nbins, **kwargs)
    if title is not None:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=20), xanchor="auto"))
    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)
    if y_label is not None:
        if "histnorm" in kwargs and kwargs["histnorm"] == "percent":
            fig.update_layout(yaxis_title=f"{y_label} (%)")
        else:
            fig.update_layout(yaxis_title=y_label)
    if bargap > 0:
        fig.update_layout(bargap=bargap)
    fig.show()


def plot_px_bar(
    object_to_plot: pl.DataFrame | pd.DataFrame | np.ndarray | list[Any],
    x: str,
    y: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    bargap: int = 0,
    **kwargs,
) -> None:
    """Function to plot histograms with plotly.

    Note: should be used with kwargs for more personnalisation. Example: x='column_name', color='column_name' or
    histnorm='percent'.

    Args:
        object_to_plot: dataframe or array or list of values to plot as a histogram.
        x: column name to put on x-axis.
        y: column(s) name(s) to put in y-axis
        x_label: label of the x-axis. If None will be determined automatically by plotly. Defaults to None.
        y_label: label of the y-axis. If None will be determined automatically by plotly.Defaults to None.
        title: title of the graph. If None will be determined automatically by plotly. Defaults to None.
        bargap: values of the gap between bars for an increase readability. Default to 0.
        **kwargs: kwargs to be passed to the px.histogram() function.
    """
    fig = px.bar(object_to_plot, x=x, y=y, **kwargs)
    if title is not None:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=20), xanchor="auto"))
    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)
    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)
    if bargap > 0:
        fig.update_layout(bargap=bargap)
    fig.show()


def plot_px_line(
    object_to_plot: pl.DataFrame | pd.DataFrame | np.ndarray | list[Any],
    x: str,
    y: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    **kwargs,
) -> None:
    fig = px.line(object_to_plot, x=x, y=y, **kwargs)
    if title is not None:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=20), xanchor="auto"))
    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)
    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)
    fig.show()


def plot_plt_line(
    df: pl.DataFrame | pd.DataFrame,
    x: str,
    y: str | list[str],
    group: str | None = None,
    yerr: str | None = None,
    yerr_lower: str | None = None,
    yerr_upper: str | None = None,
    mpl_rc_linewidth: float | int = 0.7,
    mpl_rc_markeredgewidth: float | int = 0.5,
    marker: str = "x",
    markersize: float | int = 2,
    capsize: float | int = 2,
    fill_opacity: float | int = 0.15,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    **kwargs,
) -> None:
    # Use ieee style provided by scienceplots
    with plt.style.context(["science", "ieee", "no-latex"]):
        # Overrides linewidth and markeredgewidth
        mpl.rc("lines", linewidth=mpl_rc_linewidth, markeredgewidth=mpl_rc_markeredgewidth)

        # If plots are grouped on a column values
        if group:
            # Sort grouping values
            if df.schema[group] == pl.String or df.schema[group] == pl.Utf8:
                grouping_values = df[group].unique().to_list()
                grouping_values_lowercase = list(map(str.lower, grouping_values))
                grouping_values = list(np.array(grouping_values)[np.argsort(grouping_values_lowercase)])
            else:
                grouping_values = df[group].unique().sort().to_list()

            # For each grouping value plots the error bars
            for value in grouping_values:
                filtered_df = df.filter(pl.col(group) == value).sort(x, descending=False)
                plt.errorbar(
                    x=x,
                    y=y,
                    yerr=yerr,
                    data=filtered_df,
                    label=value,
                    marker=marker,
                    markersize=markersize,
                    capsize=capsize,
                    **kwargs,
                )
                # If lower and upper bounds are provided, fills the space between error bounds with a transparent color
                if yerr_lower and yerr_upper:
                    plt.fill_between(
                        x=x,
                        y1=yerr_lower,
                        y2=yerr_upper,
                        alpha=fill_opacity,
                        data=filtered_df,
                    )

        # If no grouping column is provieded just plots 1 line error bar
        else:
            plt.errorbar(
                x=x,
                y=y,
                yerr=yerr,
                data=filtered_df,
                label=value,
                marker=marker,
                markersize=markersize,
                capsize=capsize,
                **kwargs,
            )
            # If lower and upper bounds are provided, fills the space between error bounds with a transparent color
            if yerr_lower and yerr_upper:
                plt.fill_between(
                    x=x,
                    y1=yerr_lower,
                    y2=yerr_upper,
                    alpha=fill_opacity,
                    data=filtered_df,
                )

        # Adds legend, axis labels and title before showing the plot
        plt.legend()
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        if title is not None:
            plt.title(title)
        plt.show()


def plot_plt_bar(
    df: pl.DataFrame | pd.DataFrame,
    x: str | list[str],
    group: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    **kwargs,
) -> None:
    # Use ieee style provided by scienceplots
    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        # Overrides linewidth and markeredgewidth
        mpl.rc("xtick.minor", top=False, bottom=False)
        mpl.rc("xtick.major", top=False, bottom=True)
        mpl.rc("xtick", direction="out")
        # mpl.rc("font", size=20)

        if isinstance(x, str):
            x = [x]
        else:
            x = list(x)
        df = df.select([group] + x)

        # If plots are grouped on a column values
        if group:
            nb_values_x_axis = len(x)
            x_axis = np.arange(nb_values_x_axis)  # the label locations
            width = 1 / (nb_values_x_axis + 1)
            multiplier = 0

            fig, ax = plt.subplots(layout="constrained")

            grouping_attributes = df[group].to_list()
            for grouping_attribute in grouping_attributes:
                offset = width * multiplier
                bars = df.row(by_predicate=(pl.col(group) == grouping_attribute))[1:]
                rects = ax.bar(x_axis + offset, bars, width, label=grouping_attribute)
                ax.bar_label(rects, padding=2)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xticks(x_axis + width, x)
            ylim = ax.get_ylim()[1]
            ax.set_ylim(0, int(ylim * 1.05))

        # Adds legend, axis labels and title before showing the plot
        plt.legend()
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)
        plt.show()
