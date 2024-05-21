# Author : Loïc Thiriet

import os
import shutil
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from tqdm.auto import tqdm

from lt_lib.data.data_extraction import (
    get_annotations_dataframe_from_geojson,
    get_bboxes_df_from_one_synthetic_xml,
)
from lt_lib.data.data_utils import (
    construct_similar_file_path_list_from_another_dir,
    copy_file_list,
    move_file_list,
    split_files_randomly_from_list,
)
from lt_lib.data.datasets import POLARS_GTS_SCHEMA
from lt_lib.utils.load_and_save import save_dict_as_json

#################################################################################
###################### Functions to extract and gather gts ######################
#################################################################################


### Functions for real data


def get_all_annotations_from_rareplanes_geojsons(
    root_dir_path: Path, tiled_version: bool = True, imgs_extension: str = ".png", save_to_file: bool = True
) -> tuple[gpd.GeoDataFrame, dict[str : dict[str : list[int]]]]:
    """
    Gather annotations from RarePlanes GeoJSONs into 1 geodataframe and 1 minimal information dictionary.

    Args:
        root_dir_path: The root directory path containing images and annotations.
        tiled_version: Whether the annotations are gathered from tiled version of the dataset. Defaults to True.
        imgs_extension: Extension of image files. Defaults to ".png".
        save_to_file: Whether to save the annotations to a file. Defaults to True.

    Returns:
        global_gdf: The global geodataframe containing all annotations.
        minimal_gts_json: A dictionary containing the main columns of `global_gdf` which are 'ids', 'bboxes', and
            'labels'.
    """
    logger.info("Launching the gathering of all annotations into 1 unique geojson file.")
    # Defining paths
    images_dir_path = root_dir_path / "imgs"
    geojsons_dir_path = root_dir_path / "annotations"

    # Construct the geodf containing all the gts
    global_gdf = gpd.GeoDataFrame()
    if tiled_version:
        for image_path in tqdm(list(images_dir_path.glob(f"*{imgs_extension}"))):
            # Construct necessary paths and file names
            xml_path = image_path.with_suffix(f"{imgs_extension}.aux.xml")
            geojson_path = geojsons_dir_path / image_path.with_suffix(".geojson").name
            # Get the annotations dataframe
            gdf = get_annotations_dataframe_from_geojson(image_path.name, geojson_path, xml_path)
            global_gdf = pd.concat([global_gdf, gdf], ignore_index=True)
    else:
        raise ValueError("Annotations extraction from the non tiled dataset folder is not yet implemented.")
    global_gdf["id"] = np.arange(len(global_gdf))

    # Exctract a minimal dataframe containing only the labels and bboxes positions
    minimal_gts_df = pl.from_pandas(
        global_gdf[["id", "sub_id", "img_name", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "label"]],
        schema_overrides=POLARS_GTS_SCHEMA,
    )

    # Constructing a json version of the minimal dataframe
    minimal_gts_json = {}
    for img_name in np.unique(minimal_gts_df["img_name"].to_list()):
        sub_img_df = minimal_gts_df.filter(pl.col("img_name") == img_name)
        minimal_gts_json[img_name] = {
            "ids": sub_img_df["id"].to_list(),
            "bboxes": sub_img_df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy().tolist(),
            "labels": sub_img_df["label"].to_list(),
        }

    # Save the files into 2 csv and 1 json + 1 json dict
    if save_to_file:
        global_gdf.to_file(geojsons_dir_path / "global_annotations.geojson", driver="GeoJSON")
        # minimal_gts_df.to_csv(geojsons_dir_path / "gts.csv", index=False)
        minimal_gts_df.write_csv(geojsons_dir_path / "gts.csv", include_header=True)
        save_dict_as_json(geojsons_dir_path / "gts.json", minimal_gts_json)

    return global_gdf, minimal_gts_json


### Functions for synthetic data


def get_all_synthetic_gts_from_rareplanes_xmls(
    annotations_dir_path: Path,
    imgs_extension: str = ".png",
    clip: bool = True,
    wingspan_label_bins: list[float] = [0, 15, 36],
    imgs_resolution: float = 0.3,
    save_to_file: bool = True,
) -> pl.DataFrame:
    """
    Extracts ground truth data (bboxes + labels) from XML annotations files of the synthetic part of RarePlanes dataset.

    Args:
        annotations_dir_path: Path to the directory containing XML annotations files.
        imgs_extension: Image file extension. Defaults to ".png".
        clip: Whether to clip the bounding boxes to image boundaries. Defaults to True.
        wingspan_label_bins: Bins for wingspan labeling. Default bins values are obtained from the wingspan
            thresholds given by the rareplanes public user guide. Bin values are divided by the image's resolution.
            Defaults to [0, 15, 36]. Defaults to [0, 15, 36].
        imgs_resolution: The resolution of the images in meters per pixel. Defaults to 0.3.
        save_to_file: Whether to save the annoations to a CSV file. Defaults to True.

    Returns:
        bboxes_df: Dataframe containing ground truth data (bounding boxes + labels).
    """
    bboxes_df = pl.DataFrame()
    for xml_path in tqdm(list(annotations_dir_path.glob("*.xml"))):
        bboxes_df = pl.concat(
            [
                bboxes_df,
                get_bboxes_df_from_one_synthetic_xml(
                    xml_path, xml_path.with_suffix(imgs_extension).name, clip, wingspan_label_bins, imgs_resolution
                ),
            ]
        )

    bboxes_df = bboxes_df.with_row_index("id")

    # Save the file into a csv
    if save_to_file:
        bboxes_df.write_csv(annotations_dir_path / "gts.csv", include_header=True)

    return bboxes_df


##############################################################################
###################### Functions to run train-val split ######################
##############################################################################


### Functions for real data


def get_rareplanes_real_xml_and_geojson_lists(img_list: list) -> tuple[list[Path], list[Path]]:
    """
    Constructs lists of file paths for the XML and GeoJSON files corresponding to a given list of image paths.

    Args:
        img_list: A list of image file paths.

    Returns:
        xml_list: A list of XML file paths.
        geojson_list: A list of GeoJSON file paths.
    """
    xml_list = construct_similar_file_path_list_from_another_dir(
        img_list, img_list[0].parent, f"{img_list[0].suffix}.aux.xml"
    )
    geojson_list = construct_similar_file_path_list_from_another_dir(
        img_list, img_list[0].parent.parent / "annotations", f".geojson"
    )
    return xml_list, geojson_list


def move_rareplanes_real_img_xml_and_geojson_files(
    img_list: list[Path], xml_list: list[Path], geojson_list: list[Path], destination_dir_name: Literal["train", "val"]
) -> None:
    """
    Moves image, XML, and GeoJSON files to a specified destination directory.

    Args:
        img_list: List of image file paths.
        xml_list: List of XML annotation file paths.
        geojson_list: List of GeoJSON annotation file paths.
        destination_dir_name: Destination directory apth. Should be either "train" or "val".
    """
    root_imgs_dir = img_list[0].parent.parent

    # Creates train and val directories
    if not os.path.isdir(root_imgs_dir / f"{destination_dir_name}/imgs"):
        os.makedirs(root_imgs_dir / f"{destination_dir_name}/imgs")
    if not os.path.isdir(root_imgs_dir / f"{destination_dir_name}/annotations"):
        os.mkdir(root_imgs_dir / f"{destination_dir_name}/annotations")

    # Moves files
    move_file_list(img_list, root_imgs_dir / f"{destination_dir_name}/imgs")
    move_file_list(xml_list, root_imgs_dir / f"{destination_dir_name}/imgs")
    move_file_list(geojson_list, root_imgs_dir / f"{destination_dir_name}/annotations")


def train_val_split_and_move_rareplanes_real_img_xml_and_geojson_files(
    imgs_dir: Path, imgs_extension: str, val_proportion: float | int, seed: int | None = 42
) -> None:
    """
    Splits image and annotation files into 2 sets (usually training and validation), and places them into appropriate
    directories.

    Args:
        imgs_dir: Path to the directory containing images and annotations.
        imgs_extension: Image file extension.
        val_proportion: The split proportion (float between 0 and 1) or the number of files to sample (integer > 1).
        seed: The seed value for random sampling. Default is 42.
    """
    # Splits images and annotations files
    img_list = list(imgs_dir.glob(f"*{imgs_extension}"))
    train_img_list, val_img_list = split_files_randomly_from_list(img_list, val_proportion, seed)
    train_xml_list, train_geojson_list = get_rareplanes_real_xml_and_geojson_lists(train_img_list)
    val_xml_list, val_geojson_list = get_rareplanes_real_xml_and_geojson_lists(val_img_list)

    # Moves files in the correct directories
    move_rareplanes_real_img_xml_and_geojson_files(train_img_list, train_xml_list, train_geojson_list, "train")
    move_rareplanes_real_img_xml_and_geojson_files(val_img_list, val_xml_list, val_geojson_list, "val")


### Functions for synthetic data


def copy_rareplanes_synthetic_img_and_xml_files(
    img_list: list[Path], xml_list: list[Path], destination_root_data_dir: Path | None = None
) -> None:
    """
    Copies synthetic image and XML annotation files from the provided lists to a specified destination directory.

    Args:
        img_list: List of synthetic image file paths.
        xml_list: List of XML annotation file paths corresponding to the synthetic images.
        destination_root_data_dir: Destination root directory where the files should be copied. If not provided, a
            default directory named "sampled_synthetic_data" will be created in the parent directory of the first image
            file's parent directory. Defaults to None.

    Note:
        This function assumes that each image file in `img_list` has a corresponding XML annotation file
        in `xml_list` at the same index.
    """
    root_data_dir = img_list[0].parent.parent

    if not destination_root_data_dir:
        destination_root_data_dir = root_data_dir / "sampled_synthetic_data"

    # Creates imgs and annotations directories
    if not os.path.isdir(destination_root_data_dir / "imgs"):
        os.makedirs(destination_root_data_dir / "imgs")
    if not os.path.isdir(destination_root_data_dir / "annotations"):
        os.mkdir(destination_root_data_dir / "annotations")

    # Moves files
    copy_file_list(img_list, destination_root_data_dir / "imgs")
    copy_file_list(xml_list, destination_root_data_dir / "annotations")


def copy_fraction_of_random_rareplanes_synthetic_img_and_xml_files(
    imgs_dir: Path,
    imgs_extension: str,
    fraction_to_copy: float,
    seed: int | None = 42,
    destination_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """
    Copies a fraction of random rareplanes synthetic image and XML files to a destination directory.

    Args:
        imgs_dir: Path to the directory containing the images and annotations files.
        imgs_extension: Image file extension.
        fraction_to_copy: Fraction of files to copy from the source directory.
        seed: The seed value for random sampling. Default is 42.
        destination_dir: Destination directory path where the files will be copied. If None, files will be copied to
            a directory named "sampled_synthetic_data" in the parent directory of the first image file's parent
            directory. Defaults to None.

    Returns:
        A tuple containing two lists: the paths of the copied image files and the paths of the copied XML files.
    """
    # Splits images and annotations files
    img_list = list(imgs_dir.glob(f"*{imgs_extension}"))
    _, img_list_to_copy = split_files_randomly_from_list(img_list, fraction_to_copy, seed)
    xml_list_to_copy = construct_similar_file_path_list_from_another_dir(
        img_list_to_copy, img_list_to_copy[0].parent.parent.parent / "annotations", ".xml"
    )

    # Copies files in the correct directories
    copy_rareplanes_synthetic_img_and_xml_files(img_list_to_copy, xml_list_to_copy, destination_dir)


def copy_nb_of_random_rareplanes_synthetic_tile_and_gts_files(
    root_data_dir: Path,
    imgs_extension: str,
    select_nb: int,
    seed: int | None = 42,
    destination_root_data_dir: Path | None = None,
) -> None:
    """
    Copies a specified number of randomly selected images along with their annotations to a destination directory.

    Args:
        root_data_dir: The root data directory containing images and annotations.
        imgs_extension: Image file extension.
        select_nb: The number of images to select randomly.
        seed: The seed value for random sampling. Default is 42.
        destination_root_data_dir: The root directory where the selected images and annotations will be copied.
            If None, files will be copied to a subdirectory named 'sampled_synthetic_tiled_data' crated in the
            `root_data_dir` directory. Defaults to None.
    """
    # Get the image names that have objects on them
    gts_df = pl.read_csv(root_data_dir / "annotations/gts.csv")
    gts_df = gts_df.drop_nulls()
    img_name_list = gts_df["img_name"].unique().to_list()
    # Randomly selects image names
    _, selected_img_name_list = split_files_randomly_from_list(img_name_list, select_nb, seed)
    # Filters the gts_df to keep only the gts of the selected images
    gts_df = gts_df.filter(pl.col("img_name").is_in(selected_img_name_list))

    # Constructs the images' paths
    selected_img_path_list = construct_similar_file_path_list_from_another_dir(
        file_list=list(map(Path, selected_img_name_list)), directory=root_data_dir / "imgs", extension=imgs_extension
    )

    # Infer a destinations dir if not specified
    if not destination_root_data_dir:
        destination_root_data_dir = root_data_dir / "sampled_synthetic_tiled_data"

    # Creates imgs and annotations directories
    if not os.path.isdir(destination_root_data_dir / "imgs"):
        os.makedirs(destination_root_data_dir / "imgs")
    if not os.path.isdir(destination_root_data_dir / "annotations"):
        os.mkdir(destination_root_data_dir / "annotations")

    # Copies imgs in the correct directory
    copy_file_list(selected_img_path_list, destination_root_data_dir / "imgs")
    # Writes the gts_df the annotations directory
    gts_df.write_csv(destination_root_data_dir / "annotations/gts.csv")


##################################################################################
###################### Functions to adjust synthetic labels ######################
##################################################################################


def adjust_labels_base_on_wingspan(
    gts_with_wingspan_path: Path,
    gts_to_adjust_path: Path,
    wingspan_label_bins: list[float] = [0, 15, 36],
    imgs_resolution: float = 0.3,
    save_to_file: bool = True,
) -> None:
    """
    Adjusts labels based on wingspan values.

    Args:
        gts_with_wingspan_path: Path to the CSV file containing ground truth data, including wingspan information.
        gts_to_adjust_path: Path of the CSV file containing ground truth data to adjust.
        wingspan_label_bins: Bins for wingspan labeling. Default bins values are obtained from the wingspan
            thresholds given by the rareplanes public user guide. Bin values are divided by the image's resolution.
            Defaults to [0, 15, 36]. Defaults to [0, 15, 36].
        imgs_resolution: Resolution of the images in meters per pixel. Default is 0.3.
        save_to_file: Whether to save the adjusted ground truth data to a CSV file. Defaults to True.
    """
    gts_with_wingspan = pl.read_csv(gts_with_wingspan_path)
    gts_to_adjust = pl.read_csv(gts_to_adjust_path)

    # Bins values can be chosen with the help of the wingspan thresholds given by the rareplanes public user guide and
    # experiments. Bin values are divided by 0.3 because synthetic pixels have a resolution of 0.3m
    wingspan_label_bins = np.array(wingspan_label_bins) / imgs_resolution

    for row in tqdm(gts_to_adjust.iter_rows(named=True), total=len(gts_to_adjust)):
        id = row["id"]
        row_with_wingspan = gts_with_wingspan.row(by_predicate=pl.col("id") == id, named=True)
        row["label"] = np.digitize(row_with_wingspan["wingspan"], wingspan_label_bins)

    if save_to_file:
        gts_to_adjust.write_csv(gts_to_adjust_path.parent / "gts_adjusted.csv")
