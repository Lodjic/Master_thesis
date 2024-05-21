import os
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
from shapely.geometry import box
from tqdm.auto import tqdm

from lt_lib.data.datasets import POLARS_GTS_SCHEMA
from lt_lib.utils.regex_matcher import get_elements_with_regex


def grid_tile_one_image(
    base_img_path: Path,
    bboxes_df: pl.DataFrame,
    tile_shape: int | list[int] | tuple[int],
    overlap: int | list[int] | tuple[int],
    min_bbox_area_on_tile: float,
    img_saving_dir: Path,
    clip_boxes: bool = True,
    png_compression_level: int = 3,
) -> pl.DataFrame:
    """
    Divides one image into a grid of tiles, and affects and converts the bounding boxes to the correct tiles.

    Args:
        base_img_path: Path to the base image.
        bboxes_df: Dataframe containing the ground truth (bounding boxes + labels) information.
        tile_shape: Shape of the tiles.
        overlap: Overlap between tiles.
        min_bbox_area_on_tile: Minimum area ratio for bounding boxes to be considered to appear on a tile.
        img_saving_dir: Directory in which to save the tiles.
        clip_boxes: Whether to clip bounding boxes to the tile boundaries. Defaults to True.
        png_compression_level: Compression level for saving PNG tiles. Defaults to 3.

    Returns:
        A DataFrame containing ground truth (bounding boxes + labels) information for each processed tile.
    """
    img_bboxes_df = bboxes_df.filter(pl.col("img_name") == base_img_path.name)
    additional_info_columns = get_elements_with_regex("^(?!.*bbox.*|.*id.*).*$", img_bboxes_df.columns, unique=False)

    # If tile_shape and overlap are passed as unique values, modifies them as a tuple
    if isinstance(tile_shape, int):
        tile_shape = (tile_shape, tile_shape)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    # Reads the big image
    base_img = cv2.imread(str(base_img_path))

    # Creates a DataFrame to store the new bounding boxes and labels
    tiles_bboxes = []

    # Calculates the number of tiles in each dimension
    num_tiles_x = (base_img.shape[1] - 1) // (tile_shape[0] - overlap[0]) + 1
    num_tiles_y = (base_img.shape[0] - 1) // (tile_shape[1] - overlap[1]) + 1

    # Iterates over each tile
    for i, j in product(range(num_tiles_x), range(num_tiles_y)):
        # Calculates the starting pixel coordinates for the current tile
        start_i = i * (tile_shape[0] - overlap[0])
        start_j = j * (tile_shape[1] - overlap[1])

        # Creates tile path
        tile_path = (
            img_saving_dir
            / f"{base_img_path.with_suffix('').name}_{tile_shape[0]}x{tile_shape[1]}_{start_i}-{start_j}_tile.png"
        )

        # Adjusts the position of the last tiles in each row and column
        if i == num_tiles_x - 1:
            start_i = max(base_img.shape[1] - tile_shape[0], 0)
        if j == num_tiles_y - 1:
            start_j = max(base_img.shape[0] - tile_shape[1], 0)

        # Defines the tile bounding box as a Shapely geometry object
        tile_bbox = box(start_i, start_j, start_i + tile_shape[0], start_j + tile_shape[1])

        # Iterates over each bounding box
        for row in img_bboxes_df.iter_rows(named=True):
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = (
                row["bbox_xmin"],
                row["bbox_ymin"],
                row["bbox_xmax"],
                row["bbox_ymax"],
            )
            additional_bbox_info = [row[col] for col in additional_info_columns]

            # Defines the bounding box as a Shapely geometry object
            original_bbox = box(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

            # Checks if the bounding box intersects with the current tile
            if original_bbox.intersects(tile_bbox):
                # Calculates the intersection of the bounding box and the tile
                intersection = original_bbox.intersection(tile_bbox)

                # Calculates the area ratio
                area_ratio = intersection.area / original_bbox.area

                # Checks if the area ratio is above the threshold
                if area_ratio >= min_bbox_area_on_tile:
                    # If clipping is enabled, clip the bounding box to the tile boundaries
                    if clip_boxes:
                        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = intersection.bounds

                    # Converts bounding box coordinates to tile coordinates
                    bbox_xmin -= start_i
                    bbox_xmax -= start_i
                    bbox_ymin -= start_j
                    bbox_ymax -= start_j

                    # Adds the new bounding box and label to the DataFrame
                    tiles_bboxes.append(
                        (tile_path.name, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, *additional_bbox_info)
                    )

        tile = base_img[start_j : start_j + tile_shape[1], start_i : start_i + tile_shape[0], :]

        cv2.imwrite(str(tile_path), tile, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])

    # Creates a new Polars DataFrame for the processed bounding boxes
    tiles_bboxes_df = pl.DataFrame(
        tiles_bboxes, schema=["img_name", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", *additional_info_columns]
    )

    return tiles_bboxes_df.with_row_index("sub_id")


TILE_TYPE_DISPATCHER = {
    "grid_tiling": grid_tile_one_image,
}


def tile_images_mono_process(
    root_data_dir: Path,
    tiling_type: str,
    tiling_args: dict[str, Any],
    saving_dir: Path | None = None,
    images_extension: str = ".png",
) -> None:
    """
    Tile all the images present in a directory using the specified tiling method and arguments with a single process.

    Args:
        root_data_dir: The root directory containing images and annotations.
        tiling_type: The type of tiling method to be used.
        tiling_args: A dictionary containing arguments specific to the tiling method.
        saving_dir: The directory where tiled images will be saved. If None, a default directory will be created.
            Defaults to None.
        images_extension: Image file extension. Defaults to '.png'.
    """
    # Defining paths
    imgs_dir_path = root_data_dir / "imgs"
    gts_dir_path = root_data_dir / "annotations"

    if not saving_dir:
        saving_dir = root_data_dir / "tiled_dataset"

    if not os.path.isdir(saving_dir / "imgs"):
        os.makedirs(saving_dir / "imgs")

    base_imgs_gts = pl.read_csv(gts_dir_path / "gts.csv")
    tiles_gts = pl.DataFrame()

    # Tile images 1 by 1 and concatenate resulting tiles_gts
    for img_path in tqdm(list(imgs_dir_path.glob(f"*{images_extension}"))):
        tiles_gts = pl.concat(
            [
                tiles_gts,
                TILE_TYPE_DISPATCHER[tiling_type](
                    img_path, base_imgs_gts, img_saving_dir=saving_dir / "imgs", **tiling_args
                ),
            ]
        )

    tiles_gts = tiles_gts.with_row_index("id")

    if not os.path.isdir(saving_dir / "annotations"):
        os.mkdir(saving_dir / "annotations")

    tiles_gts = tiles_gts.cast(POLARS_GTS_SCHEMA)
    tiles_gts.write_csv(saving_dir / "annotations/gts.csv")


def tile_images(
    root_data_dir: Path,
    tiling_type: str,
    tiling_args: dict[str, Any],
    images_extension: str = ".png",
    processes: int = 4,
    chunksize: int = 1,
    saving_dir: Path | None = None,
) -> None:
    """
    Tile all the images present in a directory using the specified tiling method and arguments with multiple processes.

    Args:
        root_data_dir: The root directory containing images and annotations.
        tiling_type: The type of tiling method to be used.
        tiling_args: A dictionary containing arguments specific to the tiling method.
        images_extension: Image file extension. Defaults to '.png'.
        processes: The number of processes to use for the parallel execution. Defaults to 4.
        chunksize: The size of chunks to process in parallel. Default is 1.
        saving_dir: The directory where tiled images will be saved. If None, a default directory will be created.
            Defaults to None.
    """
    # Defining paths
    imgs_dir_path = root_data_dir / "imgs"
    gts_dir_path = root_data_dir / "annotations"

    if not saving_dir:
        saving_dir = root_data_dir / "tiled_dataset"

    if not os.path.isdir(saving_dir / "imgs"):
        os.makedirs(saving_dir / "imgs")

    base_imgs_gts = pl.read_csv(gts_dir_path / "gts.csv")
    # tiles_gts = pl.DataFrame()
    tiles_gts_dfs = []

    # Tile images 1 by 1 and concatenate resulting tiles_gts
    img_path_list = list(imgs_dir_path.glob(f"*{images_extension}"))
    with Pool(processes=processes) as pool:
        with tqdm(total=int(np.ceil(len(img_path_list) / chunksize))) as pbar:
            for one_base_img_tiles_gts in pool.imap(
                partial(
                    TILE_TYPE_DISPATCHER[tiling_type],
                    bboxes_df=base_imgs_gts,
                    img_saving_dir=saving_dir / "imgs",
                    **tiling_args,
                ),
                img_path_list,
                chunksize=chunksize,
            ):
                # tiles_gts = pl.concat([tiles_gts, one_base_img_tiles_gts])
                tiles_gts_dfs.append(one_base_img_tiles_gts)
                pbar.update()

    tiles_gts = pl.concat(tiles_gts_dfs)
    tiles_gts = tiles_gts.with_row_index("id")

    if not os.path.isdir(saving_dir / "annotations"):
        os.mkdir(saving_dir / "annotations")

    tiles_gts = tiles_gts.cast(POLARS_GTS_SCHEMA)
    tiles_gts.write_csv(saving_dir / "annotations/gts.csv")
