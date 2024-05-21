# Author : Loïc Thiriet

import random
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def split_files_randomly_from_list(
    files_list: list[Path], split_nb: float | int, seed: int | None = 42
) -> tuple[list[Path], list[Path]]:
    """
    Splits a list of file paths randomly into two parts based on a given split proportion or quantity number.

    Args:
        files_list: A list of file paths.
        split_nb: The split proportion (float between 0 and 1) or the number of files to sample (integer > 1).
        seed: The seed value for random sampling. Default is 42.

    Returns:
        files_list: the remaining files after sampling.
        sampled_files: the sampled files.

    Raises:
        ValueError: If split_nb is not within the valid range (0 to 1 for float, > 1 for int).
    """
    # If split_nb == 1, no need to sample files path
    if split_nb == 1:
        return [], files_list

    # If split_nb == 0, no need to sample files path
    elif split_nb == 0:
        return files_list, []

    # If split_nb is between 0 and 1, then samples this proportion of files
    elif split_nb > 0 and split_nb < 1:
        if seed:
            random.seed(seed)
        sampled_files = random.sample(files_list, k=int(split_nb * len(files_list)))
        for file in sampled_files:
            files_list.remove(file)
        return files_list, sampled_files

    # If split_nb is superior to 1, it should be an int and if so, samples this nb of files
    elif split_nb > 1:
        if isinstance(split_nb, int):
            if seed:
                random.seed(seed)
            sampled_files = random.sample(files_list, k=split_nb)
            for file in sampled_files:
                files_list.remove(file)
            return files_list, sampled_files
        else:
            raise ValueError(f"split_nb should be a float between 0 and 1, or a positive int not {split_nb}")

    # split_nb should be between 0 and 1
    else:
        raise ValueError(f"split_nb should be a float between 0 and 1, or a positive int not {split_nb}")


def construct_similar_file_path_list_from_another_dir(
    file_list: list[Path], directory: Path, extension: str
) -> list[Path]:
    """
    Constructs a list of file paths with same names but a different directory and extension.

    Args:
        file_list: List of file paths to be modified.
        directory: The directory path to prepend to the file names.
        extension: The new extension to replace the existing ones.

    Returns:
        A list of modified file paths with the specified directory and extension.
    """
    return [directory / file_path.with_suffix(extension).name for file_path in file_list]


def move_file_list(file_list: list[Path], destination_dir: Path) -> None:
    """
    Moves a list of files to a destination directory.

    Args:
        file_list: List of file paths to move.
        destination_dir: destination directory path to move files into.
    """
    for file in tqdm(file_list):
        shutil.move(file, destination_dir / file.name)


def copy_file_list(file_list: list[Path], destination_dir: Path) -> None:
    """
    Copies a list of files to a destination directory.

    Args:
        file_list: List of file paths to copy.
        destination_di: Destination directory path to copy files into.
    """
    for file in tqdm(file_list):
        shutil.copy2(file, destination_dir / file.name)


def copy_file_to_destination_dir(file_path: Path, destination_dir: Path):
    """
    Copies a file to a specified destination directory.

    Args:
        file_path: The path to the file to be copied.
        destination_dir: The destination directory path where the file will be copied.
    """
    shutil.copy2(file_path, destination_dir / file_path.name)


def copy_file_list_multi_process(
    file_list: list[Path], destination_dir: Path, processes: int = 4, chunksize: int = 1
) -> None:
    """
    Copy a list of files to a destination directory using multiple processes.

    Args:
        file_list: List of file paths to copy.
        destination_dir: Destination directory path to copy files to.
        processes: Number of processes to use for copying. Defaults to 4.
        chunksize: Number of files to send to each process for copying. Defaults to 1.
    """
    # Distribute files accross processes file by file
    with Pool(processes=processes) as pool:
        with tqdm(total=int(np.ceil(len(file_list) / chunksize))) as pbar:
            for _ in pool.imap(
                partial(copy_file_to_destination_dir, destination_dir=destination_dir), file_list, chunksize=chunksize
            ):
                pbar.update()
