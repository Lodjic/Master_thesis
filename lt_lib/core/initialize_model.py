from pathlib import Path
from typing import Literal

import polars as pl
import torch
from loguru import logger
from torchvision.models.detection.retinanet import ResNet50_Weights

from lt_lib.architecture.retinanet import retinanet_with_resnet50
from lt_lib.utils.load_and_save import load_pytorch_checkpoint

MODELS = {
    "retinanet": retinanet_with_resnet50,
}

WEIGHTS = {
    "retinanet": {"imagenet": ResNet50_Weights.IMAGENET1K_V1},
}


def initialize_model(
    # Data params
    root_data_dir: Path,
    # Model params
    model: Literal["retinanet"],
    model_parameters: dict[str, list[float] | tuple[float]],
    trainable_backbone_layers: int,
    # Weights loading params
    saved_checkpoint_path: str | Path | None,
    model_loading_type: Literal["model", "model_state_dict"] | None,
    weights_backbone: Literal["retinanet"] | None,
) -> tuple[torch.nn.Module, Path | None]:
    """
    Initializes a neural network model with the specified parameters.

    Args:
        root_data_dir: Data directory root path.
        model: Model architecture name.
        model_parameters: Dictionary containing the model parameters.
        trainable_backbone_layers: Number of trainable backbone layers.
        saved_checkpoint_path: Path to a saved checkpoint file from which to restore the model. Default to None.
        model_loading_type: Str specifying how to load the model. Default to None.
        weights_backbone: Name of the backbone weights to load. Default to None.

    Returns:
        model: The initialized model
        saved_checkpoint_path: The path to the saved checkpoint file used, if any.
    """
    # Get the number of classes from the gts
    gts_csv_path = root_data_dir / "train/annotations/gts.csv"
    gts = pl.read_csv(gts_csv_path)
    n_classes = len(gts["label"].unique()) + 1
    model_name = model
    weights_backbone_name = weights_backbone

    # If a saved_checkpoint_path has been passed load the model weights following the loading scheme specified
    if saved_checkpoint_path:
        saved_checkpoint_path = Path(saved_checkpoint_path)
        checkpoint = load_pytorch_checkpoint(saved_checkpoint_path)

        if model_loading_type == "model":
            model = checkpoint["model"]
            logger.info(
                f"Model '{model_name}' is resumed from the model saved in checkpoint file '{saved_checkpoint_path}'."
            )

        elif model_loading_type == "model_state_dict":
            weights = checkpoint["model_state_dict"]
            weights_backbone = None
            logger.info(
                f"Model '{model_name}' is going to be resumed from the state_dict saved in checkpoint file "
                + f"'{saved_checkpoint_path}'."
            )
        else:
            raise ValueError(f"Parameter model_loading_type='{model_loading_type}' is not yet implemented!")

    # If no saved_checkpoint_path is specified, check for backbone weights
    else:
        if weights_backbone:
            weights_backbone = WEIGHTS[model][weights_backbone]
            weights = None
            logger.info(
                f"Model '{model_name}' is going to get loaded with pre-trained backbone weights: "
                + f"'{weights_backbone_name}'."
            )
        else:
            weights_backbone = None
            weights = None

    # If the model has not been loaded, instantiate the specified model with eventually some pre-trained weights
    if isinstance(model, str):
        if model in MODELS.keys():
            model = MODELS[model](
                num_classes=n_classes,
                weights=weights,
                weights_backbone=weights_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                **model_parameters,
            )
        else:
            raise ValueError(f"Parameter model='{model_name}' is not yet implemented!")

    return model, saved_checkpoint_path
