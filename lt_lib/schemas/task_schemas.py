from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from typing_extensions import Annotated

INPUT_TYPES = ["dataset", "model", "model_state_dict", "predictions", "gts"]


class TrainTaskConfig(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")
    # To resolve a warning of the protected namespace 'model_'
    model_config["protected_namespaces"] = ()

    # Seed used to generate random number
    manual_seed: Optional[int | float | None] = Field(default=None)

    # Data parameters
    images_extension: Optional[str] = Field(default=".png")
    batch_size: int
    n_epochs: int

    # Training parameters
    loss_reduction_factor: Optional[Annotated[float | int, Field(ge=1, le=10)]] = Field(default=1)
    loss_reduction_sign_indicator: Optional[Literal[-1, 1]] = Field(default=1)
    # weight_classification_loss: Optional[float] = Field(default=1.0)
    # weight_regression_loss: Optional[float] = Field(default=1.0)
    optimizer: Optional[Literal["Adam"]] = Field(default="Adam")
    optimizer_params: Optional[dict[str, Any]] = Field(default={"lr": 1e-5})
    lr_scheduler: Optional[Literal["ReduceLROnPlateau"] | None] = Field(default=None)
    lr_scheduler_params: Optional[dict[str, Any]] = Field(default={})

    # Matching parameters in case you want to get training recall and precision
    matching_iou_threshold: Optional[float] = Field(default=0.1)

    # Wandb logging parameter
    wandb_log_model_checkpoint: Optional[bool] = Field(default=True, strict=True)

    # Saving parameters
    model_saving_type: Optional[Literal["model", "model_state_dict", "both"]] = Field(default="model_state_dict")
    save_optimizer: Optional[bool] = Field(default=True, strict=True)
    saving_frequency: Optional[int] = Field(default=1)
    checkpoint_scoring_metric: Optional[str | None] = Field(default="val.loss")
    checkpoint_scoring_order: Optional[Literal["min", "max"] | None] = Field(default="min")
    checkpoint_file_name: Optional[str] = Field(default="model")

    # Disable tqdm
    disable_epoch_tqdm: Optional[bool | None] = Field(default=None)
    disable_batch_tqdm: Optional[bool | None] = Field(default=None)

    @field_validator("wandb_log_model_checkpoint", "save_optimizer", mode="after")
    @classmethod
    def validate_bool_default_value_to_true(cls, param: bool | None) -> bool:
        if param is None:
            return True
        else:
            return param

    # @field_validator("weight_classification_loss", "weight_regression_loss", mode="after")
    # @classmethod
    # def validate_float_default_value_to_1(cls, param: float | None) -> float:
    #     if param is None:
    #         return 1.0
    #     else:
    #         return param

    @field_validator("checkpoint_scoring_order", mode="after")
    @classmethod
    def validate_checkpoint_scoring_order(
        cls, param: Literal["min", "max"] | None, info: ValidationInfo
    ) -> Literal["min", "max"] | None:
        if info.data["checkpoint_scoring_metric"] is not None and param is None:
            raise ValueError(
                "'checkpoint_scoring_order' param is not filled while 'checkpoint_scoring_metric' param is filled,"
                + " please fill 'checkpoint_scoring_order' param"
            )
        elif param is not None and info.data["checkpoint_scoring_metric"] is None:
            raise ValueError(
                "'checkpoint_scoring_metric' param is not while 'checkpoint_scoring_order' param is filled, please"
                + " fill 'checkpoint_scoring_metric' param"
            )
        else:
            return param


class PredictTaskConfig(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")

    # Data parameters
    images_extension: Optional[str] = Field(default=".png")
    batch_size: int


class EvalTaskConfig(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")

    # Matching parameter
    matching_iou_threshold: float = Field(default=0.5)

    # Custom metrics
    custom_metrics: Optional[dict[str, str] | None] = Field(default=None)

    # Paths
    predictions_path: Optional[str | None] = Field(default=None)
    gts_path: Optional[str | None] = Field(default=None)

    # Disable tqdm
    disable_tqdm: Optional[bool | None] = Field(default=None)


class ConfidenceThresholdingTaskConfig(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")

    # Threshol parameter
    threshold: dict[int, float] | dict[str, float] | float

    # Paths
    predictions_path: Optional[str | None] = Field(default=None)
    gts_path: Optional[str | None] = Field(default=None)


class NMSConfig(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")

    # Threshol parameter
    nms_iou_threshold: float

    # Path
    predictions_path: Optional[str | None] = Field(default=None)

    # Disable tqdm
    disable_tqdm: Optional[bool | None] = Field(default=None)


TASK_CONFIG_DISPATCHER = {
    "train": TrainTaskConfig,
    "predict": PredictTaskConfig,
    "eval": EvalTaskConfig,
    "confidence_thresholding": ConfidenceThresholdingTaskConfig,
    "nms": NMSConfig,
}


class TaskSchema(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["train", "predict", "eval", "confidence_thresholding", "nms"]
    inputs: str
    config: dict[str, Any]

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, inputs: str) -> str:
        if ":" not in inputs:
            raise ValueError(f"Parameter inputs={inputs} of {cls.task_type} task should contain ':'.")
        if " " in inputs:
            raise ValueError(
                f"A space has bee detected for {cls.task_type} task. There should be no space in the inputs."
            )
        for input in inputs.split(","):
            if input.split(":")[0] not in INPUT_TYPES:
                raise ValueError(
                    f"Unknown type of input for {cls.task_type} task: {inputs}. It should be one of {INPUT_TYPES}."
                )

        return inputs

    @field_validator("config")
    @classmethod
    def parse_config(
        cls, config: dict[str, Any], info: ValidationInfo
    ) -> TrainTaskConfig | PredictTaskConfig | EvalTaskConfig:
        return TASK_CONFIG_DISPATCHER[info.data["task_type"]](**config)
