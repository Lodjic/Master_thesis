from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class RetinanetParameters(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")
    # To resolve a warning of the protected namespace 'model_'
    model_config["protected_namespaces"] = ()

    # Anchors parameters
    anchors_sizes: Optional[list[float] | tuple[float]] = Field(default=(16, 32, 64, 128, 256))
    anchors_scales: Optional[list[float] | tuple[float]] = Field(default=(1.0, 1.33, 1.66))
    anchors_ratios: Optional[list[float] | tuple[float]] = Field(default=(0.5, 1.0, 2.0))

    # Norm layer
    norm_layer: Optional[Literal["BatchNorm2d", "FrozenBatchNorm2d"]] = Field(default="FrozenBatchNorm2d")

    # Head-outputs process parameters for evaluation
    process_detections_during_training: Optional[bool] = Field(default=True, strict=True)
    score_thresh: Optional[float] = Field(default=0.01)
    nms_iou_thresh: Optional[float] = Field(default=0.7)
    topk_candidates: Optional[int] = Field(default=200)
    nb_max_detections_per_img: Optional[int] = Field(default=100)

    @field_validator("process_detections_during_training", mode="after")
    @classmethod
    def validate_bool_default_value_to_true(cls, param: bool) -> bool:
        if param is None:
            return True
        else:
            return param


MODEL_PARAMETERS_DISPATCHER = {
    "retinanet": RetinanetParameters,
}


class ModelSchema(BaseModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")
    # To resolve a warning of the protected namespace 'model_'
    model_config["protected_namespaces"] = ()

    model_type: Literal["retinanet"] = Field(default="retinanet")
    parameters: Optional[dict[str, Any] | None] = Field(default=None)

    @field_validator("parameters")
    @classmethod
    def parse_params(cls, parameters: dict[str, Any] | None, info: ValidationInfo) -> RetinanetParameters:
        if not parameters:
            return MODEL_PARAMETERS_DISPATCHER[info.data["model_type"]]
        else:
            return MODEL_PARAMETERS_DISPATCHER[info.data["model_type"]](**parameters)
