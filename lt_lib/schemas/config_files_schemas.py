# Author: Loïc Thiriet

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field

from lt_lib.schemas.base_schemas import BasePyConfig, BaseYamlConfig
from lt_lib.schemas.model_schemas import ModelSchema
from lt_lib.schemas.task_schemas import TaskSchema


class ModelConfig(BaseYamlConfig):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")
    # To resolve a warning of the protected namespace 'model_'
    model_config["protected_namespaces"] = ()

    # Config type
    config_type: Literal["ModelConfig"]

    # Core model parameters
    model: ModelSchema = Field(default=ModelSchema(model_type="retinanet"))
    model_loading_type: Optional[Literal["model", "model_state_dict"] | None] = Field(default=None)
    saved_checkpoint_path: Optional[str | None] = Field(default=None)
    weights_backbone: Optional[Literal["imagenet"] | None] = Field(default="imagenet")
    trainable_backbone_layers: Optional[Literal[0, 1, 2, 3, 4, 5]] = Field(default=5)


class RunConfig(BaseYamlConfig):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="forbid")
    # To resolve a warning of the protected namespace 'model_'
    model_config["protected_namespaces"] = ()

    # Config type
    config_type: Literal["RunConfig"]

    # Wandb init parameters
    wandb_init_params: Optional[dict[str, Any]] = Field(default={})

    # Tasks
    tasks: dict[str, TaskSchema]


class OptimizationConfig(BasePyConfig):
    def __init__(self) -> None:
        self.object_name = [
            "param_space",
            "tune_config",
            "run_config",
            "on_trial_callback_type",
            "on_experiment_callback_type",
            "log_experiment_to_wandb",
            "resources",
        ]
