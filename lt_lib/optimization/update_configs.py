# Author: Loïc Thiriet

from pathlib import Path
from typing import Any

from lt_lib.schemas.config_files_schemas import ModelConfig, RunConfig
from lt_lib.utils.dict_utils import extend_flattened_dict, flatten_dict
from lt_lib.utils.load_and_save import load_yaml_as_dict, save_dict_as_yaml


def update_one_nested_parameter_in_config_file(config_file_path: Path, nested_parameter: str, value: Any) -> None:
    """
    Updates one nested parameter inside a YAML configuration file.

    Args:
        config_file_path: The path to the YAML configuration file.
        nested_parameter: The nested parameter to be updated (e.g., 'section.subsection.parameter').
        value: The new value for the nested parameter.
    """
    config = load_yaml_as_dict(config_file_path)
    flattened_dict = flatten_dict(config)
    flattened_dict[nested_parameter] = value
    config = extend_flattened_dict(flattened_dict)
    save_dict_as_yaml(config, config_file_path)


def update_config_files(sampled_params: dict[str, Any], config_path: Path, model_config_path: Path | None):
    """
    Updates configuration and model configuration files with sampled parameters.

    Args:
        sampled_params: A dictionary containing sampled parameters.
        config_path: Path to the configuration file.
        model_config_path: Path to the model configuration file.

    Raises:
        ValueError: If the config_type of the sampled parameters is not 'RunConfig' or 'ModelConfig'.
    """
    for param, value in sampled_params.items():
        config_type, nested_parameter = param.split(".", maxsplit=1)
        if config_type == "RunConfig":
            _ = RunConfig.from_yaml(config_path)
            update_one_nested_parameter_in_config_file(config_path, nested_parameter, value)
        elif config_type == "ModelConfig":
            _ = ModelConfig.from_yaml(model_config_path)
            update_one_nested_parameter_in_config_file(model_config_path, nested_parameter, value)
        else:
            raise ValueError(f"The first characters of parameters to optimize should be 'RunConfig' or 'ModelConfig'.")
