# Author: Loïc Thiriet

import wandb

from lt_lib.schemas.config_files_schemas import ModelConfig, RunConfig
from lt_lib.utils.dict_utils import extend_flattened_dict, flatten_dict
from lt_lib.utils.regex_matcher import get_elements_with_regex

REGEX_CONFIG_PARAMS_TO_DROP = "^(?!.*images_extension.*|.*n_epochs.*|.*wandb.*|.*save.*|.*saving.*|.*checkpoint.*).*$"


def extract_info_to_log_from_config(config: RunConfig):
    wandb_config_dict = {}

    for task in config.tasks.keys():
        # Gets task_type
        task_type = config.tasks[task].task_type
        # Gets config dict of the task
        config_dict = flatten_dict(config.tasks[task].config.model_dump())
        # Removes uninteresting keys
        keys_to_keep = get_elements_with_regex(REGEX_CONFIG_PARAMS_TO_DROP, config_dict.keys(), unique=False)
        config_dict = {k: v for k, v in config_dict.items() if k in keys_to_keep}
        # Adds config_dict to wandb_config_dict
        wandb_config_dict[task_type] = extend_flattened_dict(config_dict)

    return wandb_config_dict


def extract_info_to_log_from_model_config(model_config: ModelConfig):
    # Gets root parameters
    model_params = model_config.model_dump(exclude={"config_type", "model"})
    # Removes parameters that include model
    keys_to_drop = get_elements_with_regex(f"^.*model.*$", model_params.keys(), unique=False)
    model_params = {k: v for k, v in model_params.items() if k not in keys_to_drop}
    # Adds nested model parameters
    model_params.update(model_config.model.parameters.model_dump(exclude={"process_detections_during_training"}))
    # Returns model_config_dict to log to wandb
    return {"model": model_config.model.model_type, "model_parameters": model_params}


def extract_wandb_init_config_dict(config: RunConfig, model_config: ModelConfig | None):
    # Instantiate the wandb_config_dict
    wandb_config_dict = {}

    # Fills the wandb_config_dict
    if model_config:
        wandb_config_dict = extract_info_to_log_from_model_config(model_config)
    wandb_config_dict.update(extract_info_to_log_from_config(config))

    return wandb_config_dict
