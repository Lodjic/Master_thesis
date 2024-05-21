import os
from pathlib import Path
from typing import Any

import wandb
from loguru import logger

from lt_lib.core.confidence_thresholding import confidence_thresholding
from lt_lib.core.eval import evaluation
from lt_lib.core.nms import apply_nms
from lt_lib.core.predict import predict
from lt_lib.core.train import train
from lt_lib.schemas.config_files_schemas import ModelConfig, RunConfig
from lt_lib.utils.log import initialize_logger, wandb_log_config_files
from lt_lib.utils.wandb_init import extract_wandb_init_config_dict

TASK_DISPATCHER = {
    "train": train,
    "predict": predict,
    "eval": evaluation,
    "confidence_thresholding": confidence_thresholding,
    "nms": apply_nms,
}


def find_tasks_order(tasks: dict[str, Any]):
    tasks_inputs = {task_params.inputs: task_name for task_name, task_params in tasks.items()}

    # Initialize the order task list
    tasks_order = []

    # Keyword for the inputs of the first task id 'root'
    keyword = "inputs_directory"
    # Search for the first task: the first task should have all its inputs to be in the inputs_directory
    found_input = [input for input in tasks_inputs.keys() if input.count(keyword) == len(input.split(","))]
    # If 0 or more than 1 task is found it is a configuration error
    if len(found_input) != 1:
        raise ValueError(
            f"None or more than 1 task have been found with keyword='inputs_directory'."
            + f"Please use only 'inputs_directory' keyword as inputs param for 1 task only."
        )
    else:
        tasks_order.append(tasks_inputs.pop(found_input[0]))
        keyword = tasks_order[-1]

    # While there are tasks available in the dictionary initialized keep searching for next task
    while tasks_inputs:
        found_input = [input for input in tasks_inputs.keys() if keyword in input]
        # If 0 or more than 1 task is found it is a configuration error
        if len(found_input) != 1:
            raise ValueError(
                f"None or more than 1 task have been found for keyword={keyword}: {found_input}."
                + f"Please use '{keyword}' keyword as inputs param for 1 task only."
            )
        else:
            tasks_order.append(tasks_inputs.pop(found_input[0]))
            keyword = tasks_order[-1]

    return tasks_order


class TaskOrchestrator:
    def __init__(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
        config_path: Path,
        model_config_path: Path | None,
        manually_init_wandb: bool = False,
    ) -> None:
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.config_path = config_path
        self.config = RunConfig.from_yaml(config_path)
        self.model_config_path = model_config_path
        self.model_config = ModelConfig.from_yaml(model_config_path) if model_config_path else None
        self.manually_init_wandb = manually_init_wandb

    def run(self):
        # Initiate wandb.run manually if specified in the cli_args passed to the Task Orchestrator
        if self.manually_init_wandb:
            if wandb.run:
                logger.error(
                    f"You intended to manually init wandb but it has already been initiated by module (maybe Ray)."
                    + "Impossible to initiate a second wandb run in the same session. Stopping the code run."
                )
                raise ValueError(
                    f"You intended to manually init wandb but it has already been initiated by module (maybe Ray)."
                    + "Impossible to initiate a second wandb run in the same session. Stopping the code run."
                )
            else:
                # Gets a dict with important config and model config parameters to log to wandb
                wandb_config_dict = extract_wandb_init_config_dict(self.config, self.model_config)
                # Initializes wandb
                wandb.init(config=wandb_config_dict, **self.config.wandb_init_params)
                # Logs config files to wandb
                wandb_log_config_files(self.config_path, self.model_config_path)

        # Processes tasks 1 by 1
        for task_name in find_tasks_order(self.config.tasks):
            # If outputs_dir does not exist, creates it
            if not os.path.isdir(self.outputs_dir / task_name):
                os.makedirs(self.outputs_dir / task_name)

            # Run the task
            TASK_DISPATCHER[self.config.tasks[task_name].task_type](
                inputs_dir=self.inputs_dir,
                outputs_dir=self.outputs_dir / task_name,
                configs={
                    "task_schema": self.config.tasks[task_name],
                    "model_config": self.model_config,
                    # "config_path": self.config_path,
                    # "model_config_path": self.model_config_path,
                    # "using_ray_tune": self.using_ray_tune,
                    # "using_wandb": self.using_wandb,
                },
            )

        # If wandb was manually instantiated then finish it as there are no more tasks to be run
        if self.manually_init_wandb:
            wandb.finish()
