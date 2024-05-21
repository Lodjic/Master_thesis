import argparse
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ray import train as ray_train
from ray import tune

from lt_lib.optimization.objective import objective
from lt_lib.optimization.ray_run_config import (
    add_on_trial_callback_for_wandb_model_checkpointing,
)
from lt_lib.schemas.config_files_schemas import OptimizationConfig, RunConfig


def get_cli_args(parser: argparse.ArgumentParser):
    """
    Configures the command line arguments using the provided ArgumentParser.

    Args:
        parser: Instance of ArgumentParser to which the arguments are added.

    Returns:
        parser: ArgumentParser configured with the specified arguments.
    """
    parser.add_argument(
        "-id",
        "--inputs-directory",
        type=str,
        dest="inputs_directory",
        required=True,
        help="Path of the directory where the data is contained.",
    )
    parser.add_argument(
        "-od",
        "--outputs-directory",
        type=str,
        dest="outputs_directory",
        required=True,
        help="Path of the directory where the outputs are saved.",
    )
    parser.add_argument(
        "-c", "--config", type=str, dest="config_path", required=True, default="", help="Path of the config file."
    )
    parser.add_argument(
        "-oc",
        "--optimization_config",
        type=str,
        dest="optimization_config_path",
        required=True,
        default="",
        help="Path of the config file.",
    )
    parser.add_argument(
        "-mc",
        "--model_config",
        type=str,
        dest="model_config_path",
        default="",
        help="Path of the model config file.",
    )
    parser.add_argument(
        "-wb",
        "--use_wandb",
        type=bool,
        dest="use_wandb_ray_integration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use wandb ray integration.",
    )
    parser.add_argument(
        "-r",
        "--restore_dir",
        type=str,
        dest="restore_dir_path",
        default="",
        help="Specifies the directory from which to restore the experiment.",
    )
    parser.add_argument(
        "-log",
        "--log_lvl",
        type=str,
        dest="console_log_level",
        default="DEBUG",
        help="Specifies the level of the logs to display in the console/terminal.",
    )
    return parser


@dataclass
class OptimizationCliArgs:
    """Dataclass to be able to launch optimization in notebooks in a similar fashion to the CLI arguments"""

    inputs_directory: str
    outputs_directory: str
    config_path: str
    optimization_config_path: str
    model_config_path: Optional[str] = ""
    use_wandb_ray_integration: Optional[bool] = False
    restore_dir_path: Optional[str] = ""
    console_log_level: str = "DEBUG"


def optimization(args: OptimizationCliArgs) -> None:
    """
    Runs an optimization experiment using Ray-Tune.

    Args:
        args: A OptimizationCliArgs object containing all the arguments for the optimization.
    """
    # Gets objects from python optimization config
    optim_config_objects = OptimizationConfig().from_py_file(args.optimization_config_path)
    run_config = optim_config_objects["run_config"]

    # Sets the local 'ray_results' cache dir to be the dir specified in the run config for consistency
    # Note: if not set everything but the tuner.pkl is saved to run_config.local_dir and the tuner.pkl file is saved
    # in default local_dir='~/ray_results' which causes restore and results extraction issues.
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = run_config.local_dir

    # Adds the classical OnTrialCallback for logging model checkpoint to wandb if optimization uses wandb.
    if args.use_wandb_ray_integration:
        wandb_project = RunConfig.from_yaml(args.config_path).wandb_init_params["project"]
        run_config = add_on_trial_callback_for_wandb_model_checkpointing(
            run_config,
            optim_config_objects["on_trial_callback_type"],
            optim_config_objects["on_experiment_callback_type"],
            optim_config_objects["log_experiment_to_wandb"],
            wandb_project,
        )

    # Constructs Tuner_config thta will be deserialized when instantiating Tuner
    Tuner_config = {
        "param_space": optim_config_objects["param_space"],
        "tune_config": optim_config_objects["tune_config"],
        "run_config": run_config,
    }

    if optim_config_objects["resources"]:
        trainable_fct = tune.with_resources(objective, optim_config_objects["resources"])
    else:
        trainable_fct = objective

    # If no restore_dir_path is specified, then launches the experiment
    if args.restore_dir_path == "":
        tuner = tune.Tuner(
            tune.with_parameters(
                trainable_fct, use_wandb_ray_integration=args.use_wandb_ray_integration, cli_args=args
            ),
            **Tuner_config,
        )
        # Copies python optimization_config file to experiment directory
        shutil.copy2(
            args.optimization_config_path, Path(run_config.local_dir) / f"{run_config.name}/optimization_config.py"
        )
        # shutil.copy2(
        #     args.optimization_config_path,
        #     Path(f"{tuner._local_tuner._run_config.storage_path}/{tuner._local_tuner._run_config.name}")
        #     / "optimization_config.py",
        # )

    # If a restore_dir_path is specified, then restores the experiment
    else:
        tuner = tune.Tuner.restore(
            path=args.restore_dir_path,
            trainable=tune.with_parameters(
                trainable_fct, use_wandb_ray_integration=args.use_wandb_ray_integration, cli_args=args
            ),
            restart_errored=True,
            param_space=Tuner_config["param_space"],
        )

        # If a scheduler.pkl file is available in the experiment directory restore it
        scheduler_path = Path(args.restore_dir_path) / "scheduler.pkl"
        if scheduler_path.is_file():
            tuner._local_tuner._tune_config.scheduler.restore(scheduler_path)

        # Copies python optimization_config file to experiment directory restored
        nb_config_files = len(list(Path(args.restore_dir_path).glob("optimization_config*")))
        shutil.copy2(
            args.optimization_config_path, Path(args.restore_dir_path) / f"optimization_config-{nb_config_files}.py"
        )

    # Runs the optimization
    results = tuner.fit()


def main():
    """Main function that executes the optimization based on command-line arguments."""
    parser = argparse.ArgumentParser()
    parser = get_cli_args(parser)
    args = parser.parse_args()
    optimization(args)


# For the entrypoint CLI call
if __name__ == "__main__":
    main()
