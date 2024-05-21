import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lt_lib.orchestration.task_orchestrator import TaskOrchestrator
from lt_lib.utils.log import initialize_logger


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
        dest="use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use wandb or not in this run.",
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
class RunCliArgs:
    """Dataclass to be able to launch training in notebooks in a similar fashion to the CLI arguments"""

    inputs_directory: str
    outputs_directory: str
    config_path: str
    model_config_path: Optional[str] = ""
    use_wandb: Optional[bool] = False
    console_log_level: str = "DEBUG"


def run(args):
    """
    Runs the desired tasks found specified to the TaskOrchestrator.

    Args:
        args: A RunCliArgs object containing all the arguments for the task(s) run.
    """
    # Intitalizes the logger
    initialize_logger(Path(args.outputs_directory), args.console_log_level)

    # Instantiates the TaskOrchestrator
    task_orchestrator = TaskOrchestrator(
        inputs_dir=Path(args.inputs_directory),
        outputs_dir=Path(args.outputs_directory),
        config_path=Path(args.config_path),
        model_config_path=Path(args.model_config_path) if args.model_config_path != "" else None,
        manually_init_wandb=args.use_wandb,
    )

    # Runs the tasks
    task_orchestrator.run()


def main():
    """Main function that executes the run based on command-line arguments."""
    parser = argparse.ArgumentParser()
    parser = get_cli_args(parser)
    args = parser.parse_args()
    run(args)


# For the entrypoint CLI call
if __name__ == "__main__":
    main()
