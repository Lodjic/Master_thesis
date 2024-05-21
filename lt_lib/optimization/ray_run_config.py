# Author: Loïc Thiriet

import os
import zipfile
from pathlib import Path
from typing import Literal

import wandb
from ray import tune
from ray.train import RunConfig
from ray.tune.experiment.trial import Trial


class OnTrialCallback(tune.Callback):
    """
    A Ray-Tune callback for performing actions at different stages of a trial or experiment.

    Args:
        on_trial_callback: Specifies the stage of the trial to trigger the callback. Can be one of "on_trial_complete",
            "on_trial_save", "on_trial_result", or None.
        on_experiment_callback: Specifies the stage of the experiment to trigger the callback. Can be one of
            "on_experiment_end", or None.
        log_experiment_to_wandb: Whether to log experiment results to wandb.
        wandb_project: The name of the Wandb project to log results to.
    """

    def __init__(
        self,
        on_trial_callback: Literal["on_trial_complete", "on_trial_save", "on_trial_result", None],
        on_experiment_callback: Literal["on_experiment_end", None],
        log_experiment_to_wandb: bool,
        wandb_project: str,
    ) -> None:
        self.on_trial_callback = on_trial_callback
        self.on_experiment_callback = on_experiment_callback
        self.callback_func = self.on_trial_log_archive_to_wandb if log_experiment_to_wandb else self.zip_experiment_dir
        self.wandb_project = wandb_project
        super().__init__()

    def zip_experiment_dir(self, trial: Trial) -> Path:
        """
        Compresses all files of the ray experiment folder in a zip archive.

        Args:
            trial: The current trial object triggering this callback.

        Returns:
            experiment_archive_path: The path to the compressed experiment archive.
        """
        # If wandb_run was initiated and is running then logs ray_results archive to wandb
        # Get experiement dir
        experiment_dir = Path(trial.path).parent  # trial.remote_experiment_path also works
        experiment_archive_path = experiment_dir.with_suffix(".zip")

        # Compresses all files of the ray experiment folder in a zip archive
        with zipfile.ZipFile(experiment_archive_path, mode="w", compression=zipfile.ZIP_STORED, compresslevel=0) as zf:
            for file_path in experiment_dir.rglob("*"):
                zf.write(file_path, file_path.relative_to(experiment_dir))

        return experiment_archive_path

    def on_trial_log_archive_to_wandb(self, trial: Trial) -> None:
        """
        Archives the experiment directory associated with a trial and uploads it to Weights & Biases.

        Args:
            trial: The current trial object triggering this callback.
        """
        # Compresses all files of the ray experiment folder in a zip archive
        experiment_archive_path = self.zip_experiment_dir(trial)

        wandb.init(
            dir=experiment_archive_path.parent,
            project=self.wandb_project,
            name="Archive_uploader",
            id="Archive_uploader",
            resume=None,
        )

        archive_artifact = wandb.Artifact(name=f"ray_results-{Path(trial.path).parent.name}", type="ray_results")
        archive_artifact.add_file(experiment_archive_path)
        wandb.log_artifact(archive_artifact)
        wandb.finish()

        os.remove(experiment_archive_path)

    def on_trial_complete(self, iteration: int, trials: list[Trial], trial: Trial, **info) -> None:
        """
        Callback function triggered when a trial completes.

        Args:
            iteration: The current iteration number.
            trials: List of experiment trials.
            trial: The current trial.
            **info: Additional information passed.
        """
        if self.on_trial_callback == "on_trial_complete":
            self.callback_func(trial)
        else:
            pass

    def on_trial_save(self, iteration: int, trials: list[Trial], trial: Trial, **info):
        """
        Callback function triggered when a trial is saving.

        Args:
            iteration: The current iteration number.
            trials: List of experiment trials.
            trial: The current trial.
            **info: Additional information passed.
        """
        if self.on_trial_callback == "on_trial_save":
            self.callback_func(trial)
        else:
            pass

    def on_trial_result(self, iteration: int, trials: list[Trial], trial: Trial, **info):
        """
        Callback function triggered when a trial returns results.

        Args:
            iteration: The current iteration number.
            trials: List of experiment trials.
            trial: The current trial.
            **info: Additional information passed.
        """
        if self.on_trial_callback == "on_trial_result":
            self.callback_func(trial)
        else:
            pass

    def on_trial_start(self, iteration: int, trials: list[Trial], trial: Trial, **info):
        """
        Callback function triggered when a trial starts.

        Args:
            iteration: The current iteration number.
            trials: List of experiment trials.
            trial: The current trial.
            **info: Additional information passed.
        """
        if self.on_trial_callback == "on_trial_start":
            self.callback_func(trial)
        else:
            pass

    def on_experiment_end(self, trials: list[Trial], **info):
        """
        Callback function triggered when an experiment ends.

        Args:
            iteration: The current iteration number.
            trials: List of experiment trials.
            trial: The current trial.
            **info: Additional information passed.
        """
        if self.on_experiment_callback == "on_experiment_end":
            self.callback_func(trials[0])
        else:
            pass


def add_on_trial_callback_for_wandb_model_checkpointing(
    run_config: RunConfig,
    on_trial_callback: Literal["on_trial_complete", "on_trial_save", "on_trial_result", "on_trial_start", None] = None,
    on_experiment_callback: Literal["on_experiment_end", None] = None,
    log_experiment_to_wandb: bool | None = False,
    wandb_project: str | None = None,
):
    """
    Adds a callback for Weights & Biases model checkpointing to the provided RunConfig.

    Args:
        run_config: The RunConfig object to which the callback will be added.
        on_trial_callback: The type of trial callback to be triggered. Can be one of "on_trial_complete",
            "on_trial_save", "on_trial_result", "on_trial_start", or None. Defaults to None.
        on_experiment_callback: The type of experiment callback to be triggered. Can be one of "on_experiment_end", or
            None. Defaults to None.
        log_experiment_to_wandb: Whether to log the experiment to Weights & Biases. Defaults to False.
        wandb_project: The name of the Weights & Biases project to which the experiment is be logged. Defaults to None.

    Returns:
        run_config: The updated RunConfig object with the added callback.
    """
    if isinstance(run_config.callbacks, list):
        run_config.callbacks.append(
            OnTrialCallback(on_trial_callback, on_experiment_callback, log_experiment_to_wandb, wandb_project)
        )
    else:
        run_config.callbacks = [
            OnTrialCallback(on_trial_callback, on_experiment_callback, log_experiment_to_wandb, wandb_project)
        ]

    return run_config
