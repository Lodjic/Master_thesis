from datetime import datetime

import numpy as np
import optuna
from loguru import logger
from ray import train, tune
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch


def generate_trial_name(trial: tune.experiment.trial.Trial) -> str:
    """Generates the trial_name from the trial object.

    Args:
        trial: ray trial object

    Returns:
        Returns the name of the trial as a str.
    """
    return f"trial_{datetime.now().isoformat(sep='_', timespec='seconds').replace(':', '-')}_{trial.trial_id}"


########## Tuner kwargs ##########

### Parameters search space of the optimization task. The name of the parameters should follow parameter nomclature
### of the config file.
# dict[str, Any]
param_space = {
    "ModelConfig.model.parameters.nms_iou_thresh": tune.quniform(0.1, 0.9, q=0.02),
    "RunConfig.tasks.train_task.config.optimizer_params.lr": tune.qloguniform(0.000001, 0.0001, q=0.000001),
    "RunConfig.tasks.confidence_thresholding_task.config.threshold": tune.quniform(0.1, 0.9, q=0.02),
    "RunConfig.tasks.nms_task.config.nms_iou_threshold": tune.grid_search(list(map(float, np.linspace(0.1, 0.9, 10)))),
    "RunConfig.tasks.nms_task.config.nms_iou_threshold": tune.grid_search(
        list(map(float, np.arange(10, 90, 10) / 100))
    ),
}

random_seed = 0

### Ray TuneConfig object parametrizing the tuning configuration. It comprises search algorithm definition, scheduler
### definition, as well as sampling parameters and trial name generation.
# tune.TuneConfig
tune_config = tune.TuneConfig(
    search_alg=OptunaSearch(
        sampler=optuna.samplers.TPESampler(
            consider_prior=True,
            prior_weight=1,
            consider_magic_clip=True,
            n_startup_trials=10,
            n_ei_candidates=24,
            seed=random_seed,
            multivariate=False,
            group=False,
            warn_independent_sampling=True,
            constant_liar=False,
            constraints_func=None,
            categorical_distance_func=None,
        ),
        metric=["val.level1.recall", "val.level1.precision"],
        mode=["max", "max"],
    ),
    scheduler=None,
    num_samples=50,
    max_concurrent_trials=1,
    trial_name_creator=generate_trial_name,
    trial_dirname_creator=generate_trial_name,
)

### Ray RunConfig object parametrizing the experiment configuration. It comprises experiment name, checkpoint
### configuration, log_to_file param and verbose param.
# train.RunConfig
run_config = train.RunConfig(
    name=f"experiment-seed{random_seed}_{datetime.now().isoformat(sep='_', timespec='seconds').replace(':', '-')}",
    local_dir="/content/ray_results",
    checkpoint_config=train.CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val.level2.mAP",
        checkpoint_score_order="max",
    ),
    verbose=0,
    log_to_file=True,
)

### String and boolean expliciting which on_trial function to over-implement to log model checkpoint (Artifact) to wandb
# str or bool
on_trial_callback_type = "on_trial_start"
on_experiment_callback_type = "on_experiment_end"
log_experiment_to_wandb = True

### Ray ScalingConfig object to parametrize the ressources management
# dict[str, int | float] | ray.train.ScalingConfig
resources = {"cpu": 1}
