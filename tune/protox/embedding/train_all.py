import copy
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import tqdm
import yaml
from pytorch_metric_learning.utils import logging_presets
from ray.air import session
from ray.train import FailureConfig, RunConfig, SyncConfig
from ray.tune import TuneConfig, with_parameters, with_resources
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split
from torch.optim import Adam  # type: ignore[attr-defined]
from torch.utils.data import TensorDataset
from typing_extensions import ParamSpec

from tune.protox.embedding.loss import COST_COLUMNS, CostLoss, get_bias_fn
from tune.protox.embedding.train_args import (
    EmbeddingTrainAllArgs,
    EmbeddingTrainGenericArgs,
)
from tune.protox.embedding.trainer import StratifiedRandomSampler, VAETrainer
from tune.protox.embedding.utils import f_unpack_dict, parse_hyperopt_config
from tune.protox.embedding.vae import VAE, VAELoss, gen_vae_collate
from tune.protox.env.space.primitive_space import IndexSpace
from tune.protox.env.types import (
    TableAttrAccessSetsMap,
    TableAttrListMap,
    TableColTuple,
)
from tune.protox.env.workload import Workload
from util.log import DBGYM_LOGGER_NAME
from util.workspace import DBGymConfig, open_and_save, restart_ray, save_file


def fetch_vae_parameters_from_workload(w: Workload, ntables: int) -> tuple[int, int]:
    max_indexable = w.max_indexable()
    max_cat_features = max(
        ntables, max_indexable + 1
    )  # +1 for the "null" per attribute list.
    max_attrs = max_indexable + 1  # +1 to account for the table index.
    return max_attrs, max_cat_features


def fetch_index_parameters(
    dbgym_cfg: DBGymConfig,
    data: dict[str, Any],
    workload_path: Path,
) -> tuple[int, int, TableAttrListMap, dict[TableColTuple, int]]:
    tables = data["tables"]
    attributes = data["attributes"]
    query_spec = data["query_spec"]
    workload = Workload(
        dbgym_cfg, tables, attributes, query_spec, workload_path, pid=None
    )
    modified_attrs = workload.column_usages()

    space = IndexSpace(
        tables,
        max_num_columns=data["max_num_columns"],
        max_indexable_attributes=workload.max_indexable(),
        seed=0,
        rel_metadata=modified_attrs,
        attributes_overwrite=copy.deepcopy(modified_attrs),
        tbl_include_subsets=TableAttrAccessSetsMap({}),
        index_space_aux_type=False,
        index_space_aux_include=False,
        deterministic_policy=True,
    )

    max_attrs, max_cat_features = fetch_vae_parameters_from_workload(
        workload, len(tables)
    )
    return max_attrs, max_cat_features, modified_attrs, space.class_mapping


def load_input_data(
    dbgym_cfg: DBGymConfig,
    traindata_path: Path,
    train_size: float,
    max_attrs: int,
    require_cost: bool,
    seed: int,
) -> tuple[TensorDataset, Any, Any, Optional[TensorDataset], int]:
    # Load the input data.
    columns = []
    columns += ["tbl_index", "idx_class"]
    columns += [f"col{c}" for c in range(max_attrs - 1)]
    if require_cost:
        columns += COST_COLUMNS

    save_file(dbgym_cfg, traindata_path)
    df = pd.read_parquet(traindata_path, columns=columns)
    num_classes: int = df.idx_class.max() + 1

    # Get the y's and the x's.
    targets = (COST_COLUMNS + ["idx_class"]) if require_cost else ["idx_class"]
    y = df[targets].values
    df.drop(columns=COST_COLUMNS + ["idx_class"], inplace=True, errors="ignore")
    x = df.values
    del df
    gc.collect()
    gc.collect()

    if train_size == 1.0:
        train_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        del x
        gc.collect()
        gc.collect()
        return train_dataset, y, y[:, -1], None, num_classes

    # Perform the train test split.
    train_x, val_x, train_y, val_y = train_test_split(
        x,
        y,
        test_size=1.0 - train_size,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y[:, -1],
    )
    del x
    del y
    gc.collect()
    gc.collect()

    # Form the tensor datasets.
    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    del val_x
    del val_y
    del train_x
    gc.collect()
    gc.collect()
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        "Train Dataset Size: %s", len(train_dataset)
    )
    return train_dataset, train_y, train_y[:, -1], val_dataset, num_classes


def create_vae_model(
    config: dict[str, Any], max_attrs: int, max_cat_features: int
) -> VAE:
    cat_input = max_attrs * max_cat_features

    assert config["act"] in ["relu", "mish"]
    assert config["mean_output_act"] in ["tanh_squash", "sigmoid"]

    mean_output_act = {
        "sigmoid": nn.Sigmoid,
    }[config["mean_output_act"]]

    torch.set_float32_matmul_precision("high")
    model = VAE(
        max_categorical=max_cat_features,
        input_dim=cat_input,
        hidden_sizes=list(config["hidden_sizes"]),
        latent_dim=config["latent_dim"],
        act=nn.ReLU if config["act"] == "relu" else nn.Mish,
        bias_init=config["bias_init"],
        weight_init=config["weight_init"],
        weight_uniform=config["weight_uniform"],
        mean_output_act=mean_output_act,
        output_scale=config.get("output_scale", 1.0),
    )

    return model


def train_all_embeddings(
    dbgym_cfg: DBGymConfig,
    generic_args: EmbeddingTrainGenericArgs,
    train_all_args: EmbeddingTrainAllArgs,
) -> None:
    """
    Trains all num_samples models using different samples of the hyperparameter space, writing their
    results to different embedding_*/ folders in the run_*/ folder
    """
    start_time = time.time()

    with open_and_save(dbgym_cfg, train_all_args.hpo_space_path, "r") as f:
        json_dict = json.load(f)
        space = parse_hyperopt_config(json_dict["config"])

    # Connect to cluster or die.
    restart_ray(dbgym_cfg.root_yaml["ray_gcs_port"])
    ray.init(
        address=f"localhost:{dbgym_cfg.root_yaml['ray_gcs_port']}", log_to_driver=False
    )

    scheduler = FIFOScheduler()  # type: ignore
    # Search.
    search = HyperOptSearch(
        metric="loss",
        mode="min",
        points_to_evaluate=None,
        n_initial_points=20,
        space=space,
    )
    limiter = ConcurrencyLimiter(
        search, max_concurrent=train_all_args.train_max_concurrent
    )
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=limiter,
        num_samples=train_all_args.num_samples,
        max_concurrent_trials=train_all_args.train_max_concurrent,
        chdir_to_trial_dir=True,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoXEmbeddingHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
        storage_path=str(
            dbgym_cfg.cur_task_runs_path("embedding_ray_results", mkdir=True)
        ),
    )

    resources = {"cpu": 1}
    trainable = with_resources(
        with_parameters(
            _hpo_train,
            dbgym_cfg=dbgym_cfg,
            generic_args=generic_args,
            train_all_args=train_all_args,
        ),
        resources,
    )

    # Hopefully this is now serializable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"  # makes it so Ray doesn't change dir
    tuner = ray.tune.Tuner(
        trainable,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        "Best hyperparameters found were: ",
        results.get_best_result(metric="loss", mode="min").config,
    )
    if results.num_errors > 0:
        logging.getLogger(DBGYM_LOGGER_NAME).error("Encountered exceptions!")
        for i in range(len(results)):
            if results[i].error:
                logging.getLogger(DBGYM_LOGGER_NAME).error(f"Trial {results[i]} FAILED")
        assert False

    train_all_embeddings_duration = time.time() - start_time
    with open(f"{dbgym_cfg.dbgym_this_run_path}/hpo_train_time.txt", "w") as f:
        f.write(f"{train_all_embeddings_duration}")


def _hpo_train(
    config: dict[str, Any],
    dbgym_cfg: DBGymConfig,
    generic_args: EmbeddingTrainGenericArgs,
    train_all_args: EmbeddingTrainAllArgs,
) -> None:
    sys.path.append(os.fspath(dbgym_cfg.dbgym_repo_path))

    # Explicitly set the number of torch threads.
    os.environ["OMP_NUM_THREADS"] = str(train_all_args.train_max_concurrent)

    config = f_unpack_dict(config)
    if config.get("use_bias", False):
        if (
            "bias_separation" in config
            and "addtl_bias_separation" in config
            and "output_scale" in config
        ):
            # Do a hacky reconfigure.
            if (
                config["output_scale"]
                > config["bias_separation"] + config["addtl_bias_separation"]
            ):
                config["output_scale"] = (
                    config["bias_separation"] + config["addtl_bias_separation"]
                )
        config["metric_loss_md"]["output_scale"] = config["output_scale"]

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_dpath = (
        dbgym_cfg.cur_task_runs_data_path(mkdir=True)
        / f"embeddings_{dtime}_{os.getpid()}"
    )
    assert (
        not trial_dpath.exists()
    ), f"at this point, trial_dpath ({trial_dpath}) should not exist"

    # Seed
    seed = np.random.randint(int(1), int(1e8))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["seed"] = seed
    config["iterations_per_epoch"] = train_all_args.iterations_per_epoch

    logging.getLogger(DBGYM_LOGGER_NAME).info(config)

    # Build trainer and train.
    trainer, epoch_end = _build_trainer(
        dbgym_cfg,
        config,
        generic_args.traindata_path,
        trial_dpath,
        generic_args.benchmark_config_path,
        train_all_args.train_size,
        generic_args.workload_path,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    # Dump the config that we are executing.
    with open(f"{trial_dpath}/config", "w") as f:
        f.write(json.dumps(config, indent=4))

    trainer.train(num_epochs=config["num_epochs"])
    if trainer.failed:
        # Trainer has failed.
        with open(f"{trial_dpath}/FAILED", "w") as f:
            if trainer.fail_msg is not None:
                f.write(trainer.fail_msg)

        if trainer.fail_data is not None:
            torch.save(trainer.fail_data, f"{trial_dpath}/fail_data.pth")
        session.report({"loss": 1e8})
    else:
        res_dict = epoch_end(trainer=trainer, force=True, suppress=True)
        assert res_dict
        loss = res_dict["total_avg_loss"]
        session.report({"loss": loss})


def _build_trainer(
    dbgym_cfg: DBGymConfig,
    config: dict[str, Any],
    traindata_path: Path,
    trial_dpath: Path,
    benchmark_config_path: Path,
    train_size: float,
    workload_path: Path,
    dataloader_num_workers: int = 0,
    disable_tqdm: bool = False,
) -> tuple[VAETrainer, Callable[..., Optional[dict[str, Any]]]]:
    max_cat_features = 0
    max_attrs = 0

    # Load the benchmark configuration.
    with open_and_save(dbgym_cfg, benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        data = data[[k for k in data.keys()][0]]
        max_attrs, max_cat_features, _, class_mapping = fetch_index_parameters(
            dbgym_cfg, data, workload_path
        )

    config["class_mapping"] = {}
    for (tbl, col), key in class_mapping.items():
        config["class_mapping"][str(key)] = {
            "relname": tbl,
            "ord_column": col,
        }

    # Device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get the datasets.
    train_dataset, train_y, idx_class, val_dataset, num_classes = load_input_data(
        dbgym_cfg,
        traindata_path,
        train_size,
        max_attrs,
        config["metric_loss_md"].get("require_cost", False),
        config["seed"],
    )

    # Acquire the collation function.
    collate_fn = gen_vae_collate(max_cat_features)

    # Construct the models and optimizers.
    model = create_vae_model(config, max_attrs, max_cat_features)
    model.to(device=device)

    # Trunk is the identity.
    trunk = nn.Sequential(nn.Identity())
    trunk.to(device=device)

    models = {"trunk": trunk, "embedder": model}
    optimizers = {
        "embedder_optimizer": Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        ),
    }

    metric_loss = CostLoss(config["metric_loss_md"])

    # Define the loss functions.
    loss_funcs = {
        "metric_loss": metric_loss,
        "vae_loss": VAELoss(config["loss_fn"], max_attrs, max_cat_features),
    }

    loss_weights = {"metric_loss": config["metric_loss_weight"], "vae_loss": 1}

    # Define the sampler.
    sampler = StratifiedRandomSampler(
        idx_class,
        max_class=num_classes,
        batch_size=config["batch_size"],
        allow_repeats=True,
    )

    # Define the tester hook.
    record_keeper, _, _ = logging_presets.get_record_keeper(
        trial_dpath / "logs", trial_dpath / "tboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    model_folder = trial_dpath / "models"

    # Validation step loop.
    assert val_dataset
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=4096, collate_fn=collate_fn
    )
    epoch_end: Callable[..., Optional[dict[str, Any]]] = _construct_epoch_end(
        val_dl, config, hooks, model_folder
    )

    def clip_grad() -> None:
        if config["grad_clip_amount"] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip_amount"]
            )

    bias_fn = None
    if config["use_bias"]:
        bias_fn = get_bias_fn(config)

    # Build the trainer.
    return (
        VAETrainer(
            disable_tqdm=disable_tqdm,
            bias_fn=bias_fn,
            models=models,
            optimizers=optimizers,
            batch_size=config["batch_size"],
            loss_funcs=loss_funcs,
            mining_funcs={},
            dataset=train_dataset,
            sampler=sampler,
            iterations_per_epoch=(
                config["iterations_per_epoch"]
                if config["iterations_per_epoch"] is not None
                else int(len(train_dataset) / config["batch_size"])
            ),
            data_device=device,
            dtype=None,
            loss_weights=loss_weights,
            collate_fn=collate_fn,
            lr_schedulers=None,
            gradient_clippers={"embedder_grad_clipper": clip_grad},
            dataloader_num_workers=dataloader_num_workers,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=epoch_end,
        ),
        epoch_end,
    )


P = ParamSpec("P")


def _construct_epoch_end(
    val_dl: torch.utils.data.DataLoader[Any],
    config: dict[str, Any],
    hooks: Any,
    model_folder: Union[str, Path],
) -> Callable[P, Optional[dict[str, Any]]]:
    def epoch_end(*args: P.args, **kwargs: P.kwargs) -> Optional[dict[str, Any]]:
        trainer = kwargs.get("trainer", None)
        assert trainer
        assert isinstance(trainer, VAETrainer)

        save_interval = config.get("save_every", 1)
        if (trainer.epoch - 1) % save_interval == 0:
            # Save.
            mf = Path(model_folder) / f"epoch{trainer.epoch}"
            mf.mkdir(parents=True, exist_ok=True)
            hooks.save_models(trainer, str(mf), str(trainer.epoch))

        force = bool(kwargs.get("force", False))
        suppress = bool(kwargs.get("suppress", False))

        if force:
            total_metric_loss = []
            total_recon_loss = []
            with torch.no_grad():
                # Switch to eval mode.
                trainer.switch_eval()

                pbar = None if suppress else tqdm.tqdm(total=len(val_dl))
                for i, curr_batch in enumerate(val_dl):
                    # Get the losses.
                    trainer.calculate_loss(curr_batch)
                    if isinstance(trainer.losses["metric_loss"], torch.Tensor):
                        total_metric_loss.append(trainer.losses["metric_loss"].item())
                    else:
                        total_metric_loss.append(trainer.losses["metric_loss"])
                    total_recon_loss.append(trainer.last_recon_loss)

                    if pbar is not None:
                        pbar.set_description(
                            "total_recon=%.5f total_metric=%.5f"
                            % (total_recon_loss[-1], total_metric_loss[-1])
                        )
                        pbar.update(1)

                # Switch to train mode.
                trainer.switch_train()

        if force:
            return {
                "avg_metric": np.mean(total_metric_loss),
                "avg_recon": np.mean(total_recon_loss),
                "total_avg_loss": np.mean(total_metric_loss)
                + np.mean(total_recon_loss),
            }

        return None

    return epoch_end
