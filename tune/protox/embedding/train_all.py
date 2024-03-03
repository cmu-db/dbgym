import time
import sys
import json
import os
import numpy as np
import random
import logging
import torch
import torch.nn as nn
from datetime import datetime
import yaml
from pytorch_metric_learning.utils import logging_presets
import tqdm
from pathlib import Path

import ray
from ray.train import FailureConfig, RunConfig, SyncConfig
from ray.tune import with_resources, with_parameters, TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import session

from tune.protox.embedding.loss import CostLoss, get_bias_fn
from tune.protox.embedding.utils import parse_hyperopt_config, fetch_index_parameters, load_input_data, f_unpack_dict
from tune.protox.embedding.vae import gen_vae_collate, VAELoss, create_vae_model
from tune.protox.embedding.trainer import VAETrainer, StratifiedRandomSampler

from misc.utils import open_and_save, restart_ray

def train_all_embeddings(ctx, generic_args, train_args):
    '''
    Trains all num_samples models using different samples of the hyperparameter space, writing their
    results to different embedding_*/ folders in the run_*/ folder
    '''
    start_time = time.time()

    with open_and_save(ctx, train_args.hpo_space_path, "r") as f:
        json_dict = json.load(f)
        space = parse_hyperopt_config(json_dict["config"])

    # Connect to cluster or die.
    restart_ray()
    ray.init(address="localhost:6379", log_to_driver=False)

    scheduler = FIFOScheduler()
    # Search.
    search = HyperOptSearch(
        metric="loss",
        mode="min",
        points_to_evaluate=None,
        n_initial_points=20,
        space=space,
    )
    search = ConcurrencyLimiter(search, max_concurrent=train_args.train_max_concurrent)
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=train_args.num_samples,
        max_concurrent_trials=train_args.train_max_concurrent,
        chdir_to_trial_dir=True,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoXEmbeddingHPO_{dtime}",
        storage_path=None,
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
    )

    resources = {"cpu": 1}
    trainable = with_resources(with_parameters(_hpo_train, ctx=ctx, generic_args=generic_args, train_args=train_args), resources)

    # Hopefully this is now serializable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0" # makes it so Ray doesn't change dir
    tuner = ray.tune.Tuner(
        trainable,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result(metric="loss", mode="min").config)
    if results.num_errors > 0:
        print("Encountered exceptions!")
        for i in range(len(results)):
            if results[i].error:
                print(f"Trial {results[i]} FAILED")
        assert False

    duration = time.time() - start_time
    with open(f"{ctx.obj.dbgym_this_run_path}/hpo_train_time.txt", "w") as f:
        f.write(f"{duration}")


def _hpo_train(config, ctx, generic_args, train_args):
    sys.path.append(os.fspath(ctx.obj.dbgym_repo_path))

    # Explicitly set the number of torch threads.
    os.environ["OMP_NUM_THREADS"] = str(train_args.train_max_concurrent)

    config = f_unpack_dict(config)
    if config.get("use_bias", False):
        if "bias_separation" in config and "addtl_bias_separation" in config and "output_scale" in config:
            # Do a hacky reconfigure.
            if config["output_scale"] > config["bias_separation"] + config["addtl_bias_separation"]:
                config["output_scale"] = config["bias_separation"] + config["addtl_bias_separation"]
        config["metric_loss_md"]["output_scale"] = config["output_scale"]

    output_dir = ctx.obj.dbgym_this_run_path

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_dir = output_dir / f"embeddings_{dtime}_{os.getpid()}"
    trial_dir.mkdir(parents=True, exist_ok=False)

    # Seed
    seed = np.random.randint(1, 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["seed"] = seed
    config["iterations_per_epoch"] = train_args.iterations_per_epoch

    logging.info(config)

    # Build trainer and train.
    trainer, epoch_end = _build_trainer(
        ctx,
        generic_args.benchmark,
        config,
        generic_args.dataset_path,
        trial_dir,
        generic_args.benchmark_config_path,
        train_args.train_size,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    # Dump the config that we are executing.
    with open(f"{trial_dir}/config", "w") as f:
        f.write(json.dumps(config, indent=4))

    trainer.train(num_epochs=config["num_epochs"])
    if trainer.failed:
        # Trainer has failed.
        with open(f"{trial_dir}/FAILED", "w") as f:
            if trainer.fail_msg is not None:
                f.write(trainer.fail_msg)

        if trainer.fail_data is not None:
            torch.save(trainer.fail_data, f"{trial_dir}/fail_data.pth")
        session.report({"loss": 1e8})
    else:
        loss = epoch_end(trainer, force=True, suppress=True)["total_avg_loss"]
        session.report({"loss": loss})


def _build_trainer(ctx, benchmark, config, input_path, trial_dir, benchmark_config_path, train_size, dataloader_num_workers=0, disable_tqdm=False):
    max_cat_features = 0
    max_attrs = 0

    # Load the benchmark configuration.
    with open_and_save(ctx, benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        max_attrs, max_cat_features, _, class_mapping = fetch_index_parameters(ctx, benchmark, data)

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
        ctx,
        input_path,
        train_size,
        max_attrs,
        config["metric_loss_md"].get("require_cost", False),
        config["seed"])

    # Acquire the collation function.
    collate_fn = gen_vae_collate(max_cat_features)

    # Construct the models and optimizers.
    model = create_vae_model(config, max_attrs, max_cat_features)
    model.to(device=device)

    # Trunk is the identity.
    trunk = nn.Sequential(nn.Identity())
    trunk.to(device=device)

    models = {"trunk": trunk, "embedder": model}
    optimizers = { "embedder_optimizer": torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]), }

    metric_loss = CostLoss(config["metric_loss_md"])
    # Default miner.
    tminers = {}

    # Define the loss functions.
    loss_funcs = {
        "metric_loss": metric_loss,
        "vae_loss": VAELoss(config["loss_fn"], max_attrs, max_cat_features),
    }

    loss_weights = {"metric_loss": config["metric_loss_weight"], "vae_loss": 1}

    # Define the sampler.
    sampler = StratifiedRandomSampler(idx_class, max_class=num_classes, batch_size=config["batch_size"], allow_repeats=True)

    # Define the tester hook.
    record_keeper, _, _ = logging_presets.get_record_keeper(f"{trial_dir}/logs", f"{trial_dir}/tboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    model_folder = f"{trial_dir}/models"

    # Validation step loop.
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=4096, collate_fn=collate_fn)
    epoch_end = _construct_epoch_end(val_dl, config, hooks, model_folder)

    def clip_grad():
        if config["grad_clip_amount"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_amount"])

    bias_fn = None
    if config["use_bias"]:
        bias_fn = get_bias_fn(config)

    # Build the trainer.
    return VAETrainer(
        disable_tqdm=disable_tqdm,
        bias_fn=bias_fn,
        models=models,
        optimizers=optimizers,
        batch_size=config["batch_size"],
        loss_funcs=loss_funcs,
        mining_funcs=tminers,
        dataset=train_dataset,
        sampler=sampler,
        iterations_per_epoch=config["iterations_per_epoch"] if config["iterations_per_epoch"] is not None else int(len(train_dataset) / config["batch_size"]),
        data_device=device,
        dtype=None,
        loss_weights=loss_weights,
        collate_fn=collate_fn,
        lr_schedulers=None,
        gradient_clippers={"embedder_grad_clipper": clip_grad},
        dataloader_num_workers=dataloader_num_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=epoch_end,
    ), epoch_end


def _construct_epoch_end(val_dl, config, hooks, model_folder):
    def epoch_end(trainer, *args, **kwargs):
        save_interval = config.get("save_every", 1)
        if (trainer.epoch - 1) % save_interval == 0:
            # Save.
            mf = Path(model_folder) / f"epoch{trainer.epoch}"
            mf.mkdir(parents=True, exist_ok=True)
            hooks.save_models(trainer, str(mf), str(trainer.epoch))

        force = kwargs.get("force", False)
        suppress = kwargs.get("suppress", False)

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
                        pbar.set_description("total_recon=%.5f total_metric=%.5f" % (total_recon_loss[-1], total_metric_loss[-1]))
                        pbar.update(1)

                # Switch to train mode.
                trainer.switch_train()

        if force:
            return {
                "avg_metric": np.mean(total_metric_loss),
                "avg_recon": np.mean(total_recon_loss),
                "total_avg_loss": np.mean(total_metric_loss) + np.mean(total_recon_loss),
            }

    return epoch_end