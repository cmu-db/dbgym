import json
import os
from pathlib import Path
import time
import click
import pandas as pd

from misc.utils import DEFAULT_BOOT_CONFIG_FPATH, WORKSPACE_PATH_PLACEHOLDER, DBGymConfig, conv_inputpath_to_realabspath, open_and_save, default_hpoed_agent_params_path, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, workload_name_fn
from tune.protox.agent.coerce_config import coerce_config
from tune.protox.agent.hpo import TuneTrial, build_space


# This is used when you already have a good set of HPOs and just want to tune the DBMS
@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.option("--seed-start", type=int, default=15721, help="A workload consists of queries from multiple seeds. This is the starting seed (inclusive).")
@click.option("--seed-end", type=int, default=15721, help="A workload consists of queries from multiple seeds. This is the ending seed (inclusive).")
@click.option(
    "--query-subset",
    type=click.Choice(["all", "even", "odd"]),
    default="all",
)
@click.option(
    "--scale-factor",
    default=1.0,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--hpoed-agent-params-path",
    default=None,
    type=Path,
    help=f"The path to best params found by the agent HPO process. The default is {default_hpoed_agent_params_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}",
)
@click.option(
    "--enable-boot-during-tune",
    is_flag=True,
    help="Whether to enable the Boot query accelerator during the tuning process. Deciding to use Boot during tuning is separate from deciding to use Boot during HPO.",
)
@click.option(
    "--boot-config-fpath",
    default=DEFAULT_BOOT_CONFIG_FPATH,
    type=Path,
    help="The path to the file configuring Boot.",
)
def tune(dbgym_cfg: DBGymConfig, benchmark_name: str, seed_start: int, seed_end: int, query_subset: str, scale_factor: float, hpoed_agent_params_path: Path, enable_boot_during_tune: bool, boot_config_fpath: Path) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    workload_name = workload_name_fn(scale_factor, seed_start, seed_end, query_subset)
    if hpoed_agent_params_path == None:
        hpoed_agent_params_path = default_hpoed_agent_params_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name)

    # Convert all input paths to absolute paths
    hpoed_agent_params_path = conv_inputpath_to_realabspath(dbgym_cfg, hpoed_agent_params_path)

    # Tune
    with open_and_save(dbgym_cfg, hpoed_agent_params_path, "r") as f:
        hpoed_params = json.load(f)

    # Coerce using a dummy space.
    hpoed_params = coerce_config(dbgym_cfg, build_space(
        sysknobs={},
        benchmark_config={},
        workload_path=Path(),
        embedder_path=[],
        pgconn_info={}
    ), hpoed_params)

    # Assume we are executing from the root.
    hpoed_params["dbgym_dir"] = dbgym_cfg.dbgym_repo_path

    # Get the duration.
    assert "duration" in hpoed_params

    # Piggyback off the HPO magic.
    t = TuneTrial(dbgym_cfg)
    # This is a hack.
    t.logdir = Path(dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)) # type: ignore
    t.logdir.mkdir(parents=True, exist_ok=True) # type: ignore
    t.setup(hpoed_params)
    start = time.time()

    data = []
    step_data_fpath = dbgym_cfg.cur_task_runs_data_path(mkdir=True) / "step_data.csv"
    while (time.time() - start) < hpoed_params["duration"] * 3600:
        data.append(t.step())

        # Continuously write the file out.
        pd.DataFrame(data).to_csv(step_data_fpath, index=False)

    t.cleanup()
    # Output the step data.
    pd.DataFrame(data).to_csv(step_data_fpath, index=False)