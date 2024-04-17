import json
from pathlib import Path
import shutil
import time
import click
import pandas as pd

from misc.utils import WORKSPACE_PATH_PLACEHOLDER, DBGymConfig, conv_inputpath_to_realabspath, link_result, open_and_save, default_hpoed_agent_params_path, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, workload_name_fn, default_tuning_steps_dname
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
def tune(dbgym_cfg: DBGymConfig, benchmark_name: str, seed_start: int, seed_end: int, query_subset: str, scale_factor: float, hpoed_agent_params_path: Path, enable_boot_during_tune: bool) -> None:
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

    # Add configs to the hpoed_params that are allowed to differ between HPO and tuning.
    # In general, for configs that can differ between HPO and tuning, I chose to append
    #   "_during_hpo"/"_during_tune" to the end of them instead of naming them the same
    #   and overriding the config during tuning. It's just much less confusing if we
    #   make sure to never override any configs in hpoed_params.
    hpoed_params["enable_boot_during_tune"] = enable_boot_during_tune

    # Piggyback off the HPO magic.
    t = TuneTrial(dbgym_cfg, False)
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

    # Link the tuning steps data (this directory allows you to replay the tuning run).
    # Replaying requires the params.json file, so we also copy it here.
    # Since params.json is fairly small, I choose to copy the file itself instead of just
    #   making a symlink to it.
    tuning_steps_dpath = dbgym_cfg.cur_task_runs_artifacts_path("tuning_steps")
    shutil.copy(hpoed_agent_params_path, tuning_steps_dpath)
    tuning_steps_link_dname = default_tuning_steps_dname(benchmark_name, workload_name, False)
    link_result(dbgym_cfg, tuning_steps_dpath, custom_result_name=tuning_steps_link_dname)
