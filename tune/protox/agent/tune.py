import json
import os
from pathlib import Path
import time
import click
import pandas as pd

from misc.utils import WORKSPACE_PATH_PLACEHOLDER, DBGymConfig, conv_inputpath_to_realabspath, open_and_save, default_hpoed_agent_config_path, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER
from tune.protox.agent.coerce_config import coerce_config
from tune.protox.agent.hpo import TuneTrial, build_space


# This is used when you already have a good set of HPOs and just want to tune the DBMS
@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.argument("workload-name")
@click.option(
    "--scale-factor",
    default=1.0,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--hpoed-agent-config-path",
    default=None,
    type=Path,
    help=f"The path to best config found by the agent HPO process. The default is {default_hpoed_agent_config_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}",
)
def tune(dbgym_cfg: DBGymConfig, benchmark_name: str, workload_name: str, scale_factor: float, hpoed_agent_config_path: Path) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if hpoed_agent_config_path == None:
        hpoed_agent_config_path = default_hpoed_agent_config_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, scale_factor)

    # Convert all input paths to absolute paths
    hpoed_agent_config_path = conv_inputpath_to_realabspath(dbgym_cfg, hpoed_agent_config_path)

    # Tune
    with open_and_save(dbgym_cfg, hpoed_agent_config_path, "r") as f:
        hpo_config = json.load(f)

    # Coerce using a dummy space.
    hpo_config = coerce_config(dbgym_cfg, build_space(
        sysknobs={},
        benchmark_config={},
        pristine_pgdata_snapshot_path=Path(),
        workload_path=Path(),
        embedding_path=[],
        pgconn_info={}
    ), hpo_config)

    # Assume we are executing from the root.
    # TODO(phw2): get this from dbgym_cfg
    hpo_config["dbgym_dir"] = os.getcwd()

    # Get the duration.
    assert "duration" in hpo_config

    # Piggyback off the HPO magic.
    t = TuneTrial()
    # This is a hack.
    t.logdir = Path("artifacts/") # type: ignore
    t.logdir.mkdir(parents=True, exist_ok=True) # type: ignore
    t.setup(hpo_config)
    start = time.time()

    data = []
    step_data_fpath = dbgym_cfg.cur_task_runs_data_path(mkdir=True) / "step_data.csv"
    while (time.time() - start) < hpo_config["duration"] * 3600:
        data.append(t.step())

        # Continuously write the file out.
        pd.DataFrame(data).to_csv(step_data_fpath, index=False)

    t.cleanup()
    # Output the step data.
    pd.DataFrame(data).to_csv(step_data_fpath, index=False)