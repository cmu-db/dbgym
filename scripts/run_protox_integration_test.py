import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from util.workspace import (
    default_embedder_path,
    default_hpoed_agent_params_path,
    default_pristine_dbdata_snapshot_path,
    default_replay_data_fpath,
    default_repo_path,
    default_tables_path,
    default_traindata_path,
    default_tuning_steps_dpath,
    default_workload_path,
    workload_name_fn,
)

# Be careful when changing these constants. The integration test is hardcoded to work for these specific constants.
DBMS = "postgres"
AGENT = "protox"
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01
INTEGTEST_DBGYM_CONFIG_FPATH = Path("scripts/integtest_dbgym_config.yaml")


def get_workspace_dpath(config_fpath: Path) -> Path:
    with open(config_fpath, "r") as file:
        config = yaml.safe_load(file)
    return Path(config.get("dbgym_workspace_path"))


def clear_workspace(workspace_dpath: Path) -> None:
    actual_workspace_dpath = Path("../dbgym_workspace")
    if workspace_dpath.exists():
        if actual_workspace_dpath.exists():
            assert not workspace_dpath.samefile(
                "../dbgym_workspace"
            ), "YOU MAY BE ABOUT TO DELETE YOUR ACTUAL WORKSPACE"
        shutil.rmtree(workspace_dpath)


if __name__ == "__main__":
    intended_dbdata_hardware = sys.argv[1] if len(sys.argv) > 1 else "hdd"

    # Set the config file so that we use resources that don't conflict with normal usage (e.g. a different workspace, different ports, etc.).
    os.environ["DBGYM_CONFIG_PATH"] = str(INTEGTEST_DBGYM_CONFIG_FPATH)

    # Clear the integration testing workspace so we always run the test with a clean slate.
    workspace_dpath = get_workspace_dpath(INTEGTEST_DBGYM_CONFIG_FPATH)
    clear_workspace(workspace_dpath)

    # Run the full Proto-X training pipeline, asserting things along the way
    tables_dpath = default_tables_path(workspace_dpath, BENCHMARK, SCALE_FACTOR)
    assert not tables_dpath.exists()
    subprocess.run(
        f"python task.py benchmark {BENCHMARK} data {SCALE_FACTOR}".split(), check=True
    )
    assert tables_dpath.exists()

    workload_name = workload_name_fn(SCALE_FACTOR, 15721, 15721, "all")
    workload_dpath = default_workload_path(workspace_dpath, BENCHMARK, workload_name)
    assert not workload_dpath.exists()
    subprocess.run(
        f"python task.py benchmark {BENCHMARK} workload --scale-factor {SCALE_FACTOR}".split(),
        check=True,
    )
    assert workload_dpath.exists()

    repo_dpath = default_repo_path(workspace_dpath)
    assert not repo_dpath.exists()
    subprocess.run(f"python task.py dbms {DBMS} build".split(), check=True)
    assert repo_dpath.exists()

    pristine_dbdata_snapshot_fpath = default_pristine_dbdata_snapshot_path(
        workspace_dpath, BENCHMARK, SCALE_FACTOR
    )
    assert not pristine_dbdata_snapshot_fpath.exists()
    subprocess.run(
        f"python task.py dbms {DBMS} dbdata {BENCHMARK} --scale-factor {SCALE_FACTOR} --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
        check=True,
    )
    assert pristine_dbdata_snapshot_fpath.exists()

    traindata_dpath = default_traindata_path(workspace_dpath, BENCHMARK, workload_name)
    assert not traindata_dpath.exists()
    subprocess.run(
        f"python task.py tune {AGENT} embedding datagen {BENCHMARK} --scale-factor {SCALE_FACTOR} --override-sample-limits lineitem,32768 --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
        check=True,
    )
    assert traindata_dpath.exists()

    embedder_dpath = default_embedder_path(workspace_dpath, BENCHMARK, workload_name)
    assert not embedder_dpath.exists()
    subprocess.run(
        f"python task.py tune {AGENT} embedding train {BENCHMARK} --scale-factor {SCALE_FACTOR} --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2".split(),
        check=True,
    )
    assert embedder_dpath.exists()

    hpoed_agent_params_fpath = default_hpoed_agent_params_path(
        workspace_dpath, BENCHMARK, workload_name
    )
    assert not hpoed_agent_params_fpath.exists()
    subprocess.run(
        f"python task.py tune {AGENT} agent hpo {BENCHMARK} --scale-factor {SCALE_FACTOR} --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01 --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
        check=True,
    )
    assert hpoed_agent_params_fpath.exists()

    tuning_steps_dpath = default_tuning_steps_dpath(
        workspace_dpath, BENCHMARK, workload_name, False
    )
    assert not tuning_steps_dpath.exists()
    subprocess.run(
        f"python task.py tune {AGENT} agent tune {BENCHMARK} --scale-factor {SCALE_FACTOR}".split(),
        check=True,
    )
    assert tuning_steps_dpath.exists()

    replay_data_fpath = default_replay_data_fpath(
        workspace_dpath, BENCHMARK, workload_name, False
    )
    assert not replay_data_fpath.exists()
    subprocess.run(
        f"python3 task.py tune {AGENT} agent replay {BENCHMARK} --scale-factor {SCALE_FACTOR}".split(),
        check=True,
    )
    assert replay_data_fpath.exists()

    # Clear it at the end as well to avoid leaving artifacts.
    clear_workspace(workspace_dpath)
