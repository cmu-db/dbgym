import os
from pathlib import Path
import shutil
import subprocess

import yaml

from util.workspace import default_tables_path, workload_name_fn, default_workload_path


# Be careful when changing these constants. The integration test is hardcoded to work for these specific constants.
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01
INTEGTEST_DBGYM_CONFIG_FPATH = Path("scripts/integtest_dbgym_config.yaml")


def get_workspace_dpath(config_fpath: Path) -> Path:
    with open(config_fpath, 'r') as file:
        config = yaml.safe_load(file)
    return Path(config.get('dbgym_workspace_path'))


def clear_workspace(workspace_dpath: Path) -> None:
    if workspace_dpath.exists():
        assert not workspace_dpath.samefile("../dbgym_workspace"), "YOU MAY BE ABOUT TO DELETE YOUR ACTUAL WORKSPACE"
        shutil.rmtree(workspace_dpath)


if __name__ == "__main__":
    # Set the config file so that we use resources that don't conflict with normal usage (e.g. a different workspace, different ports, etc.).
    os.environ["DBGYM_CONFIG_PATH"] = str(INTEGTEST_DBGYM_CONFIG_FPATH)

    # Clear the integration testing workspace so we always run the test with a clean slate.
    workspace_dpath = get_workspace_dpath(INTEGTEST_DBGYM_CONFIG_FPATH)
    clear_workspace(workspace_dpath)

    # # Run the full Proto-X training pipeline, asserting things along the way
    # tables_dpath = default_tables_path(workspace_dpath, BENCHMARK, SCALE_FACTOR)
    # assert(not tables_dpath.exists())
    # subprocess.run(f"python task.py benchmark {BENCHMARK} data {SCALE_FACTOR}".split(), check=True)
    # assert(tables_dpath.exists())

    workload_name = workload_name_fn(SCALE_FACTOR, 15721, 15721, "all")
    workload_dpath = default_workload_path(workspace_dpath, BENCHMARK, workload_name)
    assert(not workload_dpath.exists())
    subprocess.run(f"python task.py benchmark {BENCHMARK} workload --scale-factor {SCALE_FACTOR}".split(), check=True)
    assert(workload_dpath.exists())

    # Clear it at the end as well to avoid leaving artifacts.
    # clear_workspace(workspace_dpath)