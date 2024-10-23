import os
from pathlib import Path
import shutil
import subprocess

import yaml


# Be careful when changing these constants. The integration test is hardcoded to work for these specific constants.
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01
INTEGTEST_DBGYM_CONFIG_FPATH = Path("scripts/integtest_dbgym_config.yaml")


def get_workspace_path(config_fpath: Path) -> Path:
    with open(config_fpath, 'r') as file:
        config = yaml.safe_load(file)
    return Path(config.get('dbgym_workspace_path'))


if __name__ == "__main__":
    # Set the config file so that we use resources that don't conflict with normal usage (e.g. a different workspace, different ports, etc.).
    os.environ["DBGYM_CONFIG_PATH"] = str(INTEGTEST_DBGYM_CONFIG_FPATH)

    # Clear the integration testing workspace so we always run the test with a clean slate.
    workspace_path = get_workspace_path(INTEGTEST_DBGYM_CONFIG_FPATH)
    if workspace_path.exists():
        assert not workspace_path.samefile("../dbgym_workspace"), "YOU MAY BE ABOUT TO DELETE YOUR ACTUAL WORKSPACE"
        shutil.rmtree(workspace_path)

    # Run the full Proto-X training pipeline, asserting things along the way
    subprocess.run(f"python task.py benchmark {BENCHMARK} data {SCALE_FACTOR}".split())