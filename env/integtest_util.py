from pathlib import Path

import yaml


ENV_INTEGTESTS_DBGYM_CONFIG_FPATH = Path("env/env_integtests_dbgym_config.yaml")


def get_integtest_workspace_path() -> Path:
    with open(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH) as f:
        return Path(yaml.safe_load(f)["dbgym_workspace_path"])
    assert False