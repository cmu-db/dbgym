from pathlib import Path

import yaml

from util.workspace import DBGymConfig

ENV_INTEGTESTS_DBGYM_CONFIG_FPATH = Path("env/env_integtests_dbgym_config.yaml")
INTEGTEST_DBGYM_CFG = DBGymConfig(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH)


def get_integtest_workspace_path() -> Path:
    with open(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH) as f:
        return Path(yaml.safe_load(f)["dbgym_workspace_path"])
    assert False
