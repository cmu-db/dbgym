import subprocess
from pathlib import Path
from typing import Optional

import yaml

from util.workspace import DBGymConfig


class IntegtestWorkspace:
    """
    This is essentially a singleton class. This avoids multiple integtest_*.py files creating
    the workspace and/or the DBGymConfig redundantly.
    """

    ENV_INTEGTESTS_DBGYM_CONFIG_FPATH = Path("env/env_integtests_dbgym_config.yaml")
    INTEGTEST_DBGYM_CFG: Optional[DBGymConfig] = None

    @staticmethod
    def set_up_workspace() -> None:
        # This if statement prevents us from setting up the workspace twice, which saves time.
        if not IntegtestWorkspace.get_workspace_path().exists():
            subprocess.run(["./env/set_up_env_integtests.sh"], check=True)

        # The DBGymConfig needs to be created after running ./env/set_up_env_integtests.sh so
        # that it is created correctly.
        IntegtestWorkspace.INTEGTEST_DBGYM_CFG = DBGymConfig(
            IntegtestWorkspace.ENV_INTEGTESTS_DBGYM_CONFIG_FPATH
        )

    @staticmethod
    def get_dbgym_cfg() -> DBGymConfig:
        assert IntegtestWorkspace.INTEGTEST_DBGYM_CFG is not None
        return IntegtestWorkspace.INTEGTEST_DBGYM_CFG

    @staticmethod
    def get_workspace_path() -> Path:
        with open(IntegtestWorkspace.ENV_INTEGTESTS_DBGYM_CONFIG_FPATH) as f:
            return Path(yaml.safe_load(f)["dbgym_workspace_path"])
        assert False
