import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

from env.tuning_agent import DBMSConfigDelta, TuningAgent, TuningAgentMetadata
from util.workspace import DBGymConfig, fully_resolve_path

# These are the values used by set_up_env_integtests.sh.
# TODO: make set_up_env_integtests.sh take in these values directly as envvars.
INTEGTEST_BENCHMARK = "tpch"
INTEGTEST_SCALE_FACTOR = 0.01


class MockTuningAgent(TuningAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config_to_return: Optional[DBMSConfigDelta] = None

    @staticmethod
    def get_mock_fully_resolved_path() -> Path:
        return fully_resolve_path(
            IntegtestWorkspace.get_dbgym_cfg(), IntegtestWorkspace.get_workspace_path()
        )

    def _get_metadata(self) -> TuningAgentMetadata:
        # We just need these to be some fully resolved path, so I just picked the workspace path.
        return TuningAgentMetadata(
            workload_path=MockTuningAgent.get_mock_fully_resolved_path(),
            pristine_dbdata_snapshot_path=MockTuningAgent.get_mock_fully_resolved_path(),
            dbdata_parent_path=MockTuningAgent.get_mock_fully_resolved_path(),
            pgbin_path=MockTuningAgent.get_mock_fully_resolved_path(),
        )

    def _step(self) -> DBMSConfigDelta:
        assert self.config_to_return is not None
        ret = self.config_to_return
        # Setting this ensures you must set self.config_to_return every time.
        self.config_to_return = None
        return ret


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

        # Once we get here, we have an invariant that the workspace exists. We need this
        # invariant to be true in order to create the DBGymConfig.
        #
        # However, it also can't be created more than once so we need to check `is None`.
        if IntegtestWorkspace.INTEGTEST_DBGYM_CFG is None:
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
