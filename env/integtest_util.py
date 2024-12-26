import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

from env.tuning_artifacts import DBMSConfigDelta, TuningAgent, TuningMetadata
from util.workspace import (
    DBGymConfig,
    fully_resolve_path,
    get_default_dbdata_parent_dpath,
    get_default_pgbin_path,
    get_default_pristine_dbdata_snapshot_path,
    get_default_workload_name_suffix,
    get_default_workload_path,
    get_workload_name,
)

# These are the values used by set_up_env_integtests.sh.
# TODO: make set_up_env_integtests.sh take in these values directly as envvars.
INTEGTEST_BENCHMARK = "tpch"
INTEGTEST_SCALE_FACTOR = 0.01


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

    @staticmethod
    def get_default_metadata() -> TuningMetadata:
        dbgym_cfg = IntegtestWorkspace.get_dbgym_cfg()
        workspace_path = fully_resolve_path(
            dbgym_cfg, IntegtestWorkspace.get_workspace_path()
        )
        return TuningMetadata(
            workload_path=fully_resolve_path(
                dbgym_cfg,
                get_default_workload_path(
                    workspace_path,
                    INTEGTEST_BENCHMARK,
                    get_workload_name(
                        INTEGTEST_SCALE_FACTOR,
                        get_default_workload_name_suffix(INTEGTEST_BENCHMARK),
                    ),
                ),
            ),
            pristine_dbdata_snapshot_path=fully_resolve_path(
                dbgym_cfg,
                get_default_pristine_dbdata_snapshot_path(
                    workspace_path, INTEGTEST_BENCHMARK, INTEGTEST_SCALE_FACTOR
                ),
            ),
            dbdata_parent_path=fully_resolve_path(
                dbgym_cfg, get_default_dbdata_parent_dpath(workspace_path)
            ),
            pgbin_path=fully_resolve_path(
                dbgym_cfg, get_default_pgbin_path(workspace_path)
            ),
        )


class MockTuningAgent(TuningAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.delta_to_return: Optional[DBMSConfigDelta] = None

    def _get_metadata(self) -> TuningMetadata:
        return IntegtestWorkspace.get_default_metadata()

    def _step(self) -> DBMSConfigDelta:
        assert self.delta_to_return is not None
        ret = self.delta_to_return
        # Setting this ensures you must set self.delta_to_return every time.
        self.delta_to_return = None
        return ret
