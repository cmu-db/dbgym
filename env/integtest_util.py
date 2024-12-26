import subprocess
from pathlib import Path
from typing import Optional

import yaml

from env.tuning_artifacts import TuningMetadata
from util.workspace import (
    DBGymWorkspace,
    fully_resolve_path,
    get_default_dbdata_parent_dpath,
    get_default_pgbin_path,
    get_default_pristine_dbdata_snapshot_path,
    get_default_workload_name_suffix,
    get_default_workload_path,
    get_workload_name,
)

# These are the values used by set_up_gymlib_integtest_workspace.sh.
# TODO: make set_up_gymlib_integtest_workspace.sh take in these values directly as envvars.
INTEGTEST_BENCHMARK = "tpch"
INTEGTEST_SCALE_FACTOR = 0.01


class GymlibIntegtestWorkspaceManager:
    """
    This is essentially a singleton class. This avoids multiple integtest_*.py files creating
    the workspace and/or the DBGymWorkspace redundantly.

    The reason I put all these static methods in a class instead of directly in the module is
    that the functions have very generic names (e.g. set_up_workspace()) but having them
    inside a class makes it clear that they are related to the gymlib integration tests.
    """

    DBGYM_CONFIG_FPATH = Path("env/gymlib_integtest_dbgym_config.yaml")
    DBGYM_WORKSPACE: Optional[DBGymWorkspace] = None

    @staticmethod
    def set_up_workspace() -> None:
        # This if statement prevents us from setting up the workspace twice, which saves time.
        if not GymlibIntegtestWorkspaceManager.get_workspace_path().exists():
            subprocess.run(["./env/set_up_gymlib_integtest_workspace.sh"], check=True)

        # Once we get here, we have an invariant that the workspace exists. We need this
        # invariant to be true in order to create the DBGymWorkspace.
        #
        # However, it also can't be created more than once so we need to check `is None`.
        if GymlibIntegtestWorkspaceManager.DBGYM_WORKSPACE is None:
            GymlibIntegtestWorkspaceManager.DBGYM_WORKSPACE = DBGymWorkspace(
                GymlibIntegtestWorkspaceManager.DBGYM_CONFIG_FPATH
            )

    @staticmethod
    def get_dbgym_workspace() -> DBGymWorkspace:
        assert GymlibIntegtestWorkspaceManager.DBGYM_WORKSPACE is not None
        return GymlibIntegtestWorkspaceManager.DBGYM_WORKSPACE

    @staticmethod
    def get_workspace_path() -> Path:
        with open(
            GymlibIntegtestWorkspaceManager.DBGYM_CONFIG_FPATH
        ) as f:
            return Path(yaml.safe_load(f)["dbgym_workspace_path"])

    @staticmethod
    def get_default_metadata() -> TuningMetadata:
        dbgym_workspace = GymlibIntegtestWorkspaceManager.get_dbgym_workspace()
        workspace_path = fully_resolve_path(
            dbgym_workspace, GymlibIntegtestWorkspaceManager.get_workspace_path()
        )
        return TuningMetadata(
            workload_path=fully_resolve_path(
                dbgym_workspace,
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
                dbgym_workspace,
                get_default_pristine_dbdata_snapshot_path(
                    workspace_path, INTEGTEST_BENCHMARK, INTEGTEST_SCALE_FACTOR
                ),
            ),
            dbdata_parent_path=fully_resolve_path(
                dbgym_workspace, get_default_dbdata_parent_dpath(workspace_path)
            ),
            pgbin_path=fully_resolve_path(
                dbgym_workspace, get_default_pgbin_path(workspace_path)
            ),
        )
