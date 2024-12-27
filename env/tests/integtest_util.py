import os
import subprocess
from pathlib import Path
from typing import Optional

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
    get_workspace_path_from_config,
)


class GymlibIntegtestManager:
    """
    This is essentially a singleton class. This avoids multiple integtest_*.py files creating
    the workspace and/or the DBGymWorkspace object redundantly.

    The reason I put all these static methods in a class instead of directly in the module is
    that the functions have very generic names (e.g. set_up_workspace()) but having them
    inside a class makes it clear that they are related to the gymlib integration tests.
    """

    # These constants are also used by _set_up_gymlib_integtest_workspace.sh.
    BENCHMARK = "tpch"
    SCALE_FACTOR = 0.01
    DBGYM_CONFIG_PATH = Path("env/tests/gymlib_integtest_dbgym_config.yaml")

    # This is set at most once by set_up_workspace().
    DBGYM_WORKSPACE: Optional[DBGymWorkspace] = None

    @staticmethod
    def set_up_workspace() -> None:
        workspace_path = get_workspace_path_from_config(
            GymlibIntegtestManager.DBGYM_CONFIG_PATH
        )

        # This if statement prevents us from setting up the workspace twice, which saves time.
        if not workspace_path.exists():
            subprocess.run(
                ["./env/tests/_set_up_gymlib_integtest_workspace.sh"],
                env={
                    "BENCHMARK": GymlibIntegtestManager.BENCHMARK,
                    "SCALE_FACTOR": str(GymlibIntegtestManager.SCALE_FACTOR),
                    # By setting this envvar, we ensure that when running _set_up_gymlib_integtest_workspace.sh,
                    # make_standard_dbgym_workspace() will use the correct DBGYM_CONFIG_PATH.
                    "DBGYM_CONFIG_PATH": str(GymlibIntegtestManager.DBGYM_CONFIG_PATH),
                    **os.environ,
                },
                check=True,
            )

        # Once we get here, we have an invariant that the workspace exists. We need this
        # invariant to be true in order to create the DBGymWorkspace.
        #
        # However, it also can't be created more than once so we need to check `is None`.
        if GymlibIntegtestManager.DBGYM_WORKSPACE is None:
            GymlibIntegtestManager.DBGYM_WORKSPACE = DBGymWorkspace(workspace_path)

    @staticmethod
    def get_dbgym_workspace() -> DBGymWorkspace:
        assert GymlibIntegtestManager.DBGYM_WORKSPACE is not None
        return GymlibIntegtestManager.DBGYM_WORKSPACE

    @staticmethod
    def get_default_metadata() -> TuningMetadata:
        dbgym_workspace = GymlibIntegtestManager.get_dbgym_workspace()
        return TuningMetadata(
            workload_path=fully_resolve_path(
                get_default_workload_path(
                    dbgym_workspace.dbgym_workspace_path,
                    GymlibIntegtestManager.BENCHMARK,
                    get_workload_name(
                        GymlibIntegtestManager.SCALE_FACTOR,
                        get_default_workload_name_suffix(
                            GymlibIntegtestManager.BENCHMARK
                        ),
                    ),
                ),
            ),
            pristine_dbdata_snapshot_path=fully_resolve_path(
                get_default_pristine_dbdata_snapshot_path(
                    dbgym_workspace.dbgym_workspace_path,
                    GymlibIntegtestManager.BENCHMARK,
                    GymlibIntegtestManager.SCALE_FACTOR,
                ),
            ),
            dbdata_parent_path=fully_resolve_path(
                get_default_dbdata_parent_dpath(dbgym_workspace.dbgym_workspace_path),
            ),
            pgbin_path=fully_resolve_path(
                get_default_pgbin_path(dbgym_workspace.dbgym_workspace_path),
            ),
        )