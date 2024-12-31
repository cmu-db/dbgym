import os
import subprocess
from pathlib import Path
from typing import Optional

# TODO: remove infra_paths from the import
from gymlib.infra_paths import (
    get_dbdata_tgz_symlink_path,
    get_pgbin_symlink_path,
    get_workload_suffix,
    get_workload_symlink_path,
)
from gymlib.tuning_artifacts import TuningMetadata
from gymlib.workspace import (
    fully_resolve_path,
    get_tmp_path_from_workspace_path,
    get_workspace_path_from_config,
)

from benchmark.tpch.constants import DEFAULT_TPCH_SEED


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
    DBGYM_CONFIG_PATH = Path(
        "gymlib_package/gymlib/tests/gymlib_integtest_dbgym_config.yaml"
    )
    WORKSPACE_PATH: Optional[Path] = None

    @staticmethod
    def set_up_workspace() -> None:
        """
        Set up the workspace if it has not already been set up.
        None of the integtest_*.py files will delete the workspace so that future tests run faster.
        """
        GymlibIntegtestManager.WORKSPACE_PATH = get_workspace_path_from_config(
            GymlibIntegtestManager.DBGYM_CONFIG_PATH
        )

        # This if statement prevents us from setting up the workspace twice, which saves time.
        if not GymlibIntegtestManager.WORKSPACE_PATH.exists():
            subprocess.run(
                ["./gymlib_package/gymlib/tests/_set_up_gymlib_integtest_workspace.sh"],
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

    @staticmethod
    def get_workspace_path() -> Path:
        assert GymlibIntegtestManager.WORKSPACE_PATH is not None
        return GymlibIntegtestManager.WORKSPACE_PATH

    @staticmethod
    def get_default_metadata() -> TuningMetadata:
        assert GymlibIntegtestManager.BENCHMARK == "tpch"
        suffix = get_workload_suffix(
            GymlibIntegtestManager.BENCHMARK,
            seed_start=DEFAULT_TPCH_SEED,
            seed_end=DEFAULT_TPCH_SEED,
            query_subset="all",
        )
        return TuningMetadata(
            workload_path=fully_resolve_path(
                get_workload_symlink_path(
                    GymlibIntegtestManager.get_workspace_path(),
                    GymlibIntegtestManager.BENCHMARK,
                    GymlibIntegtestManager.SCALE_FACTOR,
                    suffix,
                ),
            ),
            pristine_dbdata_snapshot_path=fully_resolve_path(
                get_dbdata_tgz_symlink_path(
                    GymlibIntegtestManager.get_workspace_path(),
                    GymlibIntegtestManager.BENCHMARK,
                    GymlibIntegtestManager.SCALE_FACTOR,
                ),
            ),
            dbdata_parent_path=fully_resolve_path(
                get_tmp_path_from_workspace_path(
                    GymlibIntegtestManager.get_workspace_path()
                ),
            ),
            pgbin_path=fully_resolve_path(
                get_pgbin_symlink_path(GymlibIntegtestManager.get_workspace_path()),
            ),
        )
