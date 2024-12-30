import os
import shutil
import unittest
from pathlib import Path

from gymlib.symlinks_paths import get_dbdata_tgz_symlink_path, get_repo_symlink_path
from gymlib.workspace import (
    DBGymWorkspace,
    fully_resolve_path,
    get_workspace_path_from_config,
)

from benchmark.tpch.cli import _tpch_tables
from dbms.postgres.cli import _postgres_build, _postgres_dbdata


class DBMSTests(unittest.TestCase):
    DBGYM_CONFIG_PATH = Path("dbms/tests/dbms_integtest_dbgym_config.yaml")

    def setUp(self) -> None:
        workspace_path = get_workspace_path_from_config(DBMSTests.DBGYM_CONFIG_PATH)
        # Get a clean start each time.
        if workspace_path.exists():
            shutil.rmtree(workspace_path)

        # Reset this to avoid the error of it being created twice.
        # In real usage, the second run would be a different Python process so DBGymWorkspace._num_times_created_this_run would be 0.
        DBGymWorkspace._num_times_created_this_run = 0
        self.workspace = DBGymWorkspace(workspace_path)

    def tearDown(self) -> None:
        if self.workspace.dbgym_workspace_path.exists():
            shutil.rmtree(self.workspace.dbgym_workspace_path)

    def test_postgres_build(self) -> None:
        repo_path = get_repo_symlink_path(self.workspace.dbgym_workspace_path)
        self.assertFalse(repo_path.exists())
        _postgres_build(self.workspace, False)
        self.assertTrue(repo_path.exists())
        self.assertTrue(fully_resolve_path(repo_path).exists())

    def test_postgres_dbdata(self) -> None:
        # Setup
        # Make sure to recreate self.workspace so that each function call counts as its own run.
        scale_factor = 0.01
        _postgres_build(self.workspace, False)
        DBGymWorkspace._num_times_created_this_run = 0
        self.workspace = DBGymWorkspace(self.workspace.dbgym_workspace_path)
        _tpch_tables(self.workspace, scale_factor)
        DBGymWorkspace._num_times_created_this_run = 0
        self.workspace = DBGymWorkspace(self.workspace.dbgym_workspace_path)

        # Test
        dbdata_tgz_path = get_dbdata_tgz_symlink_path(
            self.workspace.dbgym_workspace_path, "tpch", scale_factor
        )
        self.assertFalse(dbdata_tgz_path.exists())
        intended_dbdata_hardware = os.environ.get("INTENDED_DBDATA_HARDWARE", "hdd")
        _postgres_dbdata(
            self.workspace, "tpch", scale_factor, None, intended_dbdata_hardware, None
        )
        self.assertTrue(dbdata_tgz_path.exists())
        self.assertTrue(fully_resolve_path(dbdata_tgz_path).exists())


if __name__ == "__main__":
    unittest.main()
