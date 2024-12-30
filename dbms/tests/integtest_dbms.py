import shutil
import unittest
from pathlib import Path

from gymlib.symlinks_paths import get_repo_symlink_path

from dbms.postgres.cli import _postgres_build
from util.workspace import (
    DBGymWorkspace,
    fully_resolve_path,
    get_workspace_path_from_config,
)


class DBMSTests(unittest.TestCase):
    DBGYM_CONFIG_PATH = Path("dbms/tests/dbms_integtest_dbgym_config.yaml")

    def setUp(self) -> None:
        workspace_path = get_workspace_path_from_config(DBMSTests.DBGYM_CONFIG_PATH)
        # Get a clean start each time.
        if workspace_path.exists():
            shutil.rmtree(workspace_path)

        # Reset this to avoid the error of it being created twice.
        # In real usage, the second run would be a different Python process so DBGymWorkspace.num_times_created_this_run would be 0.
        DBGymWorkspace.num_times_created_this_run = 0
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


if __name__ == "__main__":
    unittest.main()
