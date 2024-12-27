# TODO: figure out where to put the filesystem structure helpers. I think I want to put them inside gymlib and make a separate folder just testing the helpers.

import shutil
import unittest
from pathlib import Path

from manage.tests.unittest_clean import CleanTests, FilesystemStructure
from util.workspace import DBGymWorkspace


class WorkspaceTests(unittest.TestCase):
    scratchspace_path: Path = Path()
    workspace_path: Path = Path()

    @classmethod
    def setUpClass(cls) -> None:
        cls.scratchspace_path = Path.cwd() / "util/tests/test_workspace_scratchspace/"
        cls.workspace_path = cls.scratchspace_path / "dbgym_workspace"

    def setUp(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    @staticmethod
    def get_workspace_init_structure(workspace: DBGymWorkspace) -> FilesystemStructure:
        symlinks_structure = FilesystemStructure({})
        task_runs_structure = FilesystemStructure(
            {
                "latest_run.link": (
                    "symlink",
                    f"dbgym_workspace/task_runs/{workspace.dbgym_this_run_path.name}",
                ),
                workspace.dbgym_this_run_path.name: {},
            }
        )
        return CleanTests.make_workspace_structure(
            symlinks_structure, task_runs_structure
        )

    def test_init_from_nonexistent_workspace(self) -> None:
        starting_structure = FilesystemStructure({})
        CleanTests.create_structure(self.scratchspace_path, starting_structure)

        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure = WorkspaceTests.get_workspace_init_structure(workspace)

        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )
