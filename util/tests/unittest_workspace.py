# TODO: figure out where to put the filesystem structure helpers. I think I want to put them inside gymlib and make a separate folder just testing the helpers.

import shutil
import unittest
from pathlib import Path

from util.tests.filesystem_unittest_util import (
    FilesystemStructure,
    create_structure,
    make_workspace_structure,
    verify_structure,
)
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

        # Reset this to avoid the error of it being created twice.
        # The assertion checking that it's only been been created once is meant to avoid creating multiple
        # objects for the same workspace. However, it's fine to create it multiple times in this case because
        # we're deleting the scratchspace before each test.
        DBGymWorkspace.num_times_created_this_run = 0

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
        return make_workspace_structure(symlinks_structure, task_runs_structure)

    def test_init_from_nonexistent_workspace(self) -> None:
        starting_structure = FilesystemStructure({})
        create_structure(self.scratchspace_path, starting_structure)

        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure = WorkspaceTests.get_workspace_init_structure(workspace)

        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_init_from_empty_workspace(self) -> None:
        starting_structure = FilesystemStructure({"dbgym_workspace": {}})
        create_structure(self.scratchspace_path, starting_structure)

        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure = WorkspaceTests.get_workspace_init_structure(workspace)

        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_init_from_already_initialized_workspace(self) -> None:
        # This first initialization will create a task run.
        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure = WorkspaceTests.get_workspace_init_structure(workspace)

        # The second initialization will create a second task run.
        # Make sure to reset this. In real usage, the second run would be a different Python process
        # so DBGymWorkspace.num_times_created_this_run would be 0.
        DBGymWorkspace.num_times_created_this_run = 0
        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure["dbgym_workspace"]["task_runs"][
            workspace.dbgym_this_run_path.name
        ] = {}
        ending_structure["dbgym_workspace"]["task_runs"]["latest_run.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{workspace.dbgym_this_run_path.name}",
        )

        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))


if __name__ == "__main__":
    unittest.main()
