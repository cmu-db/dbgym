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
from util.workspace import DBGymWorkspace, link_result, save_file


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
        # You can comment this out if you want to inspect the scratchspace after a test (often used for debugging).
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

    @staticmethod
    def get_updated_structure_from_workspace_init(
        structure: FilesystemStructure, workspace: DBGymWorkspace
    ) -> FilesystemStructure:
        structure["dbgym_workspace"]["task_runs"][
            workspace.dbgym_this_run_path.name
        ] = {}
        structure["dbgym_workspace"]["task_runs"]["latest_run.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{workspace.dbgym_this_run_path.name}",
        )
        return structure

    def test_init_fields(self) -> None:
        workspace = DBGymWorkspace(self.workspace_path)
        self.assertEqual(workspace.app_name, "dbgym")

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
        ending_structure = WorkspaceTests.get_updated_structure_from_workspace_init(
            ending_structure, workspace
        )

        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_result_basic_functionality(self) -> None:
        # Setup.
        workspace = DBGymWorkspace(self.workspace_path)
        ending_structure = WorkspaceTests.get_workspace_init_structure(workspace)

        # Make a result file.
        result_path = workspace.dbgym_this_run_path / "result.txt"
        result_path.touch()
        ending_structure["dbgym_workspace"]["task_runs"][
            workspace.dbgym_this_run_path.name
        ]["result.txt"] = ("file",)

        # Link the result file.
        workspace.link_result(result_path)
        ending_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        ending_structure["dbgym_workspace"]["symlinks"]["dbgym"]["result.txt.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{workspace.dbgym_this_run_path.name}/result.txt",
        )

        # Verify structure.
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    # TODO: test overriding existing symlink
    # TODO: test linking result from another run should raise
    # TODO: test that it should link in the agent dir in the links
    # TODO: test that it will ignore the directory structure (unlike save which keeps it)
    # TODO: test linking a symlink or a non-fully-resolved path


if __name__ == "__main__":
    unittest.main()
