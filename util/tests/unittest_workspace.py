# TODO: figure out where to put the filesystem structure helpers. I think I want to put them inside gymlib and make a separate folder just testing the helpers.

import shutil
from typing import Optional
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

        self.workspace = None
        self.expected_structure = None

    def tearDown(self) -> None:
        # You can comment this out if you want to inspect the scratchspace after a test (often used for debugging).
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    # All these helper functions will perform an action, update the expected structure, and then verify the structure.
    def init_workspace_helper(self) -> None:
        # Reset this to avoid the error of it being created twice.
        # In real usage, the second run would be a different Python process so DBGymWorkspace.num_times_created_this_run would be 0.
        DBGymWorkspace.num_times_created_this_run = 0
        self.workspace = DBGymWorkspace(self.workspace_path)

        if self.expected_structure is None:
            self.expected_structure = make_workspace_structure(
                FilesystemStructure({}),
                FilesystemStructure(
                    {
                        "latest_run.link": (
                            "symlink",
                            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}",
                        ),
                        self.workspace.dbgym_this_run_path.name: {},
                    }
                ),
            )
        else:
            self.expected_structure["dbgym_workspace"]["task_runs"][
                self.workspace.dbgym_this_run_path.name
            ] = {}
            self.expected_structure["dbgym_workspace"]["task_runs"]["latest_run.link"] = (
                "symlink",
                f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}",
            )

        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def make_result_helper(self, result_name: str="result.txt") -> Path:
        result_path = self.workspace.dbgym_this_run_path / result_name
        result_path.touch()
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][result_name] = ("file",)
        self.assertTrue(verify_structure(self.scratchspace_path, self.expected_structure))
        return result_path

    def link_result_helper(self, result_path: Path, custom_link_name: Optional[str]=None) -> None:
        self.workspace.link_result(result_path, custom_link_name=custom_link_name)
        link_name = f"{result_path.name}.link" if custom_link_name is None else custom_link_name
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            link_name
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.assertTrue(verify_structure(self.scratchspace_path, self.expected_structure))

    def test_init_fields(self) -> None:
        workspace = DBGymWorkspace(self.workspace_path)
        self.assertEqual(workspace.app_name, "dbgym")

    def test_init_from_nonexistent_workspace(self) -> None:
        self.init_workspace_helper()

    def test_init_from_empty_workspace(self) -> None:
        starting_structure = FilesystemStructure({"dbgym_workspace": {}})
        create_structure(self.scratchspace_path, starting_structure)
        self.init_workspace_helper()

    def test_init_from_already_initialized_workspace(self) -> None:
        self.init_workspace_helper()
        self.init_workspace_helper()

    def test_link_result_basic_functionality(self) -> None:
        self.init_workspace_helper()
        result_path = self.make_result_helper()
        self.link_result_helper(result_path)

    def test_link_result_invalid_custom_link_name(self) -> None:
        self.init_workspace_helper()
        result_path = self.make_result_helper()
        with self.assertRaises(AssertionError):
            self.link_result_helper(result_path, custom_link_name="custom_link")

    def test_link_result_valid_custom_link_name(self) -> None:
        self.init_workspace_helper()
        result_path = self.make_result_helper()
        self.link_result_helper(result_path, custom_link_name="custom_link.link")

    def test_link_same_result_twice(self) -> None:
        self.init_workspace_helper()
        result_path = self.make_result_helper()
        self.link_result_helper(result_path)
        self.link_result_helper(result_path)

    # TODO: test overriding existing symlink
    # TODO: test linking result from another run should raise
    # TODO: test that it should link in the agent dir in the links
    # TODO: test that it will ignore the directory structure (unlike save which keeps it)
    # TODO: test linking a symlink or a non-fully-resolved path


if __name__ == "__main__":
    unittest.main()
