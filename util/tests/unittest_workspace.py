# TODO: figure out where to put the filesystem structure helpers. I think I want to put them inside gymlib and make a separate folder just testing the helpers.

import os
import shutil
import unittest
from pathlib import Path
from typing import Optional

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

        self.workspace: Optional[DBGymWorkspace] = None
        self.expected_structure: Optional[FilesystemStructure] = None

    def tearDown(self) -> None:
        # You can comment this out if you want to inspect the scratchspace after a test (often used for debugging).
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    # All these helper functions will perform an action, update the expected structure, and then verify the structure.
    # Importantly though, I don't have helper functions for the complex functions that I want to test (e.g. link_result and save_file).
    def init_workspace_helper(self) -> None:
        # Reset this to avoid the error of it being created twice.
        # In real usage, the second run would be a different Python process so DBGymWorkspace._num_times_created_this_run would be 0.
        DBGymWorkspace._num_times_created_this_run = 0
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
            self.expected_structure["dbgym_workspace"]["task_runs"][
                "latest_run.link"
            ] = (
                "symlink",
                f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}",
            )

        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def make_file_helper(
        self, relative_path: str, file_obj: tuple[str, ...] = ("file",)
    ) -> Path:
        """
        You can override file_obj to make it a symlink instead.
        """
        assert self.workspace is not None and self.expected_structure is not None
        assert (
            ".." not in relative_path
        ), 'relative_path should not contain ".." (it should be inside the scratchspace dir)'
        file_path = self.scratchspace_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_obj[0] == "file":
            assert len(file_obj) in [1, 2]
            file_path.touch()
        elif file_obj[0] == "symlink":
            assert len(file_obj) == 2
            target_path = self.scratchspace_path / file_obj[1]
            os.symlink(target_path, file_path)
        else:
            assert False, f"Unsupported file_obj: {file_obj}"

        # Build up the nested dict structure for the expected path
        current_dict = self.expected_structure
        path_parts = relative_path.split("/")
        for part in path_parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[path_parts[-1]] = file_obj

        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )
        return file_path

    def make_result_helper(
        self, relative_path: str = "result.txt", file_obj: tuple[str, ...] = ("file",)
    ) -> Path:
        assert self.workspace is not None and self.expected_structure is not None
        assert (
            ".." not in relative_path
        ), 'relative_path should not contain ".." (it should be inside the run_*/ dir)'
        return self.make_file_helper(
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{relative_path}",
            file_obj=file_obj,
        )

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
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        self.workspace.link_result(result_path)
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            f"{result_path.name}.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_link_result_does_not_copy_directory_structure_to_symlinks_dir(
        self,
    ) -> None:
        """
        We always just want link_result to link to the base symlinks dir.
        """
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper(relative_path="dir1/dir2/dir3/result.txt")
        self.workspace.link_result(result_path)
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            f"{result_path.name}.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/dir1/dir2/dir3/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_link_result_invalid_custom_link_name(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        with self.assertRaisesRegex(
            AssertionError, 'link_name \\(custom\\) should end with "\\.link"'
        ):
            self.workspace.link_result(result_path, custom_link_name=f"custom")

    def test_link_result_valid_custom_link_name(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        self.workspace.link_result(result_path, custom_link_name="custom.link")
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            "custom.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_link_same_result_twice_with_same_link_name(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        self.workspace.link_result(result_path)
        self.workspace.link_result(result_path)
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            f"{result_path.name}.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_link_same_result_with_different_name(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        self.workspace.link_result(result_path)
        self.workspace.link_result(result_path, custom_link_name="custom.link")
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"] = {}
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            f"{result_path.name}.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.expected_structure["dbgym_workspace"]["symlinks"]["dbgym"][
            f"custom.link"
        ] = (
            "symlink",
            f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_link_result_from_another_run_raises_error(self) -> None:
        self.init_workspace_helper()
        result_path = self.make_result_helper()
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        with self.assertRaisesRegex(
            AssertionError,
            "The result must have been generated in \*this\* run\_\*/ dir",
        ):
            self.workspace.link_result(result_path)

    def test_link_result_from_external_dir_raises_error(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_file_helper("external/result.txt")
        with self.assertRaisesRegex(
            AssertionError,
            "The result must have been generated in \*this\* run\_\*/ dir",
        ):
            self.workspace.link_result(result_path)

    def test_link_result_cannot_link_symlink(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        symlink_path = self.make_result_helper(
            "symlink.link",
            file_obj=(
                "symlink",
                f"dbgym_workspace/task_runs/{self.workspace.dbgym_this_run_path.name}/{result_path.name}",
            ),
        )
        with self.assertRaisesRegex(
            AssertionError,
            "result_path \(.*\) should be a fully resolved path",
        ):
            self.workspace.link_result(symlink_path)

    def test_save_file_dependency(self) -> None:
        """
        See the comments in save_file() for what a "dependency" is.
        """
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        prev_run_name = self.workspace.dbgym_this_run_path.name
        result_path = self.make_result_helper()
        self.init_workspace_helper()
        self.workspace.save_file(result_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{result_path.name}.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_same_dependency_twice(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        prev_run_name = self.workspace.dbgym_this_run_path.name
        result_path = self.make_result_helper(file_obj=("file",))
        self.init_workspace_helper()
        self.workspace.save_file(result_path)
        self.workspace.save_file(result_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{result_path.name}.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_name}/{result_path.name}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_two_different_dependencies_with_same_filename_both_directly_inside_run(
        self,
    ) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        prev_run_names = []
        prev_run_names.append(self.workspace.dbgym_this_run_path.name)
        result1_path = self.make_result_helper(file_obj=("file",))
        self.init_workspace_helper()
        prev_run_names.append(self.workspace.dbgym_this_run_path.name)
        result2_path = self.make_result_helper(file_obj=("file",))
        filename = result1_path.name
        assert filename == result2_path.name

        self.init_workspace_helper()
        self.workspace.save_file(result1_path)
        self.workspace.save_file(result2_path)
        # The second save_file() should have overwritten the first one.
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{filename}.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_names[-1]}/{filename}",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_two_different_dependencies_with_same_filename_but_different_outermost_dirs(
        self,
    ) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        prev_run_name = self.workspace.dbgym_this_run_path.name
        result1_path = self.make_result_helper("dir1/result.txt", file_obj=("file",))
        result2_path = self.make_result_helper("result.txt", file_obj=("file",))
        filename = result1_path.name
        assert filename == result2_path.name

        self.init_workspace_helper()
        self.workspace.save_file(result1_path)
        self.workspace.save_file(result2_path)
        # The second save_file() should not overwrite the first one because the outermost dirs are different.
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{filename}.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_name}/{filename}",
        )
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ]["dir1.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_name}/dir1",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_config(self) -> None:
        """
        See the comments in save_file() for what a "config" is.
        """
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_file_helper(
            "external/result.txt", file_obj=("file", "contents")
        )
        self.workspace.save_file(result_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{result_path.name}"] = ("file", "contents")
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_same_config_twice(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_file_helper(
            "external/result.txt", file_obj=("file", "contents")
        )
        self.workspace.save_file(result_path)
        self.workspace.save_file(result_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{result_path.name}"] = ("file", "contents")
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_two_different_configs_with_same_filename(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result1_path = self.make_file_helper(
            "external/result.txt", file_obj=("file", "contents1")
        )
        result2_path = self.make_file_helper(
            "external/dir1/result.txt", file_obj=("file", "contents2")
        )
        filename = result1_path.name
        assert filename == result2_path.name

        self.workspace.save_file(result1_path)
        self.workspace.save_file(result2_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ][f"{filename}"] = ("file", "contents2")
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_dependency_inside_directory(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        prev_run_name = self.workspace.dbgym_this_run_path.name
        result_path = self.make_result_helper("dir1/dir2/result.txt")
        self.make_result_helper("dir1/other1.txt")
        self.make_result_helper("dir1/dir3/other2.txt")
        self.init_workspace_helper()
        self.workspace.save_file(result_path)
        self.expected_structure["dbgym_workspace"]["task_runs"][
            self.workspace.dbgym_this_run_path.name
        ]["dir1.link"] = (
            "symlink",
            f"dbgym_workspace/task_runs/{prev_run_name}/dir1",
        )
        self.assertTrue(
            verify_structure(self.scratchspace_path, self.expected_structure)
        )

    def test_save_file_generated_this_run_raises_error(self) -> None:
        self.init_workspace_helper()
        assert self.workspace is not None and self.expected_structure is not None
        result_path = self.make_result_helper()
        with self.assertRaisesRegex(
            AssertionError,
            "path \(.*\) was generated in this task run \(.*\)\. You do not need to save it",
        ):
            self.workspace.save_file(result_path)


if __name__ == "__main__":
    unittest.main()
