import copy
import logging
import os
import shutil
import unittest
from pathlib import Path
from typing import Any, NewType, cast

from manage.cli import MockDBGymWorkspace, clean_workspace
from util.workspace import path_exists_dont_follow_symlinks

# This is here instead of on `if __name__ == "__main__"` because we often run individual tests, which
#   does not go through the `if __name__ == "__main__"` codepath.
# Make it DEBUG to see logs from verify_structure(). Make it CRITICAL to not see any logs.
# We use the root logger for unit tests to keep it separate from the standard logging subsystem which
#   uses the dbgym.* loggers.
logging.basicConfig(level=logging.CRITICAL)


FilesystemStructure = NewType("FilesystemStructure", dict[str, Any])


class CleanTests(unittest.TestCase):
    scratchspace_path: Path = Path()

    @staticmethod
    def create_structure(root_path: Path, structure: FilesystemStructure) -> None:
        def create_structure_internal(
            root_path: Path, cur_path: Path, structure: FilesystemStructure
        ) -> None:
            for path, content in structure.items():
                full_path: Path = cur_path / path

                if isinstance(content, dict):  # Directory
                    full_path.mkdir(parents=True, exist_ok=True)
                    create_structure_internal(
                        root_path,
                        full_path,
                        FilesystemStructure(cast(dict[str, Any], content)),
                    )
                elif isinstance(content, tuple) and content[0] == "file":
                    assert len(content) == 1
                    full_path.touch()
                elif isinstance(content, tuple) and content[0] == "symlink":
                    assert len(content) == 2
                    target_path = root_path / content[1]
                    os.symlink(target_path, full_path)
                else:
                    raise ValueError(f"Unsupported type for path ({path}): {content}")

        root_path.mkdir(parents=True, exist_ok=True)
        create_structure_internal(root_path, root_path, structure)

    @staticmethod
    def verify_structure(root_path: Path, structure: FilesystemStructure) -> bool:
        def verify_structure_internal(
            root_path: Path, cur_path: Path, structure: FilesystemStructure
        ) -> bool:
            # Check for the presence of each item specified in the structure
            for name, item in structure.items():
                new_cur_path = cur_path / name
                if not path_exists_dont_follow_symlinks(new_cur_path):
                    logging.debug(f"{new_cur_path} does not exist")
                    return False
                elif isinstance(item, dict):
                    if not new_cur_path.is_dir():
                        logging.debug(f"expected {new_cur_path} to be a directory")
                        return False
                    if not verify_structure_internal(
                        root_path,
                        new_cur_path,
                        FilesystemStructure(cast(dict[str, Any], item)),
                    ):
                        return False
                elif isinstance(item, tuple) and item[0] == "file":
                    if not new_cur_path.is_file():
                        logging.debug(f"expected {new_cur_path} to be a regular file")
                        return False
                elif isinstance(item, tuple) and item[0] == "symlink":
                    if not new_cur_path.is_symlink():
                        logging.debug(f"expected {new_cur_path} to be a symlink")
                        return False
                    # If item[1] is None, this indicates that we expect the symlink to be broken
                    if item[1] != None:
                        expected_target = root_path / item[1]
                        if not new_cur_path.resolve().samefile(expected_target):
                            logging.debug(
                                f"expected {new_cur_path} to link to {expected_target}, but it links to {new_cur_path.resolve()}"
                            )
                            return False
                else:
                    assert False, "structure misconfigured"

            # Check for any extra files or directories not described by the structure
            expected_names = set(structure.keys())
            actual_names = {entry.name for entry in cur_path.iterdir()}
            if not expected_names.issuperset(actual_names):
                logging.debug(
                    f"expected_names={expected_names}, actual_names={actual_names}"
                )
                return False

            return True

        if not root_path.exists():
            logging.debug(f"{root_path} does not exist")
            return False
        return verify_structure_internal(root_path, root_path, structure)

    @staticmethod
    def make_workspace_structure(
        symlinks_structure: FilesystemStructure,
        task_runs_structure: FilesystemStructure,
    ) -> FilesystemStructure:
        """
        This function exists so that it's easier to refactor the tests in case we ever change
          how the workspace is organized.
        """
        return FilesystemStructure(
            {
                "symlinks": symlinks_structure,
                "task_runs": task_runs_structure,
            }
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.scratchspace_path = Path.cwd() / "manage/tests/test_clean_scratchspace/"

    def setUp(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def test_structure_helpers(self) -> None:
        structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",), "dir2": {"file2.txt": ("file",)}},
                "dir3": {"nested_link_to_dir1": ("symlink", "dir1")},
                "link_to_dir1": ("symlink", "dir1"),
                "link_to_file2": ("symlink", "dir1/dir2/file2.txt"),
            }
        )
        CleanTests.create_structure(self.scratchspace_path, structure)
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, structure))

        extra_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, extra_dir_structure)
        )
        extra_dir_structure["dir4"] = {}
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, extra_dir_structure)
        )

        missing_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, missing_dir_structure)
        )
        del missing_dir_structure["dir1"]
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, missing_dir_structure)
        )

        extra_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, extra_file_structure)
        )
        extra_file_structure["file3.txt"] = ("file",)
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, extra_file_structure)
        )

        missing_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, missing_file_structure)
        )
        del missing_file_structure["dir1"]["file1.txt"]
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, missing_file_structure)
        )

        extra_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, extra_link_structure)
        )
        extra_link_structure["link_to_dir3"] = ("symlink", "dir3")
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, extra_link_structure)
        )

        missing_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, missing_link_structure)
        )
        del missing_link_structure["link_to_dir1"]
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, missing_link_structure)
        )

        wrong_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, wrong_link_structure)
        )
        wrong_link_structure["link_to_dir1"] = ("symlink", "dir3")
        self.assertFalse(
            CleanTests.verify_structure(self.scratchspace_path, wrong_link_structure)
        )

    def test_nonexistent_workspace(self) -> None:
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))

    def test_no_symlinks_dir_and_no_task_runs_dir(self) -> None:
        starting_structure = FilesystemStructure({})
        ending_structure = FilesystemStructure({})
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_no_symlinks_dir_and_yes_task_runs_dir(self) -> None:
        starting_structure = FilesystemStructure(
            {"task_runs": {"file1.txt": ("file",)}}
        )
        ending_structure = FilesystemStructure({"task_runs": {}})
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_yes_symlinks_dir_and_no_task_runs_dir(self) -> None:
        starting_structure = FilesystemStructure({"symlinks": {}})
        ending_structure = FilesystemStructure({"symlinks": {}})
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_no_symlinks_in_dir_and_no_task_runs_in_dir(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({})
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_no_links_in_symlinks(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({"run_0": {}})
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_file_directly_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {"file1.txt": ("file",), "file2.txt": ("file",)}
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure({"file1.txt": ("file",)})
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_dir_directly_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",)},
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",)}}
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_file_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",)},
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",)}}
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_dir_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/dir2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"dir2": {"file1.txt": ("file",)}, "file2.txt": ("file",)},
                "dir3": {"file3.txt": ("file",)},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/dir2")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"dir2": {"file1.txt": ("file",)}, "file2.txt": ("file",)},
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path))
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_link_crashes(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "symlink2": ("symlink", "task_runs/file1.txt"),
                "file1.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        with self.assertRaises(AssertionError):
            clean_workspace(MockDBGymWorkspace(self.scratchspace_path))

    def test_safe_mode_link_to_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/file1.txt")},
                "file1.txt": ("file",),
                "file2.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/file1.txt")},
                "file1.txt": ("file",),
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_safe_mode_link_to_file_in_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/file2.txt"),
                },
                "file2.txt": ("file",),
                "file3.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/file2.txt"),
                },
                "file2.txt": ("file",),
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_safe_mode_link_to_dir_with_link_to_file_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/dir2/file2.txt")},
                "dir2": {
                    "file2.txt": ("file",),
                },
                "file3.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/dir2/file2.txt")},
                "dir2": {
                    "file2.txt": ("file",),
                },
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_aggressive_mode_link_to_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/file1.txt")},
                "file1.txt": ("file",),
                "file2.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", None)},
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="aggressive")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_link_to_link_to_file_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "task_runs/file2.txt")},
                "file2.txt": ("file",),
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)

        # We disallow links to links so it's an AssertionError
        with self.assertRaises(AssertionError):
            clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")

    def test_multi_link_loop_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "symlinks/symlink1")},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)

        # pathlib disallows multi-link loops so it's a RuntimeError
        with self.assertRaises(RuntimeError):
            clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")

    def test_link_self_loop_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "symlinks/symlink1")}
        )
        starting_task_runs_structure = FilesystemStructure({})
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)

        # pathlib disallows link self-loops so it's a RuntimeError
        with self.assertRaises(RuntimeError):
            clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")

    def test_dont_loop_infinitely_if_there_are_cycles_between_different_dirs_in_runs(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir2/file2.txt"),
                },
                "dir2": {
                    "file2.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir2/file2.txt"),
                },
                "dir2": {
                    "file2.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_dont_loop_infinitely_if_there_is_a_dir_in_runs_that_links_to_a_file_in_itself(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_dont_loop_infinitely_if_there_is_loop_amongst_symlinks(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_broken_symlink_has_no_effect(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "task_runs/dir1/non_existent_file.txt"),
                },
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",), "symlink2": ("symlink", None)}}
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    # The idea behind this test is that we shouldn't be following links outside of task_runs, even on safe mode
    def test_link_to_folder_outside_runs_that_contains_link_to_other_run_doesnt_save_other_run(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "external/dir3/file3.txt"),
                },
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        starting_structure["external"] = FilesystemStructure(
            {
                "dir3": {
                    "file3.txt": ("file",),
                    "symlink3": ("symlink", "task_runs/dir2/file2.txt"),
                }
            }
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "external/dir3/file3.txt"),
                }
            }
        )
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )
        ending_structure["external"] = {
            "dir3": {"file3.txt": ("file",), "symlink3": ("symlink", None)}
        }

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )

    def test_outside_task_runs_doesnt_get_deleted(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({"dir1": {}})
        starting_structure = CleanTests.make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        starting_structure["external"] = FilesystemStructure({"file1.txt": ("file",)})
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = CleanTests.make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )
        ending_structure["external"] = FilesystemStructure({"file1.txt": ("file",)})

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.scratchspace_path), mode="safe")
        self.assertTrue(
            CleanTests.verify_structure(self.scratchspace_path, ending_structure)
        )


if __name__ == "__main__":
    unittest.main()
