import logging
import shutil
import unittest
from pathlib import Path

from gymlib.tests.filesystem_unittest_util import (
    FilesystemStructure,
    create_structure,
    make_workspace_structure,
    verify_structure,
)

from manage.cli import MockDBGymWorkspace, clean_workspace

# This is here instead of on `if __name__ == "__main__"` because we often run individual tests, which
#   does not go through the `if __name__ == "__main__"` codepath.
# Make it DEBUG to see logs from verify_structure(). Make it CRITICAL to not see any logs.
# We use the root logger for unit tests to keep it separate from the standard logging subsystem which
#   uses the dbgym.* loggers.
logging.basicConfig(level=logging.CRITICAL)


class CleanTests(unittest.TestCase):
    scratchspace_path: Path = Path()
    workspace_path: Path = Path()

    @classmethod
    def setUpClass(cls) -> None:
        cls.scratchspace_path = Path.cwd() / "manage/tests/test_clean_scratchspace/"
        cls.workspace_path = cls.scratchspace_path / "dbgym_workspace"

    def setUp(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def test_nonexistent_workspace(self) -> None:
        # This just ensures that it doesn't raise an exception.
        clean_workspace(MockDBGymWorkspace(self.workspace_path))

    def test_empty_workspace(self) -> None:
        starting_structure = FilesystemStructure({"dbgym_workspace": {}})
        ending_structure = FilesystemStructure({"dbgym_workspace": {}})
        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_no_symlinks_dir_and_yes_task_runs_dir(self) -> None:
        starting_structure = FilesystemStructure(
            {"dbgym_workspace": {"task_runs": {"file1.txt": ("file",)}}}
        )
        ending_structure = FilesystemStructure({"dbgym_workspace": {"task_runs": {}}})
        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_yes_symlinks_dir_and_no_task_runs_dir(self) -> None:
        # If there are no task runs there can't be any symlinks.
        starting_structure = FilesystemStructure({"dbgym_workspace": {"symlinks": {}}})
        ending_structure = FilesystemStructure({"dbgym_workspace": {"symlinks": {}}})
        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_no_symlinks_in_dir_and_no_task_runs_in_dir(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({})
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_no_links_in_symlinks(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({"run_0": {}})
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_file_directly_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {"file1.txt": ("file",), "file2.txt": ("file",)}
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure({"file1.txt": ("file",)})
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_dir_directly_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",)},
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",)}}
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_file_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",)},
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",)}}
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_dir_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/dir2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"dir2": {"file1.txt": ("file",)}, "file2.txt": ("file",)},
                "dir3": {"file3.txt": ("file",)},
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/dir2")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"dir2": {"file1.txt": ("file",)}, "file2.txt": ("file",)},
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path))
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_link_crashes(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "symlink2": ("symlink", "dbgym_workspace/task_runs/file1.txt"),
                "file1.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        with self.assertRaises(AssertionError):
            clean_workspace(MockDBGymWorkspace(self.workspace_path))

    def test_safe_mode_link_to_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file1.txt")
                },
                "file1.txt": ("file",),
                "file2.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file1.txt")
                },
                "file1.txt": ("file",),
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_safe_mode_link_to_file_in_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file2.txt"),
                },
                "file2.txt": ("file",),
                "file3.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file2.txt"),
                },
                "file2.txt": ("file",),
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_safe_mode_link_to_dir_with_link_to_file_in_dir_in_task_runs(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir2/file2.txt")
                },
                "dir2": {
                    "file2.txt": ("file",),
                },
                "file3.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir2/file2.txt")
                },
                "dir2": {
                    "file2.txt": ("file",),
                },
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_aggressive_mode_link_to_dir_with_link(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file1.txt")
                },
                "file1.txt": ("file",),
                "file2.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", None)},
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="aggressive")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_link_to_file_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/file2.txt")
                },
                "file2.txt": ("file",),
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)

        # We disallow links to links so it's an AssertionError
        with self.assertRaises(AssertionError):
            clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")

    def test_multi_link_loop_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/symlink2")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {"symlink2": ("symlink", "dbgym_workspace/symlinks/symlink1")},
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)

        # pathlib disallows multi-link loops so it's a RuntimeError
        with self.assertRaises(RuntimeError):
            clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")

    def test_link_self_loop_gives_error(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/symlinks/symlink1")}
        )
        starting_task_runs_structure = FilesystemStructure({})
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)

        # pathlib disallows link self-loops so it's a RuntimeError
        with self.assertRaises(RuntimeError):
            clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")

    def test_dont_loop_infinitely_if_there_are_cycles_between_different_dirs_in_runs(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir2/file2.txt"),
                },
                "dir2": {
                    "file2.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir2/file2.txt"),
                },
                "dir2": {
                    "file2.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_dont_loop_infinitely_if_there_is_a_dir_in_runs_that_links_to_a_file_in_itself(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_dont_loop_infinitely_if_there_is_loop_amongst_symlinks(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt"),
                },
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_broken_symlink_has_no_effect(self) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        starting_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": (
                        "symlink",
                        "dbgym_workspace/task_runs/dir1/non_existent_file.txt",
                    ),
                },
                "dir2": {"file2.txt": ("file",)},
            }
        )
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {"dir1": {"file1.txt": ("file",), "symlink2": ("symlink", None)}}
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    # The idea behind this test is that we shouldn't be following links outside of task_runs, even on safe mode
    def test_link_to_folder_outside_runs_that_contains_link_to_other_run_doesnt_save_other_run(
        self,
    ) -> None:
        starting_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
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
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        starting_structure["external"] = FilesystemStructure(
            {
                "dir3": {
                    "file3.txt": ("file",),
                    "symlink3": ("symlink", "dbgym_workspace/task_runs/dir2/file2.txt"),
                }
            }
        )
        ending_symlinks_structure = FilesystemStructure(
            {"symlink1": ("symlink", "dbgym_workspace/task_runs/dir1/file1.txt")}
        )
        ending_task_runs_structure = FilesystemStructure(
            {
                "dir1": {
                    "file1.txt": ("file",),
                    "symlink2": ("symlink", "external/dir3/file3.txt"),
                }
            }
        )
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )
        ending_structure["external"] = {
            "dir3": {"file3.txt": ("file",), "symlink3": ("symlink", None)}
        }

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))

    def test_outside_task_runs_doesnt_get_deleted(self) -> None:
        starting_symlinks_structure = FilesystemStructure({})
        starting_task_runs_structure = FilesystemStructure({"dir1": {}})
        starting_structure = make_workspace_structure(
            starting_symlinks_structure, starting_task_runs_structure
        )
        starting_structure["external"] = FilesystemStructure({"file1.txt": ("file",)})
        ending_symlinks_structure = FilesystemStructure({})
        ending_task_runs_structure = FilesystemStructure({})
        ending_structure = make_workspace_structure(
            ending_symlinks_structure, ending_task_runs_structure
        )
        ending_structure["external"] = FilesystemStructure({"file1.txt": ("file",)})

        create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymWorkspace(self.workspace_path), mode="safe")
        self.assertTrue(verify_structure(self.scratchspace_path, ending_structure))


if __name__ == "__main__":
    unittest.main()
