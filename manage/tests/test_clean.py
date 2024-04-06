import logging
from pathlib import Path
import unittest
import os
import shutil
import copy

from misc.utils import get_symlinks_path_from_workspace_path, get_runs_path_from_workspace_path
from manage.cli import clean_workspace


# This is here instead of on `if __name__ == "__main__"` because we often run individual tests, which
#   does not go through the `if __name__ == "__main__"` codepath.
# Make it DEBUG to see logs from verify_structure(). Make it INFO to not see logs.
logging.basicConfig(level=logging.DEBUG)


class MockDBGymConfig:
    def __init__(self, scratchspace_path: Path):
        self.dbgym_workspace_path = scratchspace_path
        self.dbgym_symlinks_path = get_symlinks_path_from_workspace_path(scratchspace_path)
        self.dbgym_runs_path = get_runs_path_from_workspace_path(scratchspace_path)


class CleanTests(unittest.TestCase):
    '''
    I deemed "clean" important enough to write unittests before because I'm really paranoid
      about losing files that took 30 hours to build.
    '''
    @staticmethod
    def create_structure(root_path: Path, structure: dict) -> None:
        def create_structure_internal(root_path: Path, cur_path: Path, structure: dict) -> None:
            for path, content in structure.items():
                full_path: Path = cur_path / path
                
                if isinstance(content, dict):  # Directory
                    full_path.mkdir(parents=True, exist_ok=True)
                    create_structure_internal(root_path, full_path, content)
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
    def verify_structure(root_path: Path, structure: dict) -> bool:
        def verify_structure_internal(root_path: Path, cur_path: Path, structure: dict) -> bool:
            # Check for the presence of each item specified in the structure
            for name, item in structure.items():
                new_cur_path = cur_path / name
                if not new_cur_path.exists(follow_symlinks=False):
                    logging.debug(f"{new_cur_path} does not exist")
                    return False
                elif isinstance(item, dict):
                    if not new_cur_path.is_dir():
                        logging.debug(f"expected {new_cur_path} to be a directory")
                        return False
                    if not verify_structure_internal(root_path, new_cur_path, item):
                        return False
                elif isinstance(item, tuple) and item[0] == "file":
                    if not new_cur_path.is_file():
                        logging.debug(f"expected {new_cur_path} to be a regular file")
                        return False
                elif isinstance(item, tuple) and item[0] == "symlink":
                    if not new_cur_path.is_symlink():
                        logging.debug(f"expected {new_cur_path} to be a symlink")
                        return False
                    expected_target = root_path / item[1]
                    if not new_cur_path.resolve().samefile(expected_target):
                        logging.debug(f"expected {new_cur_path} to link to {expected_target}, but it links to {new_cur_path.resolve()}")
                        return False
                else:
                    assert False, "structure misconfigured"
                
            # Check for any extra files or directories not described by the structure
            expected_names = set(structure.keys())
            actual_names = {entry.name for entry in cur_path.iterdir()}
            if not expected_names.issuperset(actual_names):
                logging.debug(f"expected_names={expected_names}, actual_names={actual_names}")
                return False

            return True

        if not root_path.exists():
            logging.debug(f"{root_path} does not exist")
            return False
        return verify_structure_internal(root_path, root_path, structure)

    @staticmethod
    def make_workspace_structure(symlinks_structure: dict, task_runs_structure: dict) -> dict:
        '''
        This function exists so that it's easier to refactor the tests in case we ever change
          how the workspace is organized.
        '''
        return {
            "symlinks": symlinks_structure,
            "task_runs": task_runs_structure,
        }
        
    @classmethod
    def setUpClass(cls):
        cls.scratchspace_path = Path.cwd() / "manage/tests/test_clean_scratchspace/"

    def setUp(self):
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self):
        # DEBUG(phw2)
        # if self.scratchspace_path.exists():
        #     shutil.rmtree(self.scratchspace_path)
        pass

    def test_structure_helpers(self):
        structure = {
            "dir1": {
                "file1.txt": ("file",),
                "dir2": {
                    "file2.txt": ("file",)
                }
            },
            "dir3": {
                "nested_link_to_dir1": ("symlink", "dir1")
            },
            "link_to_dir1": ("symlink", "dir1"),
            "link_to_file2": ("symlink", "dir1/dir2/file2.txt")
        }
        CleanTests.create_structure(self.scratchspace_path, structure)
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, structure))

        extra_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, extra_dir_structure))
        extra_dir_structure["dir4"] = {}
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, extra_dir_structure))

        missing_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, missing_dir_structure))
        del missing_dir_structure["dir1"]
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, missing_dir_structure))

        extra_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, extra_file_structure))
        extra_file_structure["file3.txt"] = ("file",)
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, extra_file_structure))

        missing_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, missing_file_structure))
        del missing_file_structure["dir1"]["file1.txt"]
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, missing_file_structure))

        extra_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, extra_link_structure))
        extra_link_structure["link_to_dir3"] = ("symlink", "dir3")
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, extra_link_structure))

        missing_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, missing_link_structure))
        del missing_link_structure["link_to_dir1"]
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, missing_link_structure))

        wrong_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, wrong_link_structure))
        wrong_link_structure["link_to_dir1"] = ("symlink", "dir3")
        self.assertFalse(CleanTests.verify_structure(self.scratchspace_path, wrong_link_structure))

    def test_nonexistent_workspace(self):
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
    
    def test_no_symlinks_dir_and_no_task_runs_dir(self):
        starting_structure = {}
        ending_structure = {}
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))
    
    def test_no_symlinks_dir_and_yes_task_runs_dir(self):
        starting_structure = {
            "task_runs": {
                "file1.txt": ("file",)
            }
        }
        ending_structure = {
            "task_runs": {}
        }
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))
    
    def test_yes_symlinks_dir_and_no_task_runs_dir(self):
        starting_structure = {
            "symlinks": {}
        }
        ending_structure = {
            "symlinks": {}
        }
        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_no_symlinks_in_dir_and_no_task_runs_in_dir(self):
        starting_symlinks_structure = {}
        starting_task_runs_structure = {}
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {}
        ending_task_runs_structure = {}
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_no_links_in_symlinks(self):
        starting_symlinks_structure = {}
        starting_task_runs_structure = {
            "run_0": {}
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {}
        ending_task_runs_structure = {}
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_file_directly_in_task_runs(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/file1.txt")
        }
        starting_task_runs_structure = {
            "file1.txt": ("file",),
            "file2.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/file1.txt")
        }
        ending_task_runs_structure = {
            "file1.txt": ("file",)
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_dir_directly_in_task_runs(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        starting_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",)
            },
            "dir2": {
                "file2.txt": ("file",)
            }
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        ending_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",)
            }
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_file_in_dir_in_task_runs(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/file1.txt")
        }
        starting_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",)
            },
            "dir2": {
                "file2.txt": ("file",)
            }
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/file1.txt")
        }
        ending_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",)
            }
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_dir_in_dir_in_task_runs(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/dir2")
        }
        starting_task_runs_structure = {
            "dir1": {
                "dir2": {
                    "file1.txt": ("file",)
                },
                "file2.txt": ("file",)
            },
            "dir3": {
                "file3.txt": ("file",)
            }
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/dir2")
        }
        ending_task_runs_structure = {
            "dir1": {
                "dir2": {
                    "file1.txt": ("file",)
                },
                "file2.txt": ("file",)
            },
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path))
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_link_to_link_crashes(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/symlink2")
        }
        starting_task_runs_structure = {
            "symlink2": ("symlink", "task_runs/file1.txt"),
            "file1.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        with self.assertRaises(AssertionError):
            clean_workspace(MockDBGymConfig(self.scratchspace_path))

    def test_safe_mode_link_to_dir_with_link(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        starting_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/file1.txt")
            },
            "file1.txt": ("file",),
            "file2.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        ending_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/file1.txt")
            },
            "file1.txt": ("file",),
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path), mode="safe")
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_safe_mode_link_to_file_in_dir_with_link(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/file1.txt")
        }
        starting_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",),
                "symlink2": ("symlink", "task_runs/file2.txt")
            },
            "file2.txt": ("file",),
            "file3.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1/file1.txt")
        }
        ending_task_runs_structure = {
            "dir1": {
                "file1.txt": ("file",),
                "symlink2": ("symlink", "task_runs/file2.txt")
            },
            "file2.txt": ("file",),
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path), mode="safe")
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_safe_mode_link_to_link_to_file_in_dir_in_task_runs(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        starting_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/dir2/file2.txt")
            },
            "dir2": {
                "file2.txt": ("file",),
            },
            "file3.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        ending_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/dir2/file2.txt")
            },
            "dir2": {
                "file2.txt": ("file",),
            },
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path), mode="safe")
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    def test_aggressive_mode_link_to_dir_with_link(self):
        starting_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        starting_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/file1.txt")
            },
            "file1.txt": ("file",),
            "file2.txt": ("file",)
        }
        starting_structure = CleanTests.make_workspace_structure(starting_symlinks_structure, starting_task_runs_structure)
        ending_symlinks_structure = {
            "symlink1": ("symlink", "task_runs/dir1")
        }
        ending_task_runs_structure = {
            "dir1": {
                "symlink2": ("symlink", "task_runs/file1.txt")
            },
        }
        ending_structure = CleanTests.make_workspace_structure(ending_symlinks_structure, ending_task_runs_structure)

        CleanTests.create_structure(self.scratchspace_path, starting_structure)
        clean_workspace(MockDBGymConfig(self.scratchspace_path), mode="aggressive")
        self.assertTrue(CleanTests.verify_structure(self.scratchspace_path, ending_structure))

    # if aggressive mode, doesn't follow symlinks in task runs

    # don't loop infinitely if there are task_run non-self cycles

    # don't loop infinitely if there are task_run self-cycles (symlink in task run that refers to the same task run)

    # don't loop infinitely if there are symlink non-self cycles

    # don't loop infinitely if there are symlink self cycles (symlink that refers to itself)

    # links to non-existent files are ok

    # links to files outside task_runs are ok, and stuff outside task_runs doesn't get deleted


if __name__ == '__main__':
    unittest.main()