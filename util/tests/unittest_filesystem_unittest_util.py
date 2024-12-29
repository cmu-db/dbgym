import copy
import shutil
import unittest
from pathlib import Path

from util.tests.filesystem_unittest_util import (
    FilesystemStructure,
    create_structure,
    verify_structure,
)


class FilesystemUnittestUtilTests(unittest.TestCase):
    scratchspace_path: Path = Path()

    @classmethod
    def setUpClass(cls) -> None:
        cls.scratchspace_path = (
            Path.cwd() / "util/tests/test_filesystem_unittest_util_scratchspace/"
        )

    def setUp(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self) -> None:
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def test_filesystem_unittest_util(self) -> None:
        structure = FilesystemStructure(
            {
                "dir1": {"file1.txt": ("file",), "dir2": {"file2.txt": ("file",)}},
                "dir3": {"nested_link_to_dir1": ("symlink", "dir1")},
                "link_to_dir1": ("symlink", "dir1"),
                "link_to_file2": ("symlink", "dir1/dir2/file2.txt"),
            }
        )
        create_structure(self.scratchspace_path, structure)
        self.assertTrue(verify_structure(self.scratchspace_path, structure))

        extra_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(verify_structure(self.scratchspace_path, extra_dir_structure))
        extra_dir_structure["dir4"] = {}
        self.assertFalse(verify_structure(self.scratchspace_path, extra_dir_structure))

        missing_dir_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(verify_structure(self.scratchspace_path, missing_dir_structure))
        del missing_dir_structure["dir1"]
        self.assertFalse(
            verify_structure(self.scratchspace_path, missing_dir_structure)
        )

        extra_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(verify_structure(self.scratchspace_path, extra_file_structure))
        extra_file_structure["file3.txt"] = ("file",)
        self.assertFalse(verify_structure(self.scratchspace_path, extra_file_structure))

        missing_file_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            verify_structure(self.scratchspace_path, missing_file_structure)
        )
        del missing_file_structure["dir1"]["file1.txt"]
        self.assertFalse(
            verify_structure(self.scratchspace_path, missing_file_structure)
        )

        extra_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(verify_structure(self.scratchspace_path, extra_link_structure))
        extra_link_structure["link_to_dir3"] = ("symlink", "dir3")
        self.assertFalse(verify_structure(self.scratchspace_path, extra_link_structure))

        missing_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(
            verify_structure(self.scratchspace_path, missing_link_structure)
        )
        del missing_link_structure["link_to_dir1"]
        self.assertFalse(
            verify_structure(self.scratchspace_path, missing_link_structure)
        )

        wrong_link_structure = copy.deepcopy(structure)
        # The "assertTrue, modify, assertFalse" patterns makes sure it was the modification that broke it
        self.assertTrue(verify_structure(self.scratchspace_path, wrong_link_structure))
        wrong_link_structure["link_to_dir1"] = ("symlink", "dir3")
        self.assertFalse(verify_structure(self.scratchspace_path, wrong_link_structure))


if __name__ == "__main__":
    unittest.main()
