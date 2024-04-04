from pathlib import Path
import unittest
import os
import shutil

class CleanTests(unittest.TestCase):
    @staticmethod
    def create_structure(base_path: Path, structure: dict):
        for path, content in structure.items():
            full_path: Path = base_path / path
            
            if isinstance(content, dict):  # Directory
                full_path.mkdir(parents=True, exist_ok=True)
                CleanTests.create_structure(full_path, content)
            elif isinstance(content, tuple) and content[0] == "file":
                assert len(content) == 1
                full_path.touch()
            elif isinstance(content, tuple) and content[0] == "symlink":
                assert len(content) == 2
                target_path = base_path / content[1]
                os.symlink(target_path, full_path)
            else:
                raise ValueError(f"Unsupported type for path ({path}): {content}")
    
    @classmethod
    def setUpClass(cls):
        cls.scratchspace_path = Path.cwd() / "manage/tests/test_clean_scratchspace/"

    def setUp(self):
        if self.scratchspace_path.exists():
            shutil.rmtree(self.scratchspace_path)

    def tearDown(self):
        pass
        # if self.scratchspace_path.exists():
        #     shutil.rmtree(self.scratchspace_path)

    def test_test(self):
        structure = {
            "dir1": {
                "file1.txt": ("file",),  # Now just an empty string to indicate an empty file
                "dir2": {
                    "file2.txt": ("file",)
                }
            },
            "dir3": {},
            "link_to_dir1": ("symlink", "dir1")
        }
        CleanTests.create_structure(self.scratchspace_path, structure)