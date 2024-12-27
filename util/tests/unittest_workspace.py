# TODO: figure out where to put the filesystem structure helpers. I think I want to put them inside gymlib and make a separate folder just testing the helpers.

from pathlib import Path
import shutil
import unittest

from manage.tests.unittest_clean import FilesystemStructure


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

    def test_workspace_init(self) -> None:
        pass
