import shutil
import unittest
from pathlib import Path

from gymlib.symlinks_paths import get_tables_symlink_path

# It's ok to import private functions from the benchmark module because this is an integration test.
from benchmark.tpch.cli import _tpch_tables
from util.workspace import DBGymWorkspace, get_workspace_path_from_config


class TestBenchmark(unittest.TestCase):
    DBGYM_CONFIG_PATH = Path("benchmark/tests/benchmark_integtest_dbgym_config.yaml")

    def setUp(self) -> None:
        workspace_path = get_workspace_path_from_config(TestBenchmark.DBGYM_CONFIG_PATH)
        # Get a clean start each time.
        shutil.rmtree(workspace_path)
        self.workspace = DBGymWorkspace(workspace_path)

    # def tearDown(self) -> None:
    #     shutil.rmtree(self.workspace.dbgym_workspace_path)

    def test_tpch_tables(self) -> None:
        scale_factor = 0.01
        tables_path = get_tables_symlink_path(
            self.workspace.dbgym_workspace_path, "tpch", scale_factor
        )
        self.assertFalse(tables_path.exists())
        _tpch_tables(self.workspace, scale_factor)
        self.assertTrue(tables_path.exists())


if __name__ == "__main__":
    unittest.main()
