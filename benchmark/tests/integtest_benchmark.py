import shutil
import unittest
from pathlib import Path

from gymlib.symlinks_paths import (
    get_tables_symlink_path,
    get_workload_suffix,
    get_workload_symlink_path,
)

# It's ok to import private functions from the benchmark module because this is an integration test.
from benchmark.constants import DEFAULT_SCALE_FACTOR
from benchmark.job.cli import _job_tables, _job_workload
from benchmark.tpch.cli import _tpch_tables, _tpch_workload
from benchmark.tpch.constants import DEFAULT_TPCH_SEED
from util.workspace import (
    DBGymWorkspace,
    fully_resolve_path,
    get_workspace_path_from_config,
)


class TestBenchmark(unittest.TestCase):
    DBGYM_CONFIG_PATH = Path("benchmark/tests/benchmark_integtest_dbgym_config.yaml")

    def setUp(self) -> None:
        workspace_path = get_workspace_path_from_config(TestBenchmark.DBGYM_CONFIG_PATH)
        # Get a clean start each time.
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        self.workspace = DBGymWorkspace(workspace_path)

    def tearDown(self) -> None:
        if self.workspace.dbgym_workspace_path.exists():
            shutil.rmtree(self.workspace.dbgym_workspace_path)

    def test_tpch_tables(self) -> None:
        scale_factor = 0.01
        tables_path = get_tables_symlink_path(
            self.workspace.dbgym_workspace_path, "tpch", scale_factor
        )
        self.assertFalse(tables_path.exists())
        _tpch_tables(self.workspace, scale_factor)
        self.assertTrue(tables_path.exists())
        self.assertTrue(fully_resolve_path(tables_path).exists())

    def test_job_tables(self) -> None:
        tables_path = get_tables_symlink_path(
            self.workspace.dbgym_workspace_path, "job", DEFAULT_SCALE_FACTOR
        )
        self.assertFalse(tables_path.exists())
        _job_tables(self.workspace, DEFAULT_SCALE_FACTOR)
        self.assertTrue(tables_path.exists())
        self.assertTrue(fully_resolve_path(tables_path).exists())

    def test_tpch_workload(self) -> None:
        scale_factor = 0.01
        workload_path = get_workload_symlink_path(
            self.workspace.dbgym_workspace_path,
            "tpch",
            scale_factor,
            get_workload_suffix(
                "tpch",
                seed_start=DEFAULT_TPCH_SEED,
                seed_end=DEFAULT_TPCH_SEED,
                query_subset="all",
            ),
        )
        self.assertFalse(workload_path.exists())
        _tpch_workload(
            self.workspace, DEFAULT_TPCH_SEED, DEFAULT_TPCH_SEED, "all", scale_factor
        )
        self.assertTrue(workload_path.exists())
        self.assertTrue(fully_resolve_path(workload_path).exists())

    def test_job_workload(self) -> None:
        workload_path = get_workload_symlink_path(
            self.workspace.dbgym_workspace_path,
            "job",
            DEFAULT_SCALE_FACTOR,
            get_workload_suffix(
                "job",
                query_subset="all",
            ),
        )
        self.assertFalse(workload_path.exists())
        _job_workload(self.workspace, "all", DEFAULT_SCALE_FACTOR)
        self.assertTrue(workload_path.exists())
        self.assertTrue(fully_resolve_path(workload_path).exists())


if __name__ == "__main__":
    unittest.main()
