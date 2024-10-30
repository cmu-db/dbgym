from pathlib import Path
import subprocess
import unittest

import yaml

from tune.env.pg_conn import PostgresConn
from util.workspace import DEFAULT_BOOT_CONFIG_FPATH, get_symlinks_path_from_workspace_path, default_pristine_dbdata_snapshot_path, default_dbdata_parent_dpath, default_pgbin_path, get_tmp_path_from_workspace_path


ENV_TESTS_DBGYM_CONFIG_FPATH = Path("tune/env/env_tests_dbgym_config.yaml")
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01
PGPORT = 5432


def get_unittest_workspace_path() -> Path:
    with open(ENV_TESTS_DBGYM_CONFIG_FPATH) as f:
        return Path(yaml.safe_load(f)["dbgym_workspace_path"])
    assert False


class MockDBGymConfig:
    def __init__(self):
        self.dbgym_workspace_path = get_unittest_workspace_path()
        self.dbgym_symlinks_path = get_symlinks_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        self.dbgym_tmp_path = get_tmp_path_from_workspace_path(
            self.dbgym_workspace_path
        )


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass():
        if not get_unittest_workspace_path().exists():
            subprocess.run(["./tune/env/set_up_env_tests.sh"], check=True)

    def setUp(self):
        self.dbgym_cfg = MockDBGymConfig()
        self.pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            self.dbgym_cfg.dbgym_workspace_path,
            BENCHMARK,
            SCALE_FACTOR
        )
        self.dbdata_parent_dpath = default_dbdata_parent_dpath(self.dbgym_cfg.dbgym_workspace_path)
        self.pgbin_dpath = default_pgbin_path(self.dbgym_cfg.dbgym_workspace_path)

    def create_pg_conn(self) -> PostgresConn:
        return PostgresConn(
            self.dbgym_cfg,
            PGPORT,
            self.pristine_dbdata_snapshot_path,
            self.dbdata_parent_dpath,
            self.pgbin_dpath,
            False,
            DEFAULT_BOOT_CONFIG_FPATH
        )

    def test_init(self) -> None:
        _ = self.create_pg_conn()


if __name__ == "__main__":
    unittest.main()