from pathlib import Path
import subprocess
import unittest

import yaml

from tune.env.pg_conn import PostgresConn
from util.pg import get_is_postgres_running, get_running_postgres_ports
from util.workspace import DEFAULT_BOOT_CONFIG_FPATH, DBGymConfig, get_symlinks_path_from_workspace_path, default_pristine_dbdata_snapshot_path, default_dbdata_parent_dpath, default_pgbin_path, get_tmp_path_from_workspace_path


ENV_TESTS_DBGYM_CONFIG_FPATH = Path("tune/env/env_tests_dbgym_config.yaml")
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01
PGPORT = 5432


def get_unittest_workspace_path() -> Path:
    with open(ENV_TESTS_DBGYM_CONFIG_FPATH) as f:
        return Path(yaml.safe_load(f)["dbgym_workspace_path"])
    assert False


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass():
        if not get_unittest_workspace_path().exists():
            subprocess.run(["./tune/env/set_up_env_tests.sh"], check=True)

        PostgresConnTests.dbgym_cfg = DBGymConfig(ENV_TESTS_DBGYM_CONFIG_FPATH)

    def setUp(self):
        self.assertFalse(get_is_postgres_running())
        self.pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            self.dbgym_cfg.dbgym_workspace_path,
            BENCHMARK,
            SCALE_FACTOR
        )
        self.dbdata_parent_dpath = default_dbdata_parent_dpath(self.dbgym_cfg.dbgym_workspace_path)
        self.pgbin_dpath = default_pgbin_path(self.dbgym_cfg.dbgym_workspace_path)

    def tearDown(self):
        self.assertFalse(get_is_postgres_running())

    def create_pg_conn(self) -> PostgresConn:
        return PostgresConn(
            PostgresConnTests.dbgym_cfg,
            PGPORT,
            self.pristine_dbdata_snapshot_path,
            self.dbdata_parent_dpath,
            self.pgbin_dpath,
            False,
            DEFAULT_BOOT_CONFIG_FPATH
        )

    def test_init(self) -> None:
        _ = self.create_pg_conn()

    def test_start_and_stop(self) -> None:
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.start_with_changes()
        self.assertTrue(get_is_postgres_running())
        pg_conn.shutdown_postgres()

    def test_connect_and_disconnect(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.start_with_changes()

        # Test
        self.assertIsNone(pg_conn._conn)
        conn = pg_conn.conn()
        self.assertIsNotNone(conn)
        self.assertIs(conn, pg_conn._conn) # The conn should be cached so these objects should be the same
        self.assertIs(conn, pg_conn.conn()) # Same thing here
        pg_conn.disconnect()
        self.assertIsNone(pg_conn._conn)

        # Cleanup
        pg_conn.shutdown_postgres()


if __name__ == "__main__":
    unittest.main()