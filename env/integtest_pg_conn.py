import copy
import subprocess
import unittest
from pathlib import Path

import yaml
from psycopg.errors import QueryCanceled

from env.pg_conn import PostgresConn
from util.pg import (
    DEFAULT_POSTGRES_PORT,
    get_is_postgres_running,
    get_running_postgres_ports,
)
from util.workspace import (
    DEFAULT_BOOT_CONFIG_FPATH,
    DBGymConfig,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    default_pristine_dbdata_snapshot_path,
)

ENV_INTEGTESTS_DBGYM_CONFIG_FPATH = Path("env/env_integtests_dbgym_config.yaml")
BENCHMARK = "tpch"
SCALE_FACTOR = 0.01


def get_unittest_workspace_path() -> Path:
    with open(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH) as f:
        return Path(yaml.safe_load(f)["dbgym_workspace_path"])
    assert False


class PostgresConnTests(unittest.TestCase):
    dbgym_cfg: DBGymConfig

    @staticmethod
    def setUpClass() -> None:
        # If you're running the test locally, this check makes runs past the first one much faster.
        if not get_unittest_workspace_path().exists():
            subprocess.run(["./env/set_up_env_integtests.sh"], check=True)

        PostgresConnTests.dbgym_cfg = DBGymConfig(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH)

    def setUp(self) -> None:
        self.assertFalse(
            get_is_postgres_running(),
            "Make sure Postgres isn't running before starting the integration test. `pkill postgres` is one way "
            + "to ensure this. Be careful about accidentally taking down other people's Postgres instances though.",
        )
        self.pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            self.dbgym_cfg.dbgym_workspace_path, BENCHMARK, SCALE_FACTOR
        )
        self.dbdata_parent_dpath = default_dbdata_parent_dpath(
            self.dbgym_cfg.dbgym_workspace_path
        )
        self.pgbin_dpath = default_pgbin_path(self.dbgym_cfg.dbgym_workspace_path)

    def tearDown(self) -> None:
        self.assertFalse(get_is_postgres_running())

    def create_pg_conn(self, pgport: int = DEFAULT_POSTGRES_PORT) -> PostgresConn:
        return PostgresConn(
            PostgresConnTests.dbgym_cfg,
            pgport,
            self.pristine_dbdata_snapshot_path,
            self.dbdata_parent_dpath,
            self.pgbin_dpath,
            False,
            DEFAULT_BOOT_CONFIG_FPATH,
        )

    def test_init(self) -> None:
        _ = self.create_pg_conn()

    def test_start_and_stop(self) -> None:
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()
        self.assertTrue(get_is_postgres_running())
        pg_conn.shutdown_postgres()

    def test_start_on_multiple_ports(self) -> None:
        pg_conn0 = self.create_pg_conn()
        pg_conn0.restore_pristine_snapshot()
        pg_conn0.restart_postgres()
        self.assertEqual(set(get_running_postgres_ports()), {DEFAULT_POSTGRES_PORT})
        pg_conn1 = self.create_pg_conn(DEFAULT_POSTGRES_PORT + 1)
        pg_conn1.restore_pristine_snapshot()
        pg_conn1.restart_postgres()
        self.assertEqual(
            set(get_running_postgres_ports()),
            {DEFAULT_POSTGRES_PORT, DEFAULT_POSTGRES_PORT + 1},
        )

        # Clean up
        pg_conn0.shutdown_postgres()
        pg_conn1.shutdown_postgres()

    def test_connect_and_disconnect(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()

        # Test
        self.assertIsNone(pg_conn._conn)
        conn = pg_conn.conn()
        self.assertIsNotNone(conn)
        self.assertIs(
            conn, pg_conn._conn
        )  # The conn should be cached so these objects should be the same
        self.assertIs(conn, pg_conn.conn())  # Same thing here
        pg_conn.disconnect()
        self.assertIsNone(pg_conn._conn)

        # Cleanup
        pg_conn.shutdown_postgres()

    def test_start_with_changes(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()

        # Test
        initial_sysknobs = pg_conn.get_system_knobs()
        self.assertEqual(initial_sysknobs["wal_buffers"], "4MB")
        pg_conn.restart_with_changes({"wal_buffers": "8MB"})
        new_sysknobs = pg_conn.get_system_knobs()
        self.assertEqual(new_sysknobs["wal_buffers"], "8MB")

        # Cleanup
        pg_conn.shutdown_postgres()

    def test_multiple_start_with_changes(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()

        # Test
        initial_sysknobs = pg_conn.get_system_knobs()

        # First call
        self.assertEqual(initial_sysknobs["wal_buffers"], "4MB")
        pg_conn.restart_with_changes({"wal_buffers": "8MB"})
        new_sysknobs = pg_conn.get_system_knobs()
        self.assertEqual(new_sysknobs["wal_buffers"], "8MB")

        # Second call
        self.assertEqual(initial_sysknobs["enable_nestloop"], "on")
        pg_conn.restart_with_changes({"enable_nestloop": "off"})
        new_sysknobs = pg_conn.get_system_knobs()
        self.assertEqual(new_sysknobs["enable_nestloop"], "off")
        # The changes should not be additive. The "wal_buffers" should have "reset" to 4MB.
        self.assertEqual(new_sysknobs["wal_buffers"], "4MB")

        # Cleanup
        pg_conn.shutdown_postgres()

    def test_start_with_changes_doesnt_modify_input(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()

        # Test
        conf_changes = {"wal_buffers": "8MB"}
        orig_conf_changes = copy.deepcopy(conf_changes)
        pg_conn.restart_with_changes(conf_changes)
        self.assertEqual(conf_changes, orig_conf_changes)

        # Cleanup
        pg_conn.shutdown_postgres()

    def test_time_query(self) -> None:
        # Setup
        pg_conn = self.create_pg_conn()
        pg_conn.restore_pristine_snapshot()
        pg_conn.restart_postgres()

        # Test
        # Testing no explain no timeout.
        runtime, did_time_out, explain_data = pg_conn.time_query("select pg_sleep(1)")
        # The runtime should be about 1 second.
        self.assertTrue(abs(runtime - 1_000_000) < 100_000)
        self.assertFalse(did_time_out)
        self.assertIsNone(explain_data)

        # Testing with explain.
        runtime, did_time_out, explain_data = pg_conn.time_query(
            "explain (analyze, format json, timing off) select pg_sleep(1)"
        )
        self.assertTrue(abs(runtime - 1_000_000) < 100_000)
        self.assertFalse(did_time_out)
        self.assertIsNotNone(explain_data)

        # Testing with timeout.
        runtime, did_time_out, _ = pg_conn.time_query("select pg_sleep(3)", 2)
        # The runtime should be about what the timeout is.
        self.assertTrue(abs(runtime - 2_000_000) < 100_000)
        self.assertTrue(did_time_out)

        # Cleanup
        pg_conn.shutdown_postgres()


if __name__ == "__main__":
    unittest.main()
