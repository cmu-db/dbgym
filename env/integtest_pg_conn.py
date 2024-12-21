import copy
import unittest

import psycopg

from env.integtest_util import IntegtestWorkspace
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

BENCHMARK = "tpch"
SCALE_FACTOR = 0.01


class PostgresConnTests(unittest.TestCase):
    dbgym_cfg: DBGymConfig

    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    def setUp(self) -> None:
        self.assertFalse(
            get_is_postgres_running(),
            "Make sure Postgres isn't running before starting the integration test. `pkill postgres` is one way "
            + "to ensure this. Be careful about accidentally taking down other people's Postgres instances though.",
        )
        self.pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            IntegtestWorkspace.get_dbgym_cfg().dbgym_workspace_path,
            BENCHMARK,
            SCALE_FACTOR,
        )
        self.dbdata_parent_dpath = default_dbdata_parent_dpath(
            IntegtestWorkspace.get_dbgym_cfg().dbgym_workspace_path
        )
        self.pgbin_dpath = default_pgbin_path(
            IntegtestWorkspace.get_dbgym_cfg().dbgym_workspace_path
        )

        # The reason we restart Postgres every time is to ensure a "clean" starting point
        # so that all tests are independent of each other.
        self.pg_conn = self.create_pg_conn()
        self.pg_conn.restore_pristine_snapshot()
        self.pg_conn.restart_postgres()
        self.assertTrue(get_is_postgres_running())

    def tearDown(self) -> None:
        self.pg_conn.shutdown_postgres()
        self.assertFalse(get_is_postgres_running())

    def create_pg_conn(self, pgport: int = DEFAULT_POSTGRES_PORT) -> PostgresConn:
        return PostgresConn(
            IntegtestWorkspace.get_dbgym_cfg(),
            pgport,
            self.pristine_dbdata_snapshot_path,
            self.dbdata_parent_dpath,
            self.pgbin_dpath,
            False,
            DEFAULT_BOOT_CONFIG_FPATH,
        )

    def test_start_on_multiple_ports(self) -> None:
        # The setUp() function should have started Postgres on DEFAULT_POSTGRES_PORT.
        self.assertEqual(set(get_running_postgres_ports()), {DEFAULT_POSTGRES_PORT})

        # Now, we start Postgres on a new port.
        pg_conn1 = self.create_pg_conn(DEFAULT_POSTGRES_PORT + 1)
        pg_conn1.restore_pristine_snapshot()
        pg_conn1.restart_postgres()
        self.assertEqual(
            set(get_running_postgres_ports()),
            {DEFAULT_POSTGRES_PORT, DEFAULT_POSTGRES_PORT + 1},
        )

        # Clean up
        pg_conn1.shutdown_postgres()

    def test_connect_and_disconnect(self) -> None:
        self.assertIsNone(self.pg_conn._conn)
        conn = self.pg_conn.conn()
        self.assertIsNotNone(conn)
        self.assertIs(
            conn, self.pg_conn._conn
        )  # The conn should be cached so these objects should be the same
        self.assertIs(conn, self.pg_conn.conn())  # Same thing here
        self.pg_conn.disconnect()
        self.assertIsNone(self.pg_conn._conn)

    def test_start_with_changes(self) -> None:
        initial_sysknobs = self.pg_conn.get_system_knobs()

        # First call
        self.assertEqual(initial_sysknobs["wal_buffers"], "4MB")
        self.pg_conn.restart_with_changes({"wal_buffers": "8MB"})
        new_sysknobs = self.pg_conn.get_system_knobs()
        self.assertEqual(new_sysknobs["wal_buffers"], "8MB")

        # Second call
        self.assertEqual(initial_sysknobs["enable_nestloop"], "on")
        self.pg_conn.restart_with_changes({"enable_nestloop": "off"})
        new_sysknobs = self.pg_conn.get_system_knobs()
        self.assertEqual(new_sysknobs["enable_nestloop"], "off")
        # The changes should not be additive. The "wal_buffers" should have "reset" to 4MB.
        self.assertEqual(new_sysknobs["wal_buffers"], "4MB")

    def test_start_with_changes_doesnt_modify_input(self) -> None:
        conf_changes = {"wal_buffers": "8MB"}
        orig_conf_changes = copy.deepcopy(conf_changes)
        self.pg_conn.restart_with_changes(conf_changes)
        self.assertEqual(conf_changes, orig_conf_changes)

    def test_time_query(self) -> None:
        runtime, did_time_out, explain_data = self.pg_conn.time_query(
            "select pg_sleep(1)"
        )
        # The runtime should be about 1 second.
        self.assertTrue(abs(runtime - 1_000_000) < 100_000)
        self.assertFalse(did_time_out)
        self.assertIsNone(explain_data)

    def test_time_query_with_explain(self) -> None:
        runtime, did_time_out, explain_data = self.pg_conn.time_query(
            "select pg_sleep(1)", add_explain=True
        )
        self.assertTrue(abs(runtime - 1_000_000) < 100_000)
        self.assertFalse(did_time_out)
        self.assertIsNotNone(explain_data)

    def test_time_query_with_timeout(self) -> None:
        runtime, did_time_out, explain_data = self.pg_conn.time_query(
            "select pg_sleep(3)", timeout=2
        )
        # The runtime should be about what the timeout is.
        self.assertTrue(abs(runtime - 2_000_000) < 100_000)
        self.assertTrue(did_time_out)
        self.assertIsNone(explain_data)

    def test_time_query_with_valid_table(self) -> None:
        _, did_time_out, explain_data = self.pg_conn.time_query(
            "select * from lineitem limit 10"
        )
        self.assertFalse(did_time_out)
        self.assertIsNone(explain_data)

    def test_time_query_with_invalid_table(self) -> None:
        with self.assertRaises(psycopg.errors.UndefinedTable):
            self.pg_conn.time_query(
                "select * from itemline limit 10"
            )


if __name__ == "__main__":
    unittest.main()
