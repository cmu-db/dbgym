'''
At a high level, this file's goal is to provide helpers to manage a Postgres instance during
    agent tuning.
On the other hand, the goal of dbms.postgres.cli is to (1) install+build postgres and (2)
    create pgdata.
util.pg provides helpers used by *both* of the above files (as well as other files).
'''
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import psutil
import psycopg
from plumbum import local
from psycopg.errors import ProgramLimitExceeded, QueryCanceled

from tune.protox.env.logger import Logger, time_record


class PostgresConn:
    def __init__(
        self,
        postgres_conn: str,
        postgres_data: Union[str, Path],
        postgres_path: Union[str, Path],
        postgres_logs_dir: Union[str, Path],
        connect_timeout: int,
        logger: Logger,
    ) -> None:

        Path(postgres_logs_dir).mkdir(parents=True, exist_ok=True)
        self.postgres_path = postgres_path
        self.postgres_data = postgres_data
        self.postgres_logs_dir = postgres_logs_dir
        self.connection = postgres_conn
        self.connect_timeout = connect_timeout
        self.log_step = 0
        self.logger = logger

        kvs = {s.split("=")[0]: s.split("=")[1] for s in postgres_conn.split(" ")}
        self.postgres_db = kvs["dbname"]
        self.postgres_host = kvs["host"]
        self.postgres_port = kvs["port"]
        self._conn: Optional[psycopg.Connection[Any]] = None

    def conn(self) -> psycopg.Connection[Any]:
        if self._conn is None:
            self._conn = psycopg.connect(
                self.connection, autocommit=True, prepare_threshold=None
            )
        return self._conn

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def move_log(self) -> None:
        if Path(f"{self.postgres_logs_dir}/pg.log").exists():
            shutil.move(
                f"{self.postgres_logs_dir}/pg.log",
                f"{self.postgres_logs_dir}/pg.log.{self.log_step}",
            )
            self.log_step += 1

    @time_record("shutdown")
    def shutdown_postgres(self) -> None:
        """Shuts down postgres."""
        self.disconnect()
        if not Path(self.postgres_data).exists():
            return

        while True:
            self.logger.get_logger(__name__).debug("Shutting down postgres...")
            _, stdout, stderr = local[f"{self.postgres_path}/pg_ctl"][
                "stop", "--wait", "-t", "180", "-D", self.postgres_data
            ].run(retcode=None)
            time.sleep(1)
            self.logger.get_logger(__name__).debug(
                "Stop message: (%s, %s)", stdout, stderr
            )

            # Wait until pg_isready fails.
            retcode, _, _ = local[f"{self.postgres_path}/pg_isready"][
                "--host",
                self.postgres_host,
                "--port",
                str(self.postgres_port),
                "--dbname",
                self.postgres_db,
            ].run(retcode=None)

            exists = (Path(self.postgres_data) / "postmaster.pid").exists()
            if not exists and retcode != 0:
                break

    @time_record("start")
    def start_with_changes(
        self,
        conf_changes: Optional[list[str]] = None,
        dump_page_cache: bool = False,
        save_snapshot: bool = False,
    ) -> bool:
        # Install the new configuration changes.
        if conf_changes is not None:
            conf_changes.append("shared_preload_libraries='pg_hint_plan'")
            with open(f"{self.postgres_data}/postgresql.auto.conf", "w") as f:
                f.write("\n".join(conf_changes))

        # Start postgres instance.
        self.shutdown_postgres()
        self.move_log()

        if save_snapshot:
            # Create an archive of pgdata as a snapshot.
            local["tar"][
                "cf",
                f"{self.postgres_data}.tgz.tmp",
                "-C",
                self.postgres_path,
                self.postgres_data,
            ].run()

        # Make sure the PID lock file doesn't exist.
        pid_lock = Path(f"{self.postgres_data}/postmaster.pid")
        assert not pid_lock.exists()

        if dump_page_cache:
            # Dump the OS page cache.
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        attempts = 0
        while not pid_lock.exists():
            # Try starting up.
            retcode, stdout, stderr = local[f"{self.postgres_path}/pg_ctl"][
                "-D",
                self.postgres_data,
                "--wait",
                "-t",
                "180",
                "-l",
                f"{self.postgres_logs_dir}/pg.log",
                "start",
            ].run(retcode=None)

            if retcode == 0 or pid_lock.exists():
                break

            self.logger.get_logger(__name__).warn(
                "startup encountered: (%s, %s)", stdout, stderr
            )
            attempts += 1
            if attempts >= 5:
                self.logger.get_logger(__name__).error(
                    "Number of attempts to start postgres has exceeded limit."
                )
                assert False, "Could not start postgres."

        # Wait until postgres is ready to accept connections.
        num_cycles = 0
        while True:
            if self.connect_timeout is not None and num_cycles >= self.connect_timeout:
                # In this case, we've failed to start postgres.
                self.logger.get_logger(__name__).error(
                    "Failed to start postgres before timeout..."
                )
                return False

            retcode, _, _ = local[f"{self.postgres_path}/pg_isready"][
                "--host",
                self.postgres_host,
                "--port",
                str(self.postgres_port),
                "--dbname",
                self.postgres_db,
            ].run(retcode=None)
            if retcode == 0:
                break

            time.sleep(1)
            num_cycles += 1
            self.logger.get_logger(__name__).debug(
                "Waiting for postgres to bootup but it is not..."
            )

        # Copy the temporary over since we know the temporary can load.
        if save_snapshot:
            shutil.move(f"{self.postgres_data}.tgz.tmp", f"{self.postgres_data}.tgz")

        return True

    @time_record("psql")
    def psql(self, sql: str) -> Tuple[int, Optional[str]]:
        low_sql = sql.lower()

        def cancel_fn(conn_str: str) -> None:
            with psycopg.connect(
                conn_str, autocommit=True, prepare_threshold=None
            ) as tconn:
                r = [
                    r
                    for r in tconn.execute(
                        "SELECT pid FROM pg_stat_progress_create_index"
                    )
                ]

            for row in r:
                self.logger.get_logger(__name__).info(f"Killing process {row[0]}")
                try:
                    psutil.Process(row[0]).kill()
                except:
                    pass

        # Get a fresh connection.
        self.disconnect()
        conn = self.conn()
        conn.execute("SET maintenance_work_mem = '4GB'")
        # TODO(wz2): Make this a configuration/runtime option for action timeout.
        conn.execute("SET statement_timeout = 300000")

        try:
            timer = threading.Timer(300.0, cancel_fn, args=(self.connection,))
            timer.start()

            conn.execute(sql)
            timer.cancel()
        except ProgramLimitExceeded as e:
            timer.cancel()
            self.disconnect()
            self.logger.get_logger(__name__).debug(f"Action error: {e}")
            return -1, str(e)
        except QueryCanceled as e:
            timer.cancel()
            self.disconnect()
            self.logger.get_logger(__name__).debug(f"Action error: {e}")
            return -1, f"canceling statement: {sql}."
        except psycopg.OperationalError as e:
            timer.cancel()
            self.disconnect()
            self.logger.get_logger(__name__).debug(f"Action error: {e}")
            return -1, f"operational error: {sql}."
        except psycopg.errors.UndefinedTable:
            timer.cancel()
            raise

        self.disconnect()
        return 0, None

    @time_record("restore")
    def restore_snapshot(
        self, archive: Union[str, Path, None] = None, last: bool = False
    ) -> bool:
        assert archive or last
        self.shutdown_postgres()

        local["rm"]["-rf", self.postgres_data].run()
        local["mkdir"]["-m", "0700", "-p", self.postgres_data].run()

        # Strip the "pgdata" so we can implant directly into the target postgres_data.
        archive = archive if not last else f"{self.postgres_data}.tgz"
        assert archive
        assert Path(archive).exists()

        local["tar"][
            "xf", archive, "-C", self.postgres_data, "--strip-components", "1"
        ].run()
        # Imprint the required port.
        (
            (local["echo"][f"port={self.postgres_port}"])
            >> f"{self.postgres_data}/postgresql.conf"
        )()

        return self.start_with_changes(conf_changes=None)
