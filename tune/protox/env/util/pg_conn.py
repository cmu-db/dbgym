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
from misc.utils import DBGymConfig, parent_dir
from util.pg import DBGYM_POSTGRES_USER, DBGYM_POSTGRES_PASS, DBGYM_POSTGRES_DBNAME


class PostgresConn:
    def __init__(
        self,
        dbgym_cfg: DBGymConfig,
        pgport: int,
        pristine_pgdata_snapshot_fpath: Path,
        pgbin_dpath: Union[str, Path],
        postgres_logs_dir: Union[str, Path],
        connect_timeout: int,
        logger: Logger,
    ) -> None:

        Path(postgres_logs_dir).mkdir(parents=True, exist_ok=True)
        self.dbgym_cfg = dbgym_cfg
        self.pgport = pgport
        self.pgbin_dpath = pgbin_dpath
        self.postgres_logs_dir = postgres_logs_dir
        self.connect_timeout = connect_timeout
        self.log_step = 0
        self.logger = logger

        # All the paths related to pgdata
        # pristine_pgdata_snapshot_fpath is the .tgz snapshot that represents the starting state
        #   of the database (with the default configuration). It is generated by a call to
        #   `python tune.py dbms postgres ...` and should not be overwritten.
        self.pristine_pgdata_snapshot_fpath = pristine_pgdata_snapshot_fpath
        # checkpoint_pgdata_snapshot_fpath is the .tgz snapshot that represents the current
        #   state of the database as it is being tuned. It is generated while tuning and is
        #   discarded once tuning is completed.
        self.checkpoint_pgdata_snapshot_fpath = dbgym_cfg.dbgym_tmp_path / "checkpoint_pgdata.tgz"
        # pgdata_dpath is the pgdata that is *actively being tuned*
        self.pgdata_dpath = dbgym_cfg.dbgym_tmp_path / f"pgdata{self.pgport}"

        self._conn: Optional[psycopg.Connection[Any]] = None

    def _get_connstr(self):
        return f"host=localhost port={self.pgport} user={DBGYM_POSTGRES_USER} password={DBGYM_POSTGRES_PASS} dbname={DBGYM_POSTGRES_DBNAME}"

    def conn(self) -> psycopg.Connection[Any]:
        if self._conn is None:
            self._conn = psycopg.connect(
                self._get_connstr(), autocommit=True, prepare_threshold=None
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
        if not Path(self.pgdata_dpath).exists():
            return

        while True:
            self.logger.get_logger(__name__).debug("Shutting down postgres...")
            _, stdout, stderr = local[f"{self.pgbin_dpath}/pg_ctl"][
                "stop", "--wait", "-t", "180", "-D", self.pgdata_dpath
            ].run(retcode=None)
            time.sleep(1)
            self.logger.get_logger(__name__).debug(
                "Stop message: (%s, %s)", stdout, stderr
            )

            # Wait until pg_isready fails.
            retcode, _, _ = local[f"{self.pgbin_dpath}/pg_isready"][
                "--host",
                "localhost",
                "--port",
                str(self.pgport),
                "--dbname",
                DBGYM_POSTGRES_DBNAME,
            ].run(retcode=None)

            exists = (Path(self.pgdata_dpath) / "postmaster.pid").exists()
            if not exists and retcode != 0:
                break

    @time_record("start")
    def start_with_changes(
        self,
        conf_changes: Optional[list[str]] = None,
        dump_page_cache: bool = False,
        save_checkpoint: bool = False,
    ) -> bool:
        '''
        This function assumes that some snapshot has already been untarred into self.pgdata_dpath
        '''
        # Install the new configuration changes.
        if conf_changes is not None:
            conf_changes.append("shared_preload_libraries='pg_hint_plan'")
            with open(f"{self.pgdata_dpath}/postgresql.auto.conf", "w") as f:
                f.write("\n".join(conf_changes))

        # Start postgres instance.
        self.shutdown_postgres()
        self.move_log()

        if save_checkpoint:
            local["tar"][
                "cf",
                # We append .tmp so that if we fail in the *middle* of running tar, we
                #   still have the previous checkpoint available to us
                f"{self.checkpoint_pgdata_snapshot_fpath}.tmp",
                "-C",
                parent_dir(self.pgdata_dpath),
                self.pgdata_dpath,
            ].run()

        # Make sure the PID lock file doesn't exist.
        pid_lock = Path(f"{self.pgdata_dpath}/postmaster.pid")
        assert not pid_lock.exists()

        if dump_page_cache:
            # Dump the OS page cache.
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        attempts = 0
        while not pid_lock.exists():
            # Try starting up.
            retcode, stdout, stderr = local[f"{self.pgbin_dpath}/pg_ctl"][
                "-D",
                self.pgdata_dpath,
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

            retcode, _, _ = local[f"{self.pgbin_dpath}/pg_isready"][
                "--host",
                "localhost",
                "--port",
                str(self.pgport),
                "--dbname",
                DBGYM_POSTGRES_DBNAME,
            ].run(retcode=None)
            if retcode == 0:
                break

            time.sleep(1)
            num_cycles += 1
            self.logger.get_logger(__name__).debug(
                "Waiting for postgres to bootup but it is not..."
            )

        # Move the temporary over since we know the temporary can load.
        if save_checkpoint:
            shutil.move(f"{self.pgdata_dpath}.tgz.tmp", f"{self.pgdata_dpath}.tgz")

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
            timer = threading.Timer(300.0, cancel_fn, args=(self._get_connstr(),))
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
    
    def restore_pristine_snapshot(self):
        self._restore_snapshot(self.pristine_pgdata_snapshot_fpath)

    def restore_checkpointed_snapshot(self):
        self._restore_snapshot(self.checkpoint_pgdata_snapshot_fpath)

    @time_record("restore")
    def _restore_snapshot(
        self, pgdata_snapshot_path: Path,
    ) -> bool:
        self.shutdown_postgres()

        local["rm"]["-rf", self.pgdata_dpath].run()
        local["mkdir"]["-m", "0700", "-p", self.pgdata_dpath].run()

        # Strip the "pgdata" so we can implant directly into the target pgdata_dpath.
        assert pgdata_snapshot_path.exists()
        local["tar"][
            "xf", pgdata_snapshot_path, "-C", self.pgdata_dpath, "--strip-components", "1"
        ].run()
        # Imprint the required port.
        (
            (local["echo"][f"port={self.pgport}"])
            >> f"{self.pgdata_dpath}/postgresql.conf"
        )()

        return self.start_with_changes(conf_changes=None)
