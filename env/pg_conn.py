"""
At a high level, this file's goal is to provide helpers to manage a Postgres instance during
agent tuning.

On the other hand, the goal of dbms.postgres.cli is to (1) install+build postgres and (2)
create dbdata.

util.pg provides helpers used by *both* of the above files (as well as other files).
"""

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import psutil
import psycopg
import yaml
from plumbum import local
from psycopg.errors import ProgramLimitExceeded, QueryCanceled

from util.log import DBGYM_LOGGER_NAME
from util.pg import DBGYM_POSTGRES_DBNAME, SHARED_PRELOAD_LIBRARIES, get_kv_connstr
from util.workspace import DBGymConfig, open_and_save, parent_dpath_of_path

DEFAULT_CONNECT_TIMEOUT = 300


class PostgresConn:
    # The reason that PostgresConn takes in all these paths (e.g. `pgbin_path`) is so that
    # it's fully decoupled from how the files are organized in the workspace.
    def __init__(
        self,
        dbgym_cfg: DBGymConfig,
        pgport: int,
        pristine_dbdata_snapshot_fpath: Path,
        dbdata_parent_dpath: Path,
        pgbin_path: Union[str, Path],
        enable_boot: bool,
        boot_config_fpath: Path,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:

        self.dbgym_cfg = dbgym_cfg
        self.pgport = pgport
        self.pgbin_path = pgbin_path
        self.connect_timeout = connect_timeout
        self.enable_boot = enable_boot
        self.boot_config_fpath = boot_config_fpath
        self.log_step = 0

        # All the paths related to dbdata
        # pristine_dbdata_snapshot_fpath is the .tgz snapshot that represents the starting state
        #   of the database (with the default configuration). It is generated by a call to
        #   `python tune.py dbms postgres ...` and should not be overwritten.
        self.pristine_dbdata_snapshot_fpath = pristine_dbdata_snapshot_fpath
        # checkpoint_dbdata_snapshot_fpath is the .tgz snapshot that represents the current
        #   state of the database as it is being tuned. It is generated while tuning and is
        #   discarded once tuning is completed.
        self.checkpoint_dbdata_snapshot_fpath = (
            dbgym_cfg.dbgym_tmp_path / "checkpoint_dbdata.tgz"
        )
        # dbdata_parent_dpath is the parent directory of the dbdata that is *actively being tuned*.
        #   Setting this lets us control the hardware device dbdata is built on (e.g. HDD vs. SSD).
        self.dbdata_parent_dpath = dbdata_parent_dpath
        # dbdata_dpath is the dbdata that is *actively being tuned*
        self.dbdata_dpath = self.dbdata_parent_dpath / f"dbdata{self.pgport}"

        self._conn: Optional[psycopg.Connection[Any]] = None

    def get_kv_connstr(self) -> str:
        return get_kv_connstr(self.pgport)

    def conn(self) -> psycopg.Connection[Any]:
        if self._conn is None:
            self._conn = psycopg.connect(
                self.get_kv_connstr(), autocommit=True, prepare_threshold=None
            )
        return self._conn

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def move_log(self) -> None:
        pglog_fpath = (
            self.dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)
            / f"pg{self.pgport}.log"
        )
        pglog_this_step_fpath = (
            self.dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)
            / f"pg{self.pgport}.log.{self.log_step}"
        )
        if pglog_fpath.exists():
            shutil.move(pglog_fpath, pglog_this_step_fpath)
            self.log_step += 1

    def shutdown_postgres(self) -> None:
        """Shuts down postgres."""
        self.disconnect()
        if not Path(self.dbdata_dpath).exists():
            return

        while True:
            logging.getLogger(DBGYM_LOGGER_NAME).debug("Shutting down postgres...")
            _, stdout, stderr = local[f"{self.pgbin_path}/pg_ctl"][
                "stop", "--wait", "-t", "180", "-D", self.dbdata_dpath
            ].run(retcode=None)
            time.sleep(1)
            logging.getLogger(DBGYM_LOGGER_NAME).debug(
                "Stop message: (%s, %s)", stdout, stderr
            )

            # Wait until pg_isready fails.
            retcode, _, _ = local[f"{self.pgbin_path}/pg_isready"][
                "--host",
                "localhost",
                "--port",
                str(self.pgport),
                "--dbname",
                DBGYM_POSTGRES_DBNAME,
            ].run(retcode=None)

            exists = (Path(self.dbdata_dpath) / "postmaster.pid").exists()
            if not exists and retcode != 0:
                break

    def restart_postgres(self) -> bool:
        return self.restart_with_changes(None)

    def restart_with_changes(
        self,
        conf_changes: Optional[list[tuple[str, str]]],
        dump_page_cache: bool = False,
        save_checkpoint: bool = False,
    ) -> bool:
        """
        This function is called "(re)start" because it also shuts down Postgres before starting it.
        This function assumes that some snapshot has already been untarred into self.dbdata_dpath.
        You can do this by calling one of the wrappers around _restore_snapshot().

        Note that multiple calls are not "additive". Calling this will restart from the latest saved
        snapshot. If you want it to be additive without the overhead of saving a snapshot, pass in
        multiple changes to `conf_changes`.
        """
        # Install the new configuration changes.
        if conf_changes is not None:
            if SHARED_PRELOAD_LIBRARIES:
                # Using single quotes around SHARED_PRELOAD_LIBRARIES works for both single or multiple libraries.
                conf_changes.append(
                    ("shared_preload_libraries", f"'{SHARED_PRELOAD_LIBRARIES}'")
                )
            dbdata_auto_conf_path = self.dbdata_dpath / "postgresql.auto.conf"
            with open(dbdata_auto_conf_path, "w") as f:
                f.write("\n".join([f"{knob} = {val}" for knob, val in conf_changes]))

        # Start postgres instance.
        self.shutdown_postgres()
        self.move_log()

        if save_checkpoint:
            local["tar"][
                "cf",
                # We append .tmp so that if we fail in the *middle* of running tar, we
                #   still have the previous checkpoint available to us
                f"{self.checkpoint_dbdata_snapshot_fpath}.tmp",
                "-C",
                parent_dpath_of_path(self.dbdata_dpath),
                self.dbdata_dpath,
            ].run()

        # Make sure the PID lock file doesn't exist.
        pid_lock = Path(f"{self.dbdata_dpath}/postmaster.pid")
        assert not pid_lock.exists()

        if dump_page_cache:
            # Dump the OS page cache.
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        attempts = 0
        while not pid_lock.exists():
            # Try starting up.
            retcode, stdout, stderr = local[f"{self.pgbin_path}/pg_ctl"][
                "-D",
                self.dbdata_dpath,
                "--wait",
                "-t",
                "180",
                "-l",
                # We log to pg{self.pgport}.log instead of pg.log so that different PostgresConn objects
                #   don't all try to write to the same file.
                self.dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)
                / f"pg{self.pgport}.log",
                "start",
            ].run(retcode=None)

            if retcode == 0 or pid_lock.exists():
                break

            logging.getLogger(DBGYM_LOGGER_NAME).warning(
                "startup encountered: (%s, %s)", stdout, stderr
            )
            attempts += 1
            if attempts >= 5:
                logging.getLogger(DBGYM_LOGGER_NAME).error(
                    "Number of attempts to start postgres has exceeded limit."
                )
                assert False, "Could not start postgres."

        # Wait until postgres is ready to accept connections.
        num_cycles = 0
        while True:
            if self.connect_timeout is not None and num_cycles >= self.connect_timeout:
                # In this case, we've failed to start postgres.
                logging.getLogger(DBGYM_LOGGER_NAME).error(
                    "Failed to start postgres before timeout..."
                )
                return False

            retcode, _, _ = local[f"{self.pgbin_path}/pg_isready"][
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
            logging.getLogger(DBGYM_LOGGER_NAME).debug(
                "Waiting for postgres to bootup but it is not..."
            )

        # Set up Boot if we're told to do so
        if self.enable_boot:
            # I'm choosing to only load the file if enable_boot is on, so we
            # don't crash if enable_boot is off and the file doesn't exist.
            with open_and_save(self.dbgym_cfg, self.boot_config_fpath) as f:
                boot_config = yaml.safe_load(f)

            self._set_up_boot(
                boot_config["intelligent_cache"],
                boot_config["early_stop"],
                boot_config["seq_sample"],
                boot_config["seq_sample_pct"],
                boot_config["seq_sample_seed"],
                boot_config["mu_hyp_opt"],
                boot_config["mu_hyp_time"],
                boot_config["mu_hyp_stdev"],
            )

        # Move the temporary over since we now know the temporary can load.
        if save_checkpoint:
            shutil.move(f"{self.dbdata_dpath}.tgz.tmp", f"{self.dbdata_dpath}.tgz")

        return True

    def _set_up_boot(
        self,
        intelligent_cache: bool,
        early_stop: bool,
        seq_sample: bool,
        seq_sample_pct: int,
        seq_sample_seed: int,
        mu_hyp_opt: float,
        mu_hyp_time: int,
        mu_hyp_stdev: float,
    ) -> None:
        """
        Sets up Boot on the currently running Postgres instances.
        Uses instance vars of PostgresConn for configuration.
        I chose to not encode any "default values" in this function. This is so that all values
            are explicitly included in the config file. This way, we can know what Boot config
            was used in a given experiment by looking only at the config file. If we did encode
            "default values" in the function, we would need to know the state of the code at the
            time of the experiment, which is very difficult in the general case.
        """
        # If any of these commands fail, they'll throw a Python exception
        # Thus, if none of them throw an exception, we know they passed
        logging.getLogger(DBGYM_LOGGER_NAME).debug("Setting up boot")
        self.conn().execute("DROP EXTENSION IF EXISTS boot")
        self.conn().execute("CREATE EXTENSION IF NOT EXISTS boot")
        self.conn().execute("SELECT boot_connect()")
        self.conn().execute("SELECT boot_cache_clear()")
        self.conn().execute("SET boot.enable=true")
        self.conn().execute("SET boot.intercept_explain_analyze=true")
        self.conn().execute(f"SET boot.intelligent_cache={intelligent_cache}")
        self.conn().execute(f"SET boot.early_stop={early_stop}")
        self.conn().execute(f"SET boot.seq_sample={seq_sample}")
        self.conn().execute(f"SET boot.seq_sample_pct={seq_sample_pct}")
        self.conn().execute(f"SET boot.seq_sample_seed={seq_sample_seed}")
        self.conn().execute(f"SET boot.mu_hyp_opt={mu_hyp_opt}")
        self.conn().execute(f"SET boot.mu_hyp_time={mu_hyp_time}")
        self.conn().execute(f"SET boot.mu_hyp_stdev={mu_hyp_stdev}")
        logging.getLogger(DBGYM_LOGGER_NAME).debug("Set up boot")

    def psql(self, sql: str) -> tuple[int, Optional[str]]:
        """
        Execute a SQL command (equivalent to psql -C "[cmd]") and return a status code and its stderr.

        This is meant for commands that modify the database, not those that get information from the database, which
        is why it doesn't return a Cursor with the result. I designed it this way because it's difficult to provide
        a general-purpose API which returns results for arbitrary SQL queries as those results could be very large.

        A return code of 0 means success while a non-zero return code means failure. The stderr will be None if success
        and a string if failure.
        """

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
                logging.getLogger(DBGYM_LOGGER_NAME).info(f"Killing process {row[0]}")
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
            timer = threading.Timer(300.0, cancel_fn, args=(self.get_kv_connstr(),))
            timer.start()

            conn.execute(sql)
            timer.cancel()
        except ProgramLimitExceeded as e:
            timer.cancel()
            self.disconnect()
            logging.getLogger(DBGYM_LOGGER_NAME).debug(f"Action error: {e}")
            return -1, str(e)
        except QueryCanceled as e:
            timer.cancel()
            self.disconnect()
            logging.getLogger(DBGYM_LOGGER_NAME).debug(f"Action error: {e}")
            return -1, f"canceling statement: {sql}."
        except psycopg.OperationalError as e:
            timer.cancel()
            self.disconnect()
            logging.getLogger(DBGYM_LOGGER_NAME).debug(f"Action error: {e}")
            return -1, f"operational error: {sql}."
        except psycopg.errors.UndefinedTable:
            timer.cancel()
            raise

        self.disconnect()
        return 0, None

    def get_system_knobs(self) -> dict[str, str]:
        """
        System knobs are those applied across the entire system. They do not include table-specific
        knobs, query-specific knobs (aka query hints), or indexes.
        """
        conn = self.conn()
        result = conn.execute("SHOW ALL").fetchall()
        knobs = {}
        for row in result:
            knobs[row[0]] = row[1]
        return knobs

    def restore_pristine_snapshot(self) -> bool:
        return self._restore_snapshot(self.pristine_dbdata_snapshot_fpath)

    def restore_checkpointed_snapshot(self) -> bool:
        return self._restore_snapshot(self.checkpoint_dbdata_snapshot_fpath)

    def _restore_snapshot(
        self,
        dbdata_snapshot_path: Path,
    ) -> bool:
        self.shutdown_postgres()

        local["rm"]["-rf", self.dbdata_dpath].run()
        local["mkdir"]["-m", "0700", "-p", self.dbdata_dpath].run()

        # Strip the "dbdata" so we can implant directly into the target dbdata_dpath.
        assert dbdata_snapshot_path.exists()
        local["tar"][
            "xf",
            dbdata_snapshot_path,
            "-C",
            self.dbdata_dpath,
            "--strip-components",
            "1",
        ].run()
        # Imprint the required port.
        (
            (local["echo"][f"port={self.pgport}"])
            >> f"{self.dbdata_dpath}/postgresql.conf"
        )()

        return self.restart_with_changes(conf_changes=None)
