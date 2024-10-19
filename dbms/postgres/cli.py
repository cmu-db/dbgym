"""
At a high level, this file's goal is to (1) build postgres and (2) create dbdata (aka pgdata).
On the other hand, the goal of tune.protox.env.util.postgres is to provide helpers to manage
    a Postgres instance during agent tuning.
util.pg provides helpers used by *both* of the above files (as well as other files).
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click
import sqlalchemy

from benchmark.tpch.load_info import TpchLoadInfo
from dbms.load_info_base_class import LoadInfoBaseClass
from misc.utils import (
    WORKSPACE_PATH_PLACEHOLDER,
    DBGymConfig,
    conv_inputpath_to_realabspath,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    get_dbdata_tgz_name,
    is_ssd,
    link_result,
    open_and_save,
    save_file,
)
from util.pg import (
    DBGYM_POSTGRES_DBNAME,
    DBGYM_POSTGRES_PASS,
    DBGYM_POSTGRES_USER,
    DEFAULT_POSTGRES_DBNAME,
    DEFAULT_POSTGRES_PORT,
    SHARED_PRELOAD_LIBRARIES,
    create_sqlalchemy_conn,
    sql_file_execute,
    sqlalchemy_conn_execute,
)
from util.shell import subprocess_run


@click.group(name="postgres")
@click.pass_obj
def postgres_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("postgres")


@postgres_group.command(
    name="build",
    help="Download and build the Postgres repository and all necessary extensions/shared libraries. Does not create dbdata.",
)
@click.pass_obj
@click.option(
    "--rebuild",
    is_flag=True,
    help="Include this flag to rebuild Postgres even if it already exists.",
)
def postgres_build(dbgym_cfg: DBGymConfig, rebuild: bool) -> None:
    _build_repo(dbgym_cfg, rebuild)


@postgres_group.command(
    name="dbdata",
    help="Build a .tgz file of dbdata with various specifications for its contents.",
)
@click.pass_obj
@click.argument("benchmark_name", type=str)
@click.option("--scale-factor", type=float, default=1)
@click.option(
    "--pgbin-path",
    type=Path,
    default=None,
    help=f"The path to the bin containing Postgres executables. The default is {default_pgbin_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--intended-dbdata-hardware",
    type=click.Choice(["hdd", "ssd"]),
    default="hdd",
    help=f"The intended hardware dbdata should be on. Used as a sanity check for --dbdata-parent-dpath.",
)
@click.option(
    "--dbdata-parent-dpath",
    default=None,
    type=Path,
    help=f"The path to the parent directory of the dbdata which will be actively tuned. The default is {default_dbdata_parent_dpath(WORKSPACE_PATH_PLACEHOLDER)}.",
)
def postgres_dbdata(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    scale_factor: float,
    pgbin_path: Optional[Path],
    intended_dbdata_hardware: str,
    dbdata_parent_dpath: Optional[Path],
) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if pgbin_path is None:
        pgbin_path = default_pgbin_path(dbgym_cfg.dbgym_workspace_path)
    if dbdata_parent_dpath is None:
        dbdata_parent_dpath = default_dbdata_parent_dpath(
            dbgym_cfg.dbgym_workspace_path
        )

    # Convert all input paths to absolute paths
    pgbin_path = conv_inputpath_to_realabspath(dbgym_cfg, pgbin_path)
    dbdata_parent_dpath = conv_inputpath_to_realabspath(dbgym_cfg, dbdata_parent_dpath)

    # Check assertions on args
    if intended_dbdata_hardware == "hdd":
        assert not is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is HDD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an SSD"
    elif intended_dbdata_hardware == "ssd":
        assert is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is SSD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an HDD"
    else:
        assert False

    # Create dbdata
    _create_dbdata(
        dbgym_cfg, benchmark_name, scale_factor, pgbin_path, dbdata_parent_dpath
    )


def _get_pgbin_symlink_path(dbgym_cfg: DBGymConfig) -> Path:
    return dbgym_cfg.cur_symlinks_build_path(
        "repo.link", "boot", "build", "postgres", "bin"
    )


def _get_repo_symlink_path(dbgym_cfg: DBGymConfig) -> Path:
    return dbgym_cfg.cur_symlinks_build_path("repo.link")


def _build_repo(dbgym_cfg: DBGymConfig, rebuild: bool) -> None:
    expected_repo_symlink_dpath = _get_repo_symlink_path(dbgym_cfg)
    if not rebuild and expected_repo_symlink_dpath.exists():
        logging.info(f"Skipping _build_repo: {expected_repo_symlink_dpath}")
        return

    logging.info(f"Setting up repo in {expected_repo_symlink_dpath}")
    repo_real_dpath = dbgym_cfg.cur_task_runs_build_path("repo", mkdir=True)
    subprocess_run(
        f"./build_repo.sh {repo_real_dpath}", cwd=dbgym_cfg.cur_source_path()
    )

    # only link at the end so that the link only ever points to a complete repo
    repo_symlink_dpath = link_result(dbgym_cfg, repo_real_dpath)
    assert expected_repo_symlink_dpath.samefile(repo_symlink_dpath)
    logging.info(f"Set up repo in {expected_repo_symlink_dpath}")


def _create_dbdata(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    scale_factor: float,
    pgbin_path: Path,
    dbdata_parent_dpath: Path,
) -> None:
    """
    I chose *not* for this function to skip by default if dbdata_tgz_symlink_path already exists. This
      is because, while the generated data is deterministic given benchmark_name and scale_factor, any
      change in the _create_dbdata() function would result in a different dbdata. Since _create_dbdata()
      may change somewhat frequently, I decided to get rid of the footgun of having changes to
      _create_dbdata() not propagate to [dbdata].tgz by default.
    """

    # It's ok for the dbdata/ directory to be temporary. It just matters that the .tgz is saved in a safe place.
    dbdata_dpath = dbdata_parent_dpath / "dbdata_being_created"
    # We might be reusing the same dbdata_parent_dpath, so delete dbdata_dpath if it already exists
    if dbdata_dpath.exists():
        shutil.rmtree(dbdata_dpath)

    # Call initdb.
    # Save any script we call from pgbin_symlink_dpath because they are dependencies generated from another task run.
    save_file(dbgym_cfg, pgbin_path / "initdb")
    subprocess_run(f'./initdb -D "{dbdata_dpath}"', cwd=pgbin_path)

    # Start Postgres (all other dbdata setup requires postgres to be started).
    # Note that subprocess_run() never returns when running "pg_ctl start", so I'm using subprocess.run() instead.
    start_postgres(dbgym_cfg, pgbin_path, dbdata_dpath)

    # Set up Postgres.
    _generic_dbdata_setup(dbgym_cfg)
    _load_benchmark_into_dbdata(dbgym_cfg, benchmark_name, scale_factor)

    # Stop Postgres so that we don't "leak" processes.
    stop_postgres(dbgym_cfg, pgbin_path, dbdata_dpath)

    # Create .tgz file.
    # Note that you can't pass "[dbdata].tgz" as an arg to cur_task_runs_data_path() because that would create "[dbdata].tgz" as a dir.
    dbdata_tgz_real_fpath = dbgym_cfg.cur_task_runs_data_path(
        mkdir=True
    ) / get_dbdata_tgz_name(benchmark_name, scale_factor)
    # We need to cd into dbdata_dpath so that the tar file does not contain folders for the whole path of dbdata_dpath.
    subprocess_run(f"tar -czf {dbdata_tgz_real_fpath} .", cwd=dbdata_dpath)

    # Create symlink.
    # Only link at the end so that the link only ever points to a complete dbdata.
    dbdata_tgz_symlink_path = link_result(dbgym_cfg, dbdata_tgz_real_fpath)
    logging.info(f"Created dbdata in {dbdata_tgz_symlink_path}")


def _generic_dbdata_setup(dbgym_cfg: DBGymConfig) -> None:
    # get necessary vars
    pgbin_real_dpath = _get_pgbin_symlink_path(dbgym_cfg).resolve()
    assert pgbin_real_dpath.exists()
    dbgym_pguser = DBGYM_POSTGRES_USER
    dbgym_pgpass = DBGYM_POSTGRES_PASS
    pgport = DEFAULT_POSTGRES_PORT

    # Create user
    save_file(dbgym_cfg, pgbin_real_dpath / "psql")
    subprocess_run(
        f"./psql -c \"create user {dbgym_pguser} with superuser password '{dbgym_pgpass}'\" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost",
        cwd=pgbin_real_dpath,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {dbgym_pguser}" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost',
        cwd=pgbin_real_dpath,
    )

    # Load shared preload libraries
    if SHARED_PRELOAD_LIBRARIES:
        subprocess_run(
            # You have to use TO and you can't put single quotes around the libraries (https://postgrespro.com/list/thread-id/2580120)
            # The method I wrote here works for both one library and multiple libraries
            f'./psql -c "ALTER SYSTEM SET shared_preload_libraries TO {SHARED_PRELOAD_LIBRARIES};" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost',
            cwd=pgbin_real_dpath,
        )

    # Create the dbgym database. Since one dbdata dir maps to one benchmark, all benchmarks will use the same database
    # as opposed to using databases named after the benchmark.
    subprocess_run(
        f"./psql -c \"create database {DBGYM_POSTGRES_DBNAME} with owner = '{dbgym_pguser}'\" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost",
        cwd=pgbin_real_dpath,
    )


def _load_benchmark_into_dbdata(
    dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float
) -> None:
    with create_sqlalchemy_conn() as conn:
        if benchmark_name == "tpch":
            load_info = TpchLoadInfo(dbgym_cfg, scale_factor)
        else:
            raise AssertionError(
                f"_load_benchmark_into_dbdata(): the benchmark of name {benchmark_name} is not implemented"
            )

        _load_into_dbdata(dbgym_cfg, conn, load_info)


def _load_into_dbdata(
    dbgym_cfg: DBGymConfig, conn: sqlalchemy.Connection, load_info: LoadInfoBaseClass
) -> None:
    sql_file_execute(dbgym_cfg, conn, load_info.get_schema_fpath())

    # truncate all tables first before even loading a single one
    for table, _ in load_info.get_tables_and_fpaths():
        sqlalchemy_conn_execute(conn, f"TRUNCATE {table} CASCADE")
    # then, load the tables
    for table, table_fpath in load_info.get_tables_and_fpaths():
        with open_and_save(dbgym_cfg, table_fpath, "r") as table_csv:
            assert conn.connection.dbapi_connection is not None
            cur = conn.connection.dbapi_connection.cursor()
            try:
                with cur.copy(f"COPY {table} FROM STDIN CSV DELIMITER '|'") as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)
            finally:
                cur.close()

    constraints_fpath = load_info.get_constraints_fpath()
    if constraints_fpath is not None:
        sql_file_execute(dbgym_cfg, conn, constraints_fpath)


def start_postgres(
    dbgym_cfg: DBGymConfig, pgbin_path: Path, dbdata_dpath: Path
) -> None:
    _start_or_stop_postgres(dbgym_cfg, pgbin_path, dbdata_dpath, True)


def stop_postgres(dbgym_cfg: DBGymConfig, pgbin_path: Path, dbdata_dpath: Path) -> None:
    _start_or_stop_postgres(dbgym_cfg, pgbin_path, dbdata_dpath, False)


def _start_or_stop_postgres(
    dbgym_cfg: DBGymConfig, pgbin_path: Path, dbdata_dpath: Path, is_start: bool
) -> None:
    # They should be absolute paths and should exist
    assert pgbin_path.is_absolute() and pgbin_path.exists()
    assert dbdata_dpath.is_absolute() and dbdata_dpath.exists()
    # The inputs may be symlinks so we need to resolve them first
    pgbin_real_dpath = pgbin_path.resolve()
    dbdata_dpath = dbdata_dpath.resolve()
    pgport = DEFAULT_POSTGRES_PORT
    save_file(dbgym_cfg, pgbin_real_dpath / "pg_ctl")

    if is_start:
        # We use subprocess.run() because subprocess_run() never returns when running "pg_ctl start".
        # The reason subprocess_run() never returns is because pg_ctl spawns a postgres process so .poll() always returns None.
        # On the other hand, subprocess.run() does return normally, like calling `./pg_ctl` on the command line would do.
        result = subprocess.run(
            f"./pg_ctl -D \"{dbdata_dpath}\" -o '-p {pgport}' start",
            cwd=pgbin_real_dpath,
            shell=True,
        )
        result.check_returncode()
    else:
        subprocess_run(
            f"./pg_ctl -D \"{dbdata_dpath}\" -o '-p {pgport}' stop",
            cwd=pgbin_real_dpath,
        )
