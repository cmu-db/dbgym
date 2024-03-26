'''
At a high level, this file's goal is to (1) install+build postgres and (2) create pgdata.
On the other hand, the goal of tune.protox.env.util.postgres is to provide helpers to manage
    a Postgres instance during agent tuning.
util.pg provides helpers used by *both* of the above files (as well as other files).
'''
import logging
import os
import subprocess
from pathlib import Path
import click
import shutil

from benchmark.tpch.load_info import TpchLoadInfo
from dbms.load_info_base_class import LoadInfoBaseClass
from misc.utils import DBGymConfig, save_file, get_pgdata_tgz_name
from util.shell import subprocess_run
from sqlalchemy import Connection
from util.pg import conn_execute, sql_file_execute, DBGYM_POSTGRES_DBNAME, create_conn, DEFAULT_POSTGRES_PORT, DBGYM_POSTGRES_USER, DBGYM_POSTGRES_PASS


dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)


@click.group(name="postgres")
@click.pass_obj
def postgres_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("postgres")


@postgres_group.command(
    name="build",
    help="Download and build the Postgres repository and all necessary extensions/shared libraries. Does not create pgdata.",
)
@click.pass_obj
def postgres_build(dbgym_cfg: DBGymConfig):
    _build_repo(dbgym_cfg)


@postgres_group.command(
    name="pgdata",
    help="Build a .tgz file of pgdata with various specifications for its contents.",
)
@click.pass_obj
@click.argument("benchmark_name", type=str)
@click.option("--scale-factor", type=float, default=1)
def postgres_pgdata(dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float):
    _create_pgdata(dbgym_cfg, benchmark_name, scale_factor)


def _get_pgbin_symlink_path(dbgym_cfg: DBGymConfig) -> Path:
    return dbgym_cfg.cur_symlinks_build_path("repo", "boot", "build", "postgres", "bin")


def _get_repo_symlink_path(dbgym_cfg: DBGymConfig) -> Path:
    return dbgym_cfg.cur_symlinks_build_path("repo")


def _get_pgdata_tgz_symlink_path(
    dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float
) -> Path:
    # you can't pass "[pgdata].tgz" as an arg to cur_task_runs_data_path() because that would create "[pgdata].tgz" as a dir
    return dbgym_cfg.cur_symlinks_data_path(".", mkdir=True) / get_pgdata_tgz_name(
        benchmark_name, scale_factor
    )


def _build_repo(dbgym_cfg: DBGymConfig):
    repo_symlink_dpath = _get_repo_symlink_path(dbgym_cfg)
    if repo_symlink_dpath.exists():
        dbms_postgres_logger.info(f"Skipping _build_repo: {repo_symlink_dpath}")
        return

    dbms_postgres_logger.info(f"Setting up repo in {repo_symlink_dpath}")
    repo_real_dpath = dbgym_cfg.cur_task_runs_build_path("repo", mkdir=True)
    subprocess_run(
        f"./build_repo.sh {repo_real_dpath}", cwd=dbgym_cfg.cur_source_path()
    )

    # only link at the end so that the link only ever points to a complete repo
    subprocess_run(
        f"ln -s {repo_real_dpath} {dbgym_cfg.cur_symlinks_build_path(mkdir=True)}"
    )
    dbms_postgres_logger.info(f"Set up repo in {repo_symlink_dpath}")


def _create_pgdata(dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float):
    """
    I chose *not* for this function to skip by default if pgdata_tgz_symlink_path already exists. This
      is because, while the generated data is deterministic given benchmark_name and scale_factor, any
      change in the _create_pgdata() function would result in a different pgdata. Since _create_pgdata()
      may change somewhat frequently, I decided to get rid of the footgun of having changes to
      _create_pgdata() not propagate to [pgdata].tgz by default.
    """

    # Create a temporary dir for this pgdata
    # It's ok for the pgdata/ directory to be temporary. It just matters that the .tgz is saved in a safe place
    pgdata_dpath = dbgym_cfg.dbgym_tmp_path / "pgdata"

    # initdb
    pgbin_symlink_dpath = _get_pgbin_symlink_path(dbgym_cfg)
    assert pgbin_symlink_dpath.exists()
    # save any script we call from pgbin_symlink_dpath because they are dependencies generated from another task run
    save_file(dbgym_cfg, pgbin_symlink_dpath / "initdb")
    subprocess_run(f'./initdb -D "{pgdata_dpath}"', cwd=pgbin_symlink_dpath)

    # start postgres (all other pgdata setup requires postgres to be started)
    # note that subprocess_run() never returns when running "pg_ctl start", so I'm using subprocess.run() instead
    save_file(dbgym_cfg, pgbin_symlink_dpath / "pg_ctl")
    start_postgres(dbgym_cfg, pgbin_symlink_dpath, pgdata_dpath)

    # setup
    _generic_pgdata_setup(dbgym_cfg)
    _load_benchmark_into_pgdata(dbgym_cfg, benchmark_name, scale_factor)

    # stop postgres so that we don't "leak" processes
    stop_postgres(dbgym_cfg, pgbin_symlink_dpath, pgdata_dpath)

    # create .tgz file
    # you can't pass "[pgdata].tgz" as an arg to cur_task_runs_data_path() because that would create "[pgdata].tgz" as a dir
    pgdata_tgz_real_fpath = dbgym_cfg.cur_task_runs_data_path(
        ".", mkdir=True
    ) / get_pgdata_tgz_name(benchmark_name, scale_factor)
    # we need to cd into pgdata_dpath so that the tar file does not contain folders for the whole path of pgdata_dpath
    subprocess_run(f"tar -czf {pgdata_tgz_real_fpath} .", cwd=pgdata_dpath)

    # create symlink
    # only link at the end so that the link only ever points to a complete pgdata
    pgdata_tgz_symlink_path = _get_pgdata_tgz_symlink_path(
        dbgym_cfg, benchmark_name, scale_factor
    )
    if pgdata_tgz_symlink_path.exists():
        os.remove(pgdata_tgz_symlink_path)
    subprocess_run(
        f"ln -s {pgdata_tgz_real_fpath} {dbgym_cfg.cur_symlinks_data_path(mkdir=True)}"
    )
    assert (
        pgdata_tgz_symlink_path.exists()
    )  # basically asserts that pgdata_tgz_symlink_path matches dbgym_cfg.cur_symlinks_data_path(mkdir=True) / "[pgdata].tgz"

    dbms_postgres_logger.info(f"Created pgdata in {pgdata_tgz_symlink_path}")


def _generic_pgdata_setup(dbgym_cfg: DBGymConfig):
    # get necessary vars
    pgbin_symlink_dpath = _get_pgbin_symlink_path(dbgym_cfg)
    assert pgbin_symlink_dpath.exists()
    pguser = DBGYM_POSTGRES_USER
    pgpass = DBGYM_POSTGRES_PASS
    pgport = DEFAULT_POSTGRES_PORT

    # create user
    save_file(dbgym_cfg, pgbin_symlink_dpath / "psql")
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_symlink_dpath,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=pgbin_symlink_dpath,
    )

    # load shared preload libraries
    shared_preload_libraries_fpath = (
        dbgym_cfg.cur_source_path() / "shared_preload_libraries.sql"
    )
    subprocess_run(
        f"./psql -f {shared_preload_libraries_fpath} postgres -p {pgport} -h localhost",
        cwd=pgbin_symlink_dpath,
    )

    # create the dbgym database. since one pgdata dir maps to one benchmark, all benchmarks will use the same database
    # as opposed to using databases named after the benchmark
    subprocess_run(
        f"./psql -c \"create database {DBGYM_POSTGRES_DBNAME} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_symlink_dpath,
    )


def _load_benchmark_into_pgdata(
    dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float
):
    with create_conn(use_psycopg=False) as conn:
        if benchmark_name == "tpch":
            load_info = TpchLoadInfo(dbgym_cfg, scale_factor)
        else:
            raise AssertionError(
                f"_load_benchmark_into_pgdata(): the benchmark of name {benchmark_name} is not implemented"
            )

        _load_into_pgdata(conn, load_info)


def _load_into_pgdata(conn: Connection, load_info: LoadInfoBaseClass):
    sql_file_execute(conn, load_info.get_schema_fpath())

    # truncate all tables first before even loading a single one
    for table, _ in load_info.get_tables_and_fpaths():
        conn_execute(conn, f"TRUNCATE {table} CASCADE")
    # then, load the tables
    for table, table_fpath in load_info.get_tables_and_fpaths():
        with open(table_fpath, "r") as table_csv:
            with conn.connection.dbapi_connection.cursor() as cur:
                with cur.copy(f"COPY {table} FROM STDIN CSV DELIMITER '|'") as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)

    constraints_fpath = load_info.get_constraints_fpath()
    if constraints_fpath != None:
        sql_file_execute(conn, constraints_fpath)


def start_postgres(dbgym_cfg: DBGymConfig, pgbin_dpath: Path, pgdata_dpath: Path) -> None:
    _start_or_stop_postgres(dbgym_cfg, pgbin_dpath, pgdata_dpath, True)


def stop_postgres(dbgym_cfg: DBGymConfig, pgbin_dpath: Path, pgdata_dpath: Path) -> None:
    _start_or_stop_postgres(dbgym_cfg, pgbin_dpath, pgdata_dpath, False)


def _start_or_stop_postgres(dbgym_cfg: DBGymConfig, pgbin_dpath: Path, pgdata_dpath: Path, is_start: bool) -> None:
    # they should be absolute paths and should exist
    assert pgbin_dpath.is_absolute() and pgbin_dpath.exists()
    assert pgdata_dpath.is_absolute() and pgdata_dpath.exists()
    # the inputs may be symlinks so we need to resolve them first
    pgbin_real_dpath = pgbin_dpath.resolve()
    pgdata_dpath = pgdata_dpath.resolve()
    pgport = DEFAULT_POSTGRES_PORT
    save_file(dbgym_cfg, pgbin_real_dpath / "pg_ctl")

    if is_start:
        # note that subprocess_run() never returns when running "pg_ctl start", so I'm using subprocess.run() instead
        subprocess.run(f"./pg_ctl -D \"{pgdata_dpath}\" -o '-p {pgport}' start", cwd=pgbin_real_dpath, shell=True)
    else:
        subprocess_run(f"./pg_ctl -D \"{pgdata_dpath}\" -o '-p {pgport}' stop", cwd=pgbin_real_dpath)