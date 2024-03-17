import logging
from pathlib import Path
import subprocess
import os
import click
from sqlalchemy import create_engine

from misc.utils import DBGymConfig, save_file
from util.shell import subprocess_run
from benchmark.tpch.cli import TPCH_SCHEMA_FNAME, TPCH_CONSTRAINTS_FNAME
from util.sql import Connection, Engine, sql_file_execute, conn_execute

dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)

DBGYM_DBNAME = "dbgym"


@click.group(name="postgres")
@click.pass_obj
def postgres_group(config: DBGymConfig):
    config.append_group("postgres")


@postgres_group.command(name="repo", help="Download and build the Postgres repository and all necessary extensions/shared libraries. Does not create pgdata.")
@click.pass_obj
def postgres_repo(config: DBGymConfig):
    _build_repo(config)


@postgres_group.command(name="pgdata", help="Build a .tgz file of pgdata with various specifications for its contents.")
@click.pass_obj
@click.argument("benchmark_name", type=str)
@click.option("--scale-factor", type=float, default=1)
def postgres_pgdata(dbgym_cfg: DBGymConfig, benchmark_name: str, scale_factor: float):
    _create_pgdata(dbgym_cfg, benchmark_name, scale_factor)


def _get_pgbin_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("repo", "boot", "build", "postgres", "bin")


def _get_repo_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("repo")


def _get_pgdata_name(benchmark_name: str, scale_factor: float) -> str:
    scale_factor_str = str(scale_factor).replace(".", "point")
    return f"{benchmark_name}_sf{scale_factor_str}_pgdata"


def _get_pgdata_tgz_name(benchmark_name: str, scale_factor: float) -> str:
    return _get_pgdata_name(benchmark_name, scale_factor) + ".tgz"


def _get_pgdata_tgz_symlink_path(config: DBGymConfig, benchmark_name: str, scale_factor: float) -> Path:
    # you can't pass "[pgdata].tgz" as an arg to cur_task_runs_data_path() because that would create "[pgdata].tgz" as a dir
    return config.cur_symlinks_data_path(".", mkdir=True) / _get_pgdata_tgz_name(benchmark_name, scale_factor)


def _build_repo(config: DBGymConfig):
    repo_symlink_dpath = _get_repo_symlink_path(config)
    if repo_symlink_dpath.exists():
        dbms_postgres_logger.info(f"Skipping _build_repo: {repo_symlink_dpath}")
        return

    dbms_postgres_logger.info(f"Setting up repo in {repo_symlink_dpath}")
    repo_real_dpath = config.cur_task_runs_build_path("repo", mkdir=True)
    subprocess_run(f"./build_repo.sh {repo_real_dpath}", cwd=config.cur_source_path())

    # only link at the end so that the link only ever points to a complete repo
    subprocess_run(f"ln -s {repo_real_dpath} {config.cur_symlinks_build_path(mkdir=True)}")
    dbms_postgres_logger.info(f"Set up repo in {repo_symlink_dpath}")


def _create_pgdata(config: DBGymConfig, benchmark_name: str, scale_factor: float):
    # create a new dir for this pgdata
    pgdata_real_dpath = config.cur_task_runs_data_path(_get_pgdata_name(benchmark_name, scale_factor), mkdir=True)

    # initdb
    pgbin_path = _get_pgbin_symlink_path(config)
    assert pgbin_path.exists()
    # save any script we call from pgbin_path because they are dependencies generated from another task run
    save_file(config, pgbin_path / "initdb")
    subprocess_run(f"./initdb -D \"{pgdata_real_dpath}\"", cwd=pgbin_path)

    # start postgres (all other pgdata setup requires postgres to be started)
    pgport = config.cur_yaml["port"]
    # note that subprocess_run() never returns when running "pg_ctl start", so I'm using subprocess.run() instead
    save_file(config, pgbin_path / "pg_ctl")
    subprocess.run(
        f"./pg_ctl -D \"{pgdata_real_dpath}\" -o '-p {pgport}' start", cwd=pgbin_path, shell=True
    )

    # setup
    _generic_pgdata_setup(config)
    _load_benchmark_into_pgdata(config, benchmark_name, scale_factor)

    # stop postgres so that we don't "leak" processes
    subprocess_run(
        f"./pg_ctl -D \"{pgdata_real_dpath}\" stop", cwd=pgbin_path
    )

    # create .tgz file
    # you can't pass "[pgdata].tgz" as an arg to cur_task_runs_data_path() because that would create "[pgdata].tgz" as a dir
    pgdata_tgz_real_fpath = config.cur_task_runs_data_path(".", mkdir=True) / _get_pgdata_tgz_name(benchmark_name, scale_factor)
    # we need to cd into pgdata_real_dpath so that the tar file does not contain folders for the whole path of pgdata_real_dpath
    subprocess_run(
        f"tar -czf {pgdata_tgz_real_fpath} .", cwd=pgdata_real_dpath
    )

    # create symlink
    # only link at the end so that the link only ever points to a complete pgdata
    pgdata_tgz_symlink_path = _get_pgdata_tgz_symlink_path(config, benchmark_name, scale_factor)
    if pgdata_tgz_symlink_path.exists():
        os.remove(pgdata_tgz_symlink_path)
    subprocess_run(f"ln -s {pgdata_tgz_real_fpath} {config.cur_symlinks_data_path(mkdir=True)}")
    assert pgdata_tgz_symlink_path.exists() # basically asserts that pgdata_tgz_symlink_path matches config.cur_symlinks_data_path(mkdir=True) / "[pgdata].tgz"
    
    dbms_postgres_logger.info(f"Created pgdata in {pgdata_tgz_symlink_path}")


def _generic_pgdata_setup(config: DBGymConfig):
    # get necessary vars
    pgbin_path = _get_pgbin_symlink_path(config)
    assert pgbin_path.exists()
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]

    # create user
    save_file(config, pgbin_path / "psql")
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=pgbin_path,
    )

    # load shared preload libraries
    shared_preload_libraries_fpath = config.cur_source_path() / "shared_preload_libraries.sql"
    subprocess_run(
        f"./psql -f {shared_preload_libraries_fpath} postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )

    # create the dbgym database. since one pgdata dir maps to one benchmark, all benchmarks will use the same database
    # as opposed to using databases named after the benchmark
    subprocess_run(
        f"./psql -c \"create database {DBGYM_DBNAME} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )


def _load_benchmark_into_pgdata(config: DBGymConfig, benchmark_name: str, scale_factor: float):
    if benchmark_name == "tpch":
        with _create_conn(config) as conn:
            _load_tpch(config, conn, scale_factor)
    else:
        raise AssertionError(f"_load_benchmark_into_pgdata(): the benchmark of name {benchmark_name} is not implemented")


def _load_tpch(config: DBGymConfig, conn: Connection, scale_factor: float):
    # *This is a break of abstraction, but that is inevitable*
    # Another way to handle this is to generate the generic pgdata.tgz with a "task.py dbms postgres" invocation
    #   and a [pgdata].tgz loaded with the benchmark data with a second "task.py benchmark tpch" invocation.
    # However, doing that would require the "task.py benchmark tpch" invocation to start and stop Postgres, which
    #   is an equivalent break of abstraction, only in reverse.
    # A break of abstraction is inevitable, but this way of breaking it is preferable because it results in only
    #   a single CLI invocation instead of two, and the # of CLI invocations is something we really want to minimize
    #   (see documentation for why we want to minimize it so much).

    # currently, hardcoding the path seems like the easiest solution. If the path ever changes, it'll
    # just break an integration test and we can fix it. I don't want to prematurely overengineer it
    codebase_path_components = ["dbgym", "benchmark", "tpch"]
    codebase_dname = "_".join(codebase_path_components)
    schema_root_dpath = config.dbgym_repo_path
    for component in codebase_path_components[1:]: # [1:] to skip "dbgym"
        schema_root_dpath /= component
    data_root_dpath = config.dbgym_symlinks_path / codebase_dname / "data"

    tables = [
        "region",
        "nation",
        "part",
        "supplier",
        "partsupp",
        "customer",
        "orders",
        "lineitem",
    ]

    schema_fpath = schema_root_dpath / TPCH_SCHEMA_FNAME
    assert schema_fpath.exists(), f"schema_fpath ({schema_fpath}) does not exist"
    sql_file_execute(conn, schema_fpath)
    for table in tables:
        conn_execute(conn, f"TRUNCATE {table} CASCADE")
    tables_dpath = data_root_dpath / f"tables_sf{scale_factor}"
    assert tables_dpath.exists(), f"tables_dpath ({tables_dpath}) does not exist. Make sure you have generated the TPC-H data"
    for table in tables:
        table_path = tables_dpath / f"{table}.tbl"

        with open(table_path, "r") as table_csv:
            with conn.connection.dbapi_connection.cursor() as cur:
                with cur.copy(f"COPY {table} FROM STDIN CSV DELIMITER '|'") as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)
    sql_file_execute(conn, schema_root_dpath / TPCH_CONSTRAINTS_FNAME)


def _create_conn(config: DBGymConfig) -> Connection:
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    connstr = f"postgresql+psycopg://{pguser}:{pgpass}@localhost:{pgport}/{DBGYM_DBNAME}"
    engine: Engine = create_engine(
        connstr,
        execution_options={"isolation_level": "AUTOCOMMIT"},
    )
    return engine.connect()