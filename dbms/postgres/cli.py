"""
At a high level, this file's goal is to (1) build postgres and (2) create dbdata (aka pgdata).
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import click
import sqlalchemy
from gymlib.infra_paths import (
    DEFAULT_SCALE_FACTOR,
    get_dbdata_tgz_symlink_path,
    get_pgbin_symlink_path,
    get_repo_symlink_path,
)
from gymlib.pg import create_sqlalchemy_conn, sql_file_execute
from gymlib.workspace import (
    WORKSPACE_PATH_PLACEHOLDER,
    DBGymWorkspace,
    fully_resolve_path,
    get_tmp_path_from_workspace_path,
    is_fully_resolved,
    is_ssd,
    linkname_to_name,
)
from sqlalchemy import text

from benchmark.job.load_info import JobLoadInfo
from benchmark.tpch.load_info import TpchLoadInfo
from dbms.load_info_base_class import LoadInfoBaseClass
from util.shell import subprocess_run

DBGYM_POSTGRES_USER = "dbgym_user"
DBGYM_POSTGRES_PASS = "dbgym_pass"
DBGYM_POSTGRES_DBNAME = "dbgym"
DEFAULT_POSTGRES_DBNAME = "postgres"
DEFAULT_POSTGRES_PORT = 5432
SHARED_PRELOAD_LIBRARIES = "boot,pg_hint_plan,pg_prewarm"


@click.group(name="postgres")
@click.pass_obj
def postgres_group(dbgym_workspace: DBGymWorkspace) -> None:
    pass


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
def postgres_build(dbgym_workspace: DBGymWorkspace, rebuild: bool) -> None:
    _postgres_build(dbgym_workspace, rebuild)


def _postgres_build(dbgym_workspace: DBGymWorkspace, rebuild: bool) -> None:
    """
    This function exists as a hook for integration tests.
    """
    expected_repo_symlink_path = get_repo_symlink_path(
        dbgym_workspace.dbgym_workspace_path
    )
    if not rebuild and expected_repo_symlink_path.exists():
        logging.info(f"Skipping _postgres_build: {expected_repo_symlink_path}")
        return

    logging.info(f"Setting up repo in {expected_repo_symlink_path}")
    repo_real_path = dbgym_workspace.dbgym_this_run_path / "repo"
    repo_real_path.mkdir(parents=False, exist_ok=False)
    subprocess_run(
        f"./_build_repo.sh {repo_real_path}",
        cwd=dbgym_workspace.base_dbgym_repo_path / "dbms" / "postgres",
    )

    # only link at the end so that the link only ever points to a complete repo
    repo_symlink_path = dbgym_workspace.link_result(repo_real_path)
    assert expected_repo_symlink_path.samefile(repo_symlink_path)
    logging.info(f"Set up repo in {expected_repo_symlink_path}")


@postgres_group.command(
    name="dbdata",
    help="Build a .tgz file of dbdata with various specifications for its contents.",
)
@click.pass_obj
@click.argument("benchmark_name", type=str)
@click.option("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
@click.option(
    "--pgbin-path",
    type=Path,
    default=None,
    help=f"The path to the bin containing Postgres executables. The default is {get_pgbin_symlink_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--intended-dbdata-hardware",
    type=click.Choice(["hdd", "ssd"]),
    default="hdd",
    help=f"The intended hardware dbdata should be on. Used as a sanity check for --dbdata-parent-path.",
)
@click.option(
    "--dbdata-parent-path",
    default=None,
    type=Path,
    help=f"The path to the parent directory of the dbdata which will be actively tuned. The default is {get_tmp_path_from_workspace_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
def postgres_dbdata(
    dbgym_workspace: DBGymWorkspace,
    benchmark_name: str,
    scale_factor: float,
    pgbin_path: Optional[Path],
    intended_dbdata_hardware: str,
    dbdata_parent_path: Optional[Path],
) -> None:
    _postgres_dbdata(
        dbgym_workspace,
        benchmark_name,
        scale_factor,
        pgbin_path,
        intended_dbdata_hardware,
        dbdata_parent_path,
    )


def _postgres_dbdata(
    dbgym_workspace: DBGymWorkspace,
    benchmark_name: str,
    scale_factor: float,
    pgbin_path: Optional[Path],
    intended_dbdata_hardware: str,
    dbdata_parent_path: Optional[Path],
) -> None:
    """
    This function exists as a hook for integration tests.
    """
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if pgbin_path is None:
        pgbin_path = get_pgbin_symlink_path(dbgym_workspace.dbgym_workspace_path)
    if dbdata_parent_path is None:
        dbdata_parent_path = get_tmp_path_from_workspace_path(
            dbgym_workspace.dbgym_workspace_path
        )

    # Fully resolve all input paths.
    pgbin_path = fully_resolve_path(pgbin_path)
    dbdata_parent_path = fully_resolve_path(dbdata_parent_path)

    # Check assertions on args
    if intended_dbdata_hardware == "hdd":
        assert not is_ssd(
            dbdata_parent_path
        ), f"Intended hardware is HDD but dbdata_parent_path ({dbdata_parent_path}) is an SSD"
    elif intended_dbdata_hardware == "ssd":
        assert is_ssd(
            dbdata_parent_path
        ), f"Intended hardware is SSD but dbdata_parent_path ({dbdata_parent_path}) is an HDD"
    else:
        assert (
            False
        ), f'Intended hardware is "{intended_dbdata_hardware}" which is invalid'

    # Create dbdata
    _create_dbdata(
        dbgym_workspace, benchmark_name, scale_factor, pgbin_path, dbdata_parent_path
    )


def _create_dbdata(
    dbgym_workspace: DBGymWorkspace,
    benchmark_name: str,
    scale_factor: float,
    pgbin_path: Path,
    dbdata_parent_path: Path,
) -> None:
    """
    If you change the code of _create_dbdata(), you should also delete the symlink so that the next time you run
    `dbms postgres dbdata` it will re-create the dbdata.
    """
    expected_dbdata_tgz_symlink_path = get_dbdata_tgz_symlink_path(
        dbgym_workspace.dbgym_workspace_path,
        benchmark_name,
        scale_factor,
    )
    if expected_dbdata_tgz_symlink_path.exists():
        logging.info(f"Skipping _create_dbdata: {expected_dbdata_tgz_symlink_path}")
        return

    # It's ok for the dbdata/ directory to be temporary. It just matters that the .tgz is saved in a safe place.
    dbdata_path = dbdata_parent_path / "dbdata_being_created"
    # We might be reusing the same dbdata_parent_path, so delete dbdata_path if it already exists
    if dbdata_path.exists():
        shutil.rmtree(dbdata_path)

    # Call initdb.
    # Save any script we call from pgbin_symlink_path because they are dependencies generated from another task run.
    dbgym_workspace.save_file(pgbin_path / "initdb")
    subprocess_run(f'./initdb -D "{dbdata_path}"', cwd=pgbin_path)

    # Start Postgres (all other dbdata setup requires postgres to be started).
    # Note that subprocess_run() never returns when running "pg_ctl start", so I'm using subprocess.run() instead.
    start_postgres(dbgym_workspace, pgbin_path, dbdata_path)

    # Set up Postgres.
    _generic_dbdata_setup(dbgym_workspace)
    _load_benchmark_into_dbdata(dbgym_workspace, benchmark_name, scale_factor)

    # Stop Postgres so that we don't "leak" processes.
    stop_postgres(dbgym_workspace, pgbin_path, dbdata_path)

    # Create .tgz file.
    dbdata_tgz_real_path = dbgym_workspace.dbgym_this_run_path / linkname_to_name(
        expected_dbdata_tgz_symlink_path.name
    )
    # We need to cd into dbdata_path so that the tar file does not contain folders for the whole path of dbdata_path.
    subprocess_run(f"tar -czf {dbdata_tgz_real_path} .", cwd=dbdata_path)

    # Create symlink.
    # Only link at the end so that the link only ever points to a complete dbdata.
    dbdata_tgz_symlink_path = dbgym_workspace.link_result(dbdata_tgz_real_path)
    assert expected_dbdata_tgz_symlink_path.samefile(dbdata_tgz_symlink_path)
    logging.info(f"Created dbdata in {dbdata_tgz_symlink_path}")


def _generic_dbdata_setup(dbgym_workspace: DBGymWorkspace) -> None:
    # get necessary vars
    pgbin_real_path = get_pgbin_symlink_path(
        dbgym_workspace.dbgym_workspace_path
    ).resolve()
    assert pgbin_real_path.exists()
    dbgym_pguser = DBGYM_POSTGRES_USER
    dbgym_pgpass = DBGYM_POSTGRES_PASS
    pgport = DEFAULT_POSTGRES_PORT

    # Create user
    dbgym_workspace.save_file(pgbin_real_path / "psql")
    subprocess_run(
        f"./psql -c \"create user {dbgym_pguser} with superuser password '{dbgym_pgpass}'\" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost",
        cwd=pgbin_real_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {dbgym_pguser}" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost',
        cwd=pgbin_real_path,
    )

    # Load shared preload libraries
    if SHARED_PRELOAD_LIBRARIES:
        subprocess_run(
            # You have to use TO and you can't put single quotes around the libraries (https://postgrespro.com/list/thread-id/2580120)
            # The method I wrote here works for both one library and multiple libraries
            f'./psql -c "ALTER SYSTEM SET shared_preload_libraries TO {SHARED_PRELOAD_LIBRARIES};" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost',
            cwd=pgbin_real_path,
        )

    # Create the dbgym database. Since one dbdata dir maps to one benchmark, all benchmarks will use the same database
    # as opposed to using databases named after the benchmark.
    subprocess_run(
        f"./psql -c \"create database {DBGYM_POSTGRES_DBNAME} with owner = '{dbgym_pguser}'\" {DEFAULT_POSTGRES_DBNAME} -p {pgport} -h localhost",
        cwd=pgbin_real_path,
    )


def _load_benchmark_into_dbdata(
    dbgym_workspace: DBGymWorkspace, benchmark_name: str, scale_factor: float
) -> None:
    load_info: LoadInfoBaseClass

    with create_sqlalchemy_conn() as conn:
        if benchmark_name == "tpch":
            load_info = TpchLoadInfo(dbgym_workspace, scale_factor)
        elif benchmark_name == "job":
            load_info = JobLoadInfo(dbgym_workspace)
        else:
            raise AssertionError(
                f"_load_benchmark_into_dbdata(): the benchmark of name {benchmark_name} is not implemented"
            )

        _load_into_dbdata(dbgym_workspace, conn, load_info)


def _load_into_dbdata(
    dbgym_workspace: DBGymWorkspace,
    conn: sqlalchemy.Connection,
    load_info: LoadInfoBaseClass,
) -> None:
    sql_file_execute(dbgym_workspace, conn, load_info.get_schema_path())

    # Truncate all tables first before even loading a single one.
    for table, _ in load_info.get_tables_and_paths():
        sqlalchemy_conn_execute(conn, f"TRUNCATE {table} CASCADE")
    # Then, load the tables.
    for table, table_path in load_info.get_tables_and_paths():
        with dbgym_workspace.open_and_save(table_path, "r") as table_csv:
            assert conn.connection.dbapi_connection is not None
            cur = conn.connection.dbapi_connection.cursor()
            try:
                with cur.copy(
                    f"COPY {table} FROM STDIN CSV DELIMITER '{load_info.get_table_file_delimiter()}' ESCAPE '\\'"
                ) as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)
            finally:
                cur.close()

    constraints_path = load_info.get_constraints_path()
    if constraints_path is not None:
        sql_file_execute(dbgym_workspace, conn, constraints_path)


# The start and stop functions slightly duplicate functionality from pg_conn.py. However, I chose to do it this way
# because what the `dbms` CLI needs in terms of starting and stopping Postgres is much simpler than what an agent
# that is tuning the database needs. Because these functions are so simple, I think it's okay to leave them here
# even though they are a little redundant. It seems better than making `dbms` depend on the behavior of the
# tuning environment.
def start_postgres(
    dbgym_workspace: DBGymWorkspace, pgbin_path: Path, dbdata_path: Path
) -> None:
    _start_or_stop_postgres(dbgym_workspace, pgbin_path, dbdata_path, True)


def stop_postgres(
    dbgym_workspace: DBGymWorkspace, pgbin_path: Path, dbdata_path: Path
) -> None:
    _start_or_stop_postgres(dbgym_workspace, pgbin_path, dbdata_path, False)


def _start_or_stop_postgres(
    dbgym_workspace: DBGymWorkspace,
    pgbin_path: Path,
    dbdata_path: Path,
    is_start: bool,
) -> None:
    # They should be absolute paths and should exist
    assert is_fully_resolved(pgbin_path)
    assert is_fully_resolved(dbdata_path)
    pgport = DEFAULT_POSTGRES_PORT
    dbgym_workspace.save_file(pgbin_path / "pg_ctl")

    if is_start:
        # We use subprocess.run() because subprocess_run() never returns when running "pg_ctl start".
        # The reason subprocess_run() never returns is because pg_ctl spawns a postgres process so .poll() always returns None.
        # On the other hand, subprocess.run() does return normally, like calling `./pg_ctl` on the command line would do.
        result = subprocess.run(
            f"./pg_ctl -D \"{dbdata_path}\" -o '-p {pgport}' start",
            cwd=pgbin_path,
            shell=True,
        )
        result.check_returncode()
    else:
        subprocess_run(
            f"./pg_ctl -D \"{dbdata_path}\" -o '-p {pgport}' stop",
            cwd=pgbin_path,
        )


def sqlalchemy_conn_execute(
    conn: sqlalchemy.Connection, sql: str
) -> sqlalchemy.engine.CursorResult[Any]:
    return conn.execute(text(sql))
