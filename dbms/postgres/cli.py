import logging
from pathlib import Path

import click

from misc.utils import DBGymConfig
from util.shell import subprocess_run

dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)


@click.group(name="postgres")
@click.pass_obj
def postgres_group(config: DBGymConfig):
    config.append_group("postgres")


@postgres_group.command(name="clone")
@click.pass_obj
def postgres_clone(config: DBGymConfig):
    clone(config)


@postgres_group.command(name="init-pgdata")
@click.option("--remove-existing", is_flag=True)
@click.pass_obj
def postgres_init_pgdata(config: DBGymConfig, remove_existing: bool):
    init_pgdata(config, remove_existing)


@postgres_group.command(name="init-auth")
@click.pass_obj
def postgres_init_auth(config: DBGymConfig):
    init_auth(config)


@postgres_group.command(name="init-db")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_init_db(config: DBGymConfig, dbname: str):
    init_db(config, dbname)


@postgres_group.command(name="start")
@click.option("--restart-if-running/--no-restart-if-running", default=True)
@click.pass_obj
def postgres_start(config: DBGymConfig, restart_if_running: bool):
    start(config, restart_if_running)


@postgres_group.command(name="stop")
@click.pass_obj
def postgres_stop(config: DBGymConfig):
    stop(config)


@postgres_group.command(name="pgctl")
@click.argument("pgctl-str", type=str)
@click.pass_obj
def postgres_pgctl(config: DBGymConfig, pgctl_str: str):
    pgctl(config, pgctl_str)


@postgres_group.command(name="run-sql-file")
@click.argument("sql-path", type=Path)
@click.pass_obj
def postgres_run_sql(config: DBGymConfig, sql_path: str):
    run_sql_file(config, sql_path)


@postgres_group.command(name="print-psql")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_print_psql(config: DBGymConfig, dbname: str):
    psql_path = config.cur_symlinks_bin_path() / "psql"
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    dbms_postgres_logger.info(
        f"print-psql: PGPASSWORD={pgpass} {psql_path} -U {pguser} -h localhost -p {pgport} -d {dbname}"
    )


@postgres_group.command(name="print-connstr")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_print_connstr(config: DBGymConfig, dbname: str):
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    dbms_postgres_logger.info(
        f"print-connstr: postgresql+psycopg://{pguser}:{pgpass}@localhost:{pgport}/{dbname}"
    )


def _pgbin_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("repo", "boot", "build", "postgres", "bin")


def clone(config: DBGymConfig):
    symlink_dir = config.cur_symlinks_build_path("repo")
    if symlink_dir.exists():
        dbms_postgres_logger.info(f"Skipping clone: {symlink_dir}")
        return

    dbms_postgres_logger.info(f"Cloning: {symlink_dir}")
    real_dir = config.cur_task_runs_build_path("repo", mkdir=True)
    subprocess_run(f"./postgres_setup.sh {real_dir}", cwd=config.cur_source_path())
    subprocess_run(f"ln -s {real_dir} {config.cur_symlinks_build_path(mkdir=True)}")
    dbms_postgres_logger.info(f"Cloned: {symlink_dir}")


def init_pgdata(config: DBGymConfig, remove_existing: bool):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    if not remove_existing and (pgbin_path / "pgdata").exists():
        raise RuntimeError("pgdata already exists. Specify --remove-existing to force.")
    subprocess_run(f"rm -rf ./pgdata", cwd=pgbin_path)
    subprocess_run(f"./initdb -D ./pgdata", cwd=pgbin_path)


def init_auth(config: DBGymConfig):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=pgbin_path,
    )


def init_db(config: DBGymConfig, dbname: str):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    pguser = config.cur_yaml["user"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create database {dbname} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )


def run_sql_file(config: DBGymConfig, sql_path: str):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    sql_path = Path(sql_path).resolve().absolute()

    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -f {sql_path} postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )


def start(config: DBGymConfig, restart_if_running: bool = True):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    port = config.cur_yaml["port"]

    if restart_if_running:
        pg_isready = subprocess_run(
            f"./pg_isready -p {port} -U postgres",
            cwd=pgbin_path,
            check_returncode=False,
        )
        # From the manpage, pg_isready returns:
        # 0 to the shell if the server is accepting connections normally,
        # 1 if the server is rejecting connections (for example during startup),
        # 2 if there was no response to the connection attempt, and
        # 3 if no attempt was made (for example due to invalid parameters).
        dbms_postgres_logger.info(f"pg_isready status: {pg_isready.returncode}")
        if pg_isready.returncode != 2:
            dbms_postgres_logger.info(f"PostgreSQL is alive, stopping it.")
            subprocess_run("./pg_ctl -D ./pgdata stop", cwd=pgbin_path)
            dbms_postgres_logger.info(f"PostgreSQL stopped.")

    subprocess_run(
        f"./pg_ctl -D ./pgdata -l logfile -o '-p {port}' start", cwd=pgbin_path
    )


def stop(config: DBGymConfig):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    subprocess_run("./pg_ctl -D ./pgdata stop", cwd=pgbin_path)


def pgctl(config: DBGymConfig, pgctl_str: str):
    pgbin_path = _pgbin_path(config)
    assert pgbin_path.exists()
    subprocess_run(f"./pg_ctl -D ./pgdata {pgctl_str}", cwd=pgbin_path)
