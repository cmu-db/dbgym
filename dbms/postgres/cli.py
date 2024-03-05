import logging
from pathlib import Path

import click

from util.shell import subprocess_run

dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)


@click.group(name="postgres")
@click.pass_obj
def postgres_group(config):
    config.append_group("postgres")


@postgres_group.command(name="clone")
@click.pass_obj
def postgres_clone(config):
    clone(config)


@postgres_group.command(name="init-pgdata")
@click.option("--remove-existing", is_flag=True)
@click.pass_obj
def postgres_init_pgdata(config, remove_existing):
    init_pgdata(config, remove_existing)


@postgres_group.command(name="init-auth")
@click.pass_obj
def postgres_init_auth(config):
    init_auth(config)


@postgres_group.command(name="init-db")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_init_db(config, dbname):
    init_db(config, dbname)


@postgres_group.command(name="start")
@click.option("--restart-if-running/--no-restart-if-running", default=True)
@click.pass_obj
def postgres_start(config, restart_if_running):
    start(config, restart_if_running)


@postgres_group.command(name="stop")
@click.pass_obj
def postgres_stop(config):
    stop(config)


@postgres_group.command(name="pgctl")
@click.argument("pgctl-str", type=str)
@click.pass_obj
def postgres_pgctl(config, pgctl_str):
    pgctl(config, pgctl_str)


@postgres_group.command(name="run-sql-file")
@click.argument("sql-path", type=str)
@click.pass_obj
def postgres_run_sql(config, sql_path):
    run_sql_file(config, sql_path)


@postgres_group.command(name="print-psql")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_print_psql(config, dbname):
    psql_path = config.cur_bin_path / "psql"
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    dbms_postgres_logger.info(
        f"print-psql: PGPASSWORD={pgpass} {psql_path} -U {pguser} -h localhost -p {pgport} -d {dbname}"
    )


@postgres_group.command(name="print-connstr")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_print_connstr(config, dbname):
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    dbms_postgres_logger.info(
        f"print-connstr: postgresql+psycopg://{pguser}:{pgpass}@localhost:{pgport}/{dbname}"
    )


def clone(config):
    if config.cur_build_path.exists():
        dbms_postgres_logger.info(f"Skipping clone: {config.cur_build_path}")
        return

    dbms_postgres_logger.info(f"Cloning: {config.cur_build_path}")
    build_path = (config.cur_build_path / "..").resolve()
    subprocess_run(f"./postgres_setup.sh {config.cur_run_path}", cwd=config.cur_path)
    build_path.mkdir(parents=True, exist_ok=True)
    subprocess_run(f"ln -s {config.cur_run_path} {build_path}")
    bin_path = config.cur_bin_path
    (bin_path / "..").resolve().mkdir(parents=True, exist_ok=True)
    pgbin_path = config.cur_run_path / "boot" / "build" / "postgres" / "bin"
    subprocess_run(f"ln -sT {pgbin_path} {bin_path}")
    dbms_postgres_logger.info(f"Cloned: {config.cur_build_path}")


def init_pgdata(config, remove_existing):
    assert config.cur_build_path.exists()
    if not remove_existing and (config.cur_bin_path / "pgdata").exists():
        raise RuntimeError("pgdata already exists. Specify --remove-existing to force.")
    subprocess_run(f"rm -rf ./pgdata", cwd=config.cur_bin_path)
    subprocess_run(f"./initdb -D ./pgdata", cwd=config.cur_bin_path)


def init_auth(config):
    assert config.cur_build_path.exists()
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=config.cur_bin_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=config.cur_bin_path,
    )


def init_db(config, dbname):
    assert config.cur_build_path.exists()
    pguser = config.cur_yaml["user"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create database {dbname} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=config.cur_bin_path,
    )


def run_sql_file(config, sql_path):
    assert config.cur_build_path.exists()
    sql_path = Path(sql_path).resolve().absolute()

    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -f {sql_path} postgres -p {pgport} -h localhost",
        cwd=config.cur_bin_path,
    )


def start(config, restart_if_running=True):
    assert config.cur_build_path.exists()
    port = config.cur_yaml["port"]

    if restart_if_running:
        pg_isready = subprocess_run(
            f"./pg_isready -p {port} -U postgres",
            cwd=config.cur_bin_path,
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
            subprocess_run("./pg_ctl -D ./pgdata stop", cwd=config.cur_bin_path)
            dbms_postgres_logger.info(f"PostgreSQL stopped.")

    subprocess_run(
        f"./pg_ctl -D ./pgdata -l logfile -o '-p {port}' start", cwd=config.cur_bin_path
    )


def stop(config):
    assert config.cur_build_path.exists()
    subprocess_run("./pg_ctl -D ./pgdata stop", cwd=config.cur_bin_path)


def pgctl(config, pgctl_str):
    assert config.cur_build_path.exists()
    subprocess_run(f"./pg_ctl -D ./pgdata {pgctl_str}", cwd=config.cur_bin_path)
