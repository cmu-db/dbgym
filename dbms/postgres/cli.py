import logging
from pathlib import Path
import subprocess

import click

from misc.utils import DBGymConfig
from util.shell import subprocess_run

dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)


BASE_PGDATA_DNAME = "base_pgdata"


@click.group(name="postgres")
@click.pass_obj
def postgres_group(config: DBGymConfig):
    config.append_group("postgres")


@postgres_group.command(name="base", help="Set up all aspects of Postgres unrelated to any specific benchmark.")
@click.pass_obj
def postgres_base(config: DBGymConfig):
    setup_repo(config)
    setup_base_pgdata(config)


@postgres_group.command(name="init-auth")
@click.pass_obj
def postgres_init_auth(config: DBGymConfig):
    init_auth(config)


@postgres_group.command(name="init-db")
@click.argument("dbname", type=str)
@click.pass_obj
def postgres_init_db(config: DBGymConfig, dbname: str):
    init_db(config, dbname)
    

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


def _get_pgbin_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("repo", "boot", "build", "postgres", "bin")


def _get_repo_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("repo")


def _get_base_pgdata_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path(BASE_PGDATA_DNAME)


def setup_repo(config: DBGymConfig):
    repo_symlink_dpath = _get_repo_symlink_path(config)
    if repo_symlink_dpath.exists():
        dbms_postgres_logger.info(f"Skipping setup_repo: {repo_symlink_dpath}")
        return

    dbms_postgres_logger.info(f"Setting up repo in {repo_symlink_dpath}")
    repo_real_dpath = config.cur_task_runs_build_path("repo", mkdir=True)
    subprocess_run(f"./setup_repo.sh {repo_real_dpath}", cwd=config.cur_source_path())

    # only link at the end so that the link only ever points to a complete repo
    subprocess_run(f"ln -s {repo_real_dpath} {config.cur_symlinks_build_path(mkdir=True)}")
    dbms_postgres_logger.info(f"Set up repo in {repo_symlink_dpath}")


def setup_base_pgdata(config: DBGymConfig):
    pgdata_symlink_dpath = _get_base_pgdata_symlink_path(config)
    if pgdata_symlink_dpath.exists():
        dbms_postgres_logger.info(f"Skipping setup_repo: {pgdata_symlink_dpath}")
        return

    # initdb
    pgbin_path = _get_pgbin_symlink_path(config)
    assert pgbin_path.exists()
    pgdata_real_dpath = config.cur_task_runs_build_path(BASE_PGDATA_DNAME, mkdir=True)
    subprocess_run(f"mkdir -p \"{pgdata_real_dpath}\"")
    subprocess_run(f"./initdb -D \"{pgdata_real_dpath}\"", cwd=pgbin_path)

    # start postgres (all other pgdata setup requires postgres to be started)
    pgport = config.cur_yaml["port"]
    # note that subprocess_run() never returns when running pg_ctl start, so I'm using subprocess.run() instead
    subprocess.run(
        f"./pg_ctl -D \"{pgdata_real_dpath}\" -o '-p {pgport}' start", cwd=pgbin_path, shell=True
    )

    # create user
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=pgbin_path,
    )

    # stop postgres so that we don't "leak" anything
    subprocess_run(
        f"./pg_ctl -D \"{pgdata_real_dpath}\" stop", cwd=pgbin_path
    )

    # only link at the end so that the link only ever points to a complete pgdata
    subprocess_run(f"ln -s {pgdata_real_dpath} {config.cur_symlinks_build_path(mkdir=True)}")


def init_auth(config: DBGymConfig):
    pgbin_path = _get_pgbin_symlink_path(config)
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
    pgbin_path = _get_pgbin_symlink_path(config)
    assert pgbin_path.exists()
    pguser = config.cur_yaml["user"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create database {dbname} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
