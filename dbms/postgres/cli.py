import logging
from pathlib import Path
import subprocess
import os

import click

from misc.utils import DBGymConfig, save_file
from util.shell import subprocess_run

dbms_postgres_logger = logging.getLogger("dbms/postgres")
dbms_postgres_logger.setLevel(logging.INFO)


@click.group(name="postgres")
@click.pass_obj
def postgres_group(config: DBGymConfig):
    config.append_group("postgres")


@postgres_group.command(name="repo", help="Download and build the Postgres repository and all necessary extensions/shared libraries. Does not create pgdata.")
@click.pass_obj
def postgres_repo(config: DBGymConfig):
    setup_repo(config)


@postgres_group.command(name="pgdata", help="Build a .tgz file of pgdata with various specifications for its contents.")
@click.pass_obj
def postgres_pgdata(dbgym_cfg: DBGymConfig):
    setup_pgdata(dbgym_cfg)


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


def _get_pgdata_tgz_symlink_path(config: DBGymConfig) -> Path:
    return config.cur_symlinks_build_path("pgdata.tgz")


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


def setup_pgdata(config: DBGymConfig):
    # create a new dir for this pgdata
    pgdata_real_dpath = config.cur_task_runs_build_path("pgdata", mkdir=True)
    subprocess_run(f"mkdir -p \"{pgdata_real_dpath}\"")

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

    # create user
    pguser = config.cur_yaml["user"]
    pgpass = config.cur_yaml["pass"]
    save_file(config, pgbin_path / "psql")
    subprocess_run(
        f"./psql -c \"create user {pguser} with superuser password '{pgpass}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
    subprocess_run(
        f'./psql -c "grant pg_monitor to {pguser}" postgres -p {pgport} -h localhost',
        cwd=pgbin_path,
    )

    # stop postgres so that we don't "leak" processes
    subprocess_run(
        f"./pg_ctl -D \"{pgdata_real_dpath}\" stop", cwd=pgbin_path
    )

    # create .tgz file
    # you can't pass "pgdata.tgz" as an arg to cur_task_runs_build_path() because that would create "pgdata.tgz" as a dir
    pgdata_tgz_real_fpath = config.cur_task_runs_build_path(".", mkdir=True) / "pgdata.tgz"
    # we need to cd into pgdata_real_dpath so that the tar file does not contain folders for the whole path of pgdata_real_dpath
    subprocess_run(
        f"tar -czf {pgdata_tgz_real_fpath} .", cwd=pgdata_real_dpath
    )

    # create symlink
    # only link at the end so that the link only ever points to a complete pgdata
    pgdata_tgz_symlink_path = _get_pgdata_tgz_symlink_path(config)
    if pgdata_tgz_symlink_path.exists():
        os.remove(pgdata_tgz_symlink_path)
    subprocess_run(f"ln -s {pgdata_tgz_real_fpath} {config.cur_symlinks_build_path(mkdir=True)}")


def init_db(config: DBGymConfig, dbname: str):
    pgbin_path = _get_pgbin_symlink_path(config)
    assert pgbin_path.exists()
    pguser = config.cur_yaml["user"]
    pgport = config.cur_yaml["port"]
    subprocess_run(
        f"./psql -c \"create database {dbname} with owner = '{pguser}'\" postgres -p {pgport} -h localhost",
        cwd=pgbin_path,
    )
