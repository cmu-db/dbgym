import logging

import click

from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run
from util.workspace import DBGymConfig, link_result

JOB_TABLES_URL = "https://homepages.cwi.nl/~boncz/job/imdb.tgz"


@click.group(name="job")
@click.pass_obj
def job_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("job")


@job_group.command(name="data")
@click.pass_obj
# The reason generate data is separate from create dbdata is because generate-data is generic
#   to all DBMSs while create dbdata is specific to a single DBMS.
def job_data(dbgym_cfg: DBGymConfig) -> None:
    _download_job_data(dbgym_cfg)


@job_group.command(name="workload")
@click.pass_obj
def tpch_workload(dbgym_cfg: DBGymConfig) -> None:
    _clone_job_queries(dbgym_cfg)


def _download_job_data(dbgym_cfg: DBGymConfig) -> None:
    expected_symlink_dpath = (
        dbgym_cfg.cur_symlinks_data_path(mkdir=True) / "job-data.link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping download: {expected_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Downloading: {expected_symlink_dpath}")
    real_data_path = dbgym_cfg.cur_task_runs_data_path(mkdir=True)
    subprocess_run(f"curl -O {JOB_TABLES_URL}", cwd=real_data_path)
    job_data_dpath = dbgym_cfg.cur_task_runs_data_path("job-data", mkdir=True)
    subprocess_run("tar -zxvf ../imdb.tgz", cwd=job_data_dpath)
    subprocess_run(f"rm imdb.tgz", cwd=real_data_path)
    symlink_dpath = link_result(dbgym_cfg, job_data_dpath)
    assert expected_symlink_dpath.samefile(symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Downloaded: {expected_symlink_dpath}")


def _clone_job_queries(dbgym_cfg: DBGymConfig) -> None:
    expected_symlink_dpath = (
        dbgym_cfg.cur_symlinks_build_path(mkdir=True) / "job-queries.link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping clone: {expected_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloning: {expected_symlink_dpath}")
    real_build_path = dbgym_cfg.cur_task_runs_build_path(mkdir=True)
    subprocess_run(
        f"./clone_job_queries.sh {real_build_path}", cwd=dbgym_cfg.cur_source_path()
    )
    symlink_dpath = link_result(dbgym_cfg, real_build_path / "job-queries")
    assert expected_symlink_dpath.samefile(symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloned: {expected_symlink_dpath}")
