import logging

import click

from benchmark.job.load_info import JobLoadInfo
from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run
from util.workspace import DBGymConfig, default_tables_dname, get_workload_name, link_result


JOB_TABLES_URL = "https://homepages.cwi.nl/~boncz/job/imdb.tgz"
JOB_QUERY_NAMES = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "2d", "3a", "3b", "3c", "4a", "4b", "4c", "5a", "5b", "5c", "6a", "6b", "6c", "6d", "6e", "6f", "7a", "7b", "7c", "8a", "8b", "8c", "8d", "9a", "9b", "9c", "9d", "10a", "10b", "10c", "11a", "11b", "11c", "11d", "12a", "12b", "12c", "13a", "13b", "13c", "13d", "14a", "14b", "14c", "15a", "15b", "15c", "15d", "16a", "16b", "16c", "16d", "17a", "17b", "17c", "17d", "17e", "17f", "18a", "18b", "18c", "19a", "19b", "19c", "19d", "20a", "20b", "20c", "21a", "21b", "21c", "22a", "22b", "22c", "22d", "23a", "23b", "23c", "24a", "24b", "25a", "25b", "25c", "26a", "26b", "26c", "27a", "27b", "27c", "28a", "28b", "28c", "29a", "29b", "29c", "30a", "30b", "30c", "31a", "31b", "31c", "32a", "32b", "33a", "33b", "33c"]


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
@click.option(
    "--query-subset",
    type=click.Choice(["all", "demo"]),
    default="all",
)
@click.pass_obj
def job_workload(dbgym_cfg: DBGymConfig, query_subset: str) -> None:
    _clone_job_queries(dbgym_cfg)
    _generate_job_workload(dbgym_cfg, query_subset)


def _download_job_data(dbgym_cfg: DBGymConfig) -> None:
    expected_symlink_dpath = (
        dbgym_cfg.cur_symlinks_data_path(mkdir=True)
        / f"{default_tables_dname(JobLoadInfo.JOB_SCALE_FACTOR)}.link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping download: {expected_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Downloading: {expected_symlink_dpath}")
    real_data_path = dbgym_cfg.cur_task_runs_data_path(mkdir=True)
    subprocess_run(f"curl -O {JOB_TABLES_URL}", cwd=real_data_path)
    job_data_dpath = dbgym_cfg.cur_task_runs_data_path(
        default_tables_dname(JobLoadInfo.JOB_SCALE_FACTOR), mkdir=True
    )
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


def _generate_job_workload(
    dbgym_cfg: DBGymConfig,
    query_subset: str,
) -> None:
    workload_name = get_workload_name(
        JobLoadInfo.JOB_SCALE_FACTOR, query_subset
    )
    expected_workload_symlink_dpath = dbgym_cfg.cur_symlinks_data_path(mkdir=True) / (workload_name + ".link")

    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generating: {expected_workload_symlink_dpath}"
    )
    real_dpath = dbgym_cfg.cur_task_runs_data_path(workload_name, mkdir=True)

    query_names = None
    if query_subset == "all":
        query_names = JOB_QUERY_NAMES
    elif query_subset == "demo":
        query_names = [f"{i}a" for i in range(1, 6)]
    else:
        assert False

    with open(real_dpath / "order.txt", "w") as f:
        for qname in query_names:
            sql_fpath = (
                dbgym_cfg.cur_symlinks_build_path(mkdir=True)
                / ("job-queries.link")
            ).resolve() / f"{qname}.sql"
            assert (
                sql_fpath.exists()
                and not sql_fpath.is_symlink()
                and sql_fpath.is_absolute()
            ), "We should only write existent real absolute paths to a file"
            f.write(f"Q{qname},{sql_fpath}\n")
            # TODO(WAN): add option to deep-copy the workload.

    workload_symlink_dpath = link_result(dbgym_cfg, real_dpath)
    assert workload_symlink_dpath == expected_workload_symlink_dpath
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generated: {expected_workload_symlink_dpath}"
    )