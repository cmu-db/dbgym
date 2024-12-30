import logging
from typing import Optional

import click
from gymlib.symlinks_paths import (
    get_tables_dirname,
    get_workload_dirname,
    get_workload_suffix,
)

from benchmark.constants import DEFAULT_SCALE_FACTOR
from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run
from util.workspace import DBGymWorkspace, fully_resolve_path

JOB_TABLES_URL = "https://event.cwi.nl/da/job/imdb.tgz"
JOB_QUERIES_URL = "https://event.cwi.nl/da/job/job.tgz"
JOB_QUERY_NAMES = [
    "1a",
    "1b",
    "1c",
    "1d",
    "2a",
    "2b",
    "2c",
    "2d",
    "3a",
    "3b",
    "3c",
    "4a",
    "4b",
    "4c",
    "5a",
    "5b",
    "5c",
    "6a",
    "6b",
    "6c",
    "6d",
    "6e",
    "6f",
    "7a",
    "7b",
    "7c",
    "8a",
    "8b",
    "8c",
    "8d",
    "9a",
    "9b",
    "9c",
    "9d",
    "10a",
    "10b",
    "10c",
    "11a",
    "11b",
    "11c",
    "11d",
    "12a",
    "12b",
    "12c",
    "13a",
    "13b",
    "13c",
    "13d",
    "14a",
    "14b",
    "14c",
    "15a",
    "15b",
    "15c",
    "15d",
    "16a",
    "16b",
    "16c",
    "16d",
    "17a",
    "17b",
    "17c",
    "17d",
    "17e",
    "17f",
    "18a",
    "18b",
    "18c",
    "19a",
    "19b",
    "19c",
    "19d",
    "20a",
    "20b",
    "20c",
    "21a",
    "21b",
    "21c",
    "22a",
    "22b",
    "22c",
    "22d",
    "23a",
    "23b",
    "23c",
    "24a",
    "24b",
    "25a",
    "25b",
    "25c",
    "26a",
    "26b",
    "26c",
    "27a",
    "27b",
    "27c",
    "28a",
    "28b",
    "28c",
    "29a",
    "29b",
    "29c",
    "30a",
    "30b",
    "30c",
    "31a",
    "31b",
    "31c",
    "32a",
    "32b",
    "33a",
    "33b",
    "33c",
]
JOB_QUERIES_DNAME = "job-queries"


@click.group(name="job")
@click.pass_obj
def job_group(dbgym_workspace: DBGymWorkspace) -> None:
    dbgym_workspace.append_group("job")


@job_group.command(name="tables")
# We expose this option to keep its interface consistent with other workloads, but you should never pass in something other than DEFAULT_SCALE_FACTOR.
@click.argument("scale-factor", type=float)
@click.pass_obj
# The reason generate data is separate from create dbdata is because generate data is generic
#   to all DBMSs while create dbdata is specific to a single DBMS.
def job_tables(dbgym_workspace: DBGymWorkspace, scale_factor: float) -> None:
    _job_tables(dbgym_workspace, scale_factor)


def _job_tables(dbgym_workspace: DBGymWorkspace, scale_factor: float) -> None:
    assert scale_factor == DEFAULT_SCALE_FACTOR
    _download_job_tables(dbgym_workspace)


@job_group.command(name="workload")
@click.option(
    "--query-subset",
    type=click.Choice(["all", "a", "demo"]),
    default="all",
)
@click.option("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
@click.pass_obj
def job_workload(
    dbgym_workspace: DBGymWorkspace, query_subset: str, scale_factor: float
) -> None:
    _job_workload(dbgym_workspace, query_subset, scale_factor)


def _job_workload(
    dbgym_workspace: DBGymWorkspace, query_subset: str, scale_factor: float
) -> None:
    assert scale_factor == DEFAULT_SCALE_FACTOR
    _download_job_queries(dbgym_workspace)
    _generate_job_workload(dbgym_workspace, query_subset)


def _download_job_tables(dbgym_workspace: DBGymWorkspace) -> None:
    _download_and_untar_dir(
        dbgym_workspace,
        JOB_TABLES_URL,
        "imdb.tgz",
        get_tables_dirname("job", DEFAULT_SCALE_FACTOR),
    )


def _download_job_queries(dbgym_workspace: DBGymWorkspace) -> None:
    _download_and_untar_dir(
        dbgym_workspace,
        JOB_QUERIES_URL,
        "job.tgz",
        JOB_QUERIES_DNAME,
        untarred_original_dname="job",
    )


def _download_and_untar_dir(
    dbgym_workspace: DBGymWorkspace,
    download_url: str,
    download_tarred_fname: str,
    untarred_dname: str,
    untarred_original_dname: Optional[str] = None,
) -> None:
    """
    Some .tgz files are built from a directory while others are built from the contents of
    the directory. If the .tgz file we're untarring is built from a directory, it will have
    an "original" directory name. If this is the case, you should set
    `untarred_original_dname` to ensure that it gets renamed to `untarred_dname`.
    """
    expected_symlink_path = (
        dbgym_workspace.dbgym_cur_symlinks_path / f"{untarred_dname}.link"
    )
    if expected_symlink_path.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping download: {expected_symlink_path}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Downloading: {expected_symlink_path}")
    subprocess_run(f"curl -O {download_url}", cwd=dbgym_workspace.dbgym_this_run_path)
    untarred_data_path = dbgym_workspace.dbgym_this_run_path / untarred_dname

    if untarred_original_dname is not None:
        assert not untarred_data_path.exists()
        subprocess_run(
            f"tar -zxvf {download_tarred_fname}",
            cwd=dbgym_workspace.dbgym_this_run_path,
        )
        assert (dbgym_workspace.dbgym_this_run_path / untarred_original_dname).exists()
        subprocess_run(
            f"mv {untarred_original_dname} {untarred_dname}",
            cwd=dbgym_workspace.dbgym_this_run_path,
        )
    else:
        untarred_data_path.mkdir(parents=True, exist_ok=False)
        subprocess_run(f"tar -zxvf ../{download_tarred_fname}", cwd=untarred_data_path)

    assert untarred_data_path.exists()
    subprocess_run(
        f"rm {download_tarred_fname}", cwd=dbgym_workspace.dbgym_this_run_path
    )
    symlink_path = dbgym_workspace.link_result(untarred_data_path)
    assert expected_symlink_path.samefile(symlink_path)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Downloaded: {expected_symlink_path}")


def _generate_job_workload(
    dbgym_workspace: DBGymWorkspace,
    query_subset: str,
) -> None:
    workload_name = get_workload_dirname(
        "job",
        DEFAULT_SCALE_FACTOR,
        get_workload_suffix("job", query_subset=query_subset),
    )
    expected_workload_symlink_path = dbgym_workspace.dbgym_cur_symlinks_path / (
        workload_name + ".link"
    )
    if expected_workload_symlink_path.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping generation: {expected_workload_symlink_path}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generating: {expected_workload_symlink_path}"
    )
    workload_path = dbgym_workspace.dbgym_this_run_path / workload_name
    workload_path.mkdir(parents=False, exist_ok=False)

    query_names = None
    if query_subset == "all":
        query_names = JOB_QUERY_NAMES
    elif query_subset == "a":
        query_names = [qname for qname in JOB_QUERY_NAMES if qname[-1] == "a"]
    elif query_subset == "demo":
        query_names = [f"{i}a" for i in range(1, 6)]
    else:
        assert False

    with open(workload_path / "order.txt", "w") as f:
        queries_parent_path = dbgym_workspace.dbgym_cur_symlinks_path / (
            JOB_QUERIES_DNAME + ".link"
        )

        for qname in query_names:
            sql_path = fully_resolve_path(queries_parent_path / f"{qname}.sql")
            f.write(f"Q{qname},{sql_path}\n")

    workload_symlink_path = dbgym_workspace.link_result(workload_path)
    assert workload_symlink_path == expected_workload_symlink_path
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generated: {expected_workload_symlink_path}"
    )
