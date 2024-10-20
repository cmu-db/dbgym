import logging
from pathlib import Path

import click

from misc.utils import (
    DBGymConfig,
    get_scale_factor_string,
    link_result,
    workload_name_fn,
)
from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run


@click.group(name="tpch")
@click.pass_obj
def tpch_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("tpch")


@tpch_group.command(name="data")
@click.argument("scale-factor", type=float)
@click.pass_obj
# The reason generate data is separate from create dbdata is because generate-data is generic
#   to all DBMSs while create dbdata is specific to a single DBMS.
def tpch_data(dbgym_cfg: DBGymConfig, scale_factor: float) -> None:
    _clone(dbgym_cfg)
    _generate_data(dbgym_cfg, scale_factor)


@tpch_group.command(name="workload")
@click.option(
    "--seed-start",
    type=int,
    default=15721,
    help="A workload consists of queries from multiple seeds. This is the starting seed (inclusive).",
)
@click.option(
    "--seed-end",
    type=int,
    default=15721,
    help="A workload consists of queries from multiple seeds. This is the ending seed (inclusive).",
)
@click.option(
    "--query-subset",
    type=click.Choice(["all", "even", "odd"]),
    default="all",
)
@click.option("--scale-factor", type=float, default=1)
@click.pass_obj
def tpch_workload(
    dbgym_cfg: DBGymConfig,
    seed_start: int,
    seed_end: int,
    query_subset: str,
    scale_factor: float,
) -> None:
    assert (
        seed_start <= seed_end
    ), f"seed_start ({seed_start}) must be <= seed_end ({seed_end})"
    _clone(dbgym_cfg)
    _generate_queries(dbgym_cfg, seed_start, seed_end, scale_factor)
    _generate_workload(dbgym_cfg, seed_start, seed_end, query_subset, scale_factor)


def _get_queries_dname(seed: int, scale_factor: float) -> str:
    return f"queries_{seed}_sf{get_scale_factor_string(scale_factor)}"


def _clone(dbgym_cfg: DBGymConfig) -> None:
    expected_symlink_dpath = (
        dbgym_cfg.cur_symlinks_build_path(mkdir=True) / "tpch-kit.link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(f"Skipping clone: {expected_symlink_dpath}")
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloning: {expected_symlink_dpath}")
    real_build_path = dbgym_cfg.cur_task_runs_build_path()
    subprocess_run(
        f"./tpch_setup.sh {real_build_path}", cwd=dbgym_cfg.cur_source_path()
    )
    symlink_dpath = link_result(dbgym_cfg, real_build_path / "tpch-kit")
    assert expected_symlink_dpath.samefile(symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloned: {expected_symlink_dpath}")


def _get_tpch_kit_dpath(dbgym_cfg: DBGymConfig) -> Path:
    tpch_kit_dpath = (dbgym_cfg.cur_symlinks_build_path() / "tpch-kit.link").resolve()
    assert (
        tpch_kit_dpath.exists()
        and tpch_kit_dpath.is_absolute()
        and not tpch_kit_dpath.is_symlink()
    )
    return tpch_kit_dpath


def _generate_queries(
    dbgym_cfg: DBGymConfig, seed_start: int, seed_end: int, scale_factor: float
) -> None:
    tpch_kit_dpath = _get_tpch_kit_dpath(dbgym_cfg)
    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generating queries: {data_path} [{seed_start}, {seed_end}]")
    for seed in range(seed_start, seed_end + 1):
        expected_queries_symlink_dpath = data_path / (
            _get_queries_dname(seed, scale_factor) + ".link"
        )
        if expected_queries_symlink_dpath.exists():
            continue

        real_dir = dbgym_cfg.cur_task_runs_data_path(
            _get_queries_dname(seed, scale_factor), mkdir=True
        )
        for i in range(1, 22 + 1):
            target_sql = (real_dir / f"{i}.sql").resolve()
            subprocess_run(
                f"DSS_QUERY=./queries ./qgen {i} -r {seed} -s {scale_factor} > {target_sql}",
                cwd=tpch_kit_dpath / "dbgen",
                verbose=False,
            )
        queries_symlink_dpath = link_result(dbgym_cfg, real_dir)
        assert queries_symlink_dpath.samefile(expected_queries_symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generated queries: {data_path} [{seed_start}, {seed_end}]")


def _generate_data(dbgym_cfg: DBGymConfig, scale_factor: float) -> None:
    tpch_kit_dpath = _get_tpch_kit_dpath(dbgym_cfg)
    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    expected_tables_symlink_dpath = (
        data_path / f"tables_sf{get_scale_factor_string(scale_factor)}.link"
    )
    if expected_tables_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(f"Skipping generation: {expected_tables_symlink_dpath}")
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generating: {expected_tables_symlink_dpath}")
    subprocess_run(f"./dbgen -vf -s {scale_factor}", cwd=tpch_kit_dpath / "dbgen")
    real_dir = dbgym_cfg.cur_task_runs_data_path(
        f"tables_sf{get_scale_factor_string(scale_factor)}", mkdir=True
    )
    subprocess_run(f"mv ./*.tbl {real_dir}", cwd=tpch_kit_dpath / "dbgen")

    tables_symlink_dpath = link_result(dbgym_cfg, real_dir)
    assert tables_symlink_dpath.samefile(expected_tables_symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generated: {expected_tables_symlink_dpath}")


def _generate_workload(
    dbgym_cfg: DBGymConfig,
    seed_start: int,
    seed_end: int,
    query_subset: str,
    scale_factor: float,
) -> None:
    symlink_data_dpath = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    workload_name = workload_name_fn(scale_factor, seed_start, seed_end, query_subset)
    expected_workload_symlink_dpath = symlink_data_dpath / (workload_name + ".link")

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generating: {expected_workload_symlink_dpath}")
    real_dpath = dbgym_cfg.cur_task_runs_data_path(workload_name, mkdir=True)

    queries = None
    if query_subset == "all":
        queries = [f"{i}" for i in range(1, 22 + 1)]
    elif query_subset == "even":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 0]
    elif query_subset == "odd":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 1]
    else:
        assert False

    with open(real_dpath / "order.txt", "w") as f:
        for seed in range(seed_start, seed_end + 1):
            for qnum in queries:
                sql_fpath = (
                    symlink_data_dpath
                    / (_get_queries_dname(seed, scale_factor) + ".link")
                ).resolve() / f"{qnum}.sql"
                assert (
                    sql_fpath.exists()
                    and not sql_fpath.is_symlink()
                    and sql_fpath.is_absolute()
                ), "We should only write existent real absolute paths to a file"
                f.write(f"S{seed}-Q{qnum},{sql_fpath}\n")
                # TODO(WAN): add option to deep-copy the workload.

    workload_symlink_dpath = link_result(dbgym_cfg, real_dpath)
    assert workload_symlink_dpath == expected_workload_symlink_dpath
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Generated: {expected_workload_symlink_dpath}")
