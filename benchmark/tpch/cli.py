import logging

import click
from gymlib.symlinks_paths import (
    get_tables_dirname,
    get_tables_symlink_path,
    get_workload_dirname,
    get_workload_suffix,
)

from benchmark.constants import DEFAULT_SCALE_FACTOR
from benchmark.tpch.constants import DEFAULT_TPCH_SEED, NUM_TPCH_QUERIES
from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run
from util.workspace import (
    DBGymWorkspace,
    fully_resolve_path,
    get_scale_factor_string,
    is_fully_resolved,
    link_result,
)

TPCH_KIT_DIRNAME = "tpch-kit"


@click.group(name="tpch")
@click.pass_obj
def tpch_group(dbgym_workspace: DBGymWorkspace) -> None:
    dbgym_workspace.append_group("tpch")


@tpch_group.command(name="tables")
@click.argument("scale-factor", type=float)
@click.pass_obj
# The reason generate tables is separate from create dbdata is because tpch_tables is generic
#   to all DBMSs while create dbdata is specific to a single DBMS.
def tpch_tables(dbgym_workspace: DBGymWorkspace, scale_factor: float) -> None:
    _tpch_tables(dbgym_workspace, scale_factor)


def _tpch_tables(dbgym_workspace: DBGymWorkspace, scale_factor: float) -> None:
    """
    This function exists as a hook for integration tests.
    """
    _clone_tpch_kit(dbgym_workspace)
    _generate_tpch_tables(dbgym_workspace, scale_factor)


@tpch_group.command(name="workload")
@click.option(
    "--seed-start",
    type=int,
    default=DEFAULT_TPCH_SEED,
    help="A workload consists of queries from multiple seeds. This is the starting seed (inclusive).",
)
@click.option(
    "--seed-end",
    type=int,
    default=DEFAULT_TPCH_SEED,
    help="A workload consists of queries from multiple seeds. This is the ending seed (inclusive).",
)
@click.option(
    "--query-subset",
    type=click.Choice(["all", "even", "odd"]),
    default="all",
)
@click.option("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
@click.pass_obj
def tpch_workload(
    dbgym_workspace: DBGymWorkspace,
    seed_start: int,
    seed_end: int,
    query_subset: str,
    scale_factor: float,
) -> None:
    _tpch_workload(dbgym_workspace, seed_start, seed_end, query_subset, scale_factor)


def _tpch_workload(
    dbgym_workspace: DBGymWorkspace,
    seed_start: int,
    seed_end: int,
    query_subset: str,
    scale_factor: float,
) -> None:
    """
    This function exists as a hook for integration tests.
    """
    assert (
        seed_start <= seed_end
    ), f"seed_start ({seed_start}) must be <= seed_end ({seed_end})"
    _clone_tpch_kit(dbgym_workspace)
    _generate_tpch_queries(dbgym_workspace, seed_start, seed_end, scale_factor)
    _generate_tpch_workload(
        dbgym_workspace, seed_start, seed_end, query_subset, scale_factor
    )


def _get_queries_dirname(seed: int, scale_factor: float) -> str:
    return f"queries_{seed}_sf{get_scale_factor_string(scale_factor)}"


def _clone_tpch_kit(dbgym_workspace: DBGymWorkspace) -> None:
    expected_symlink_dpath = dbgym_workspace.dbgym_cur_symlinks_path / (
        TPCH_KIT_DIRNAME + ".link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping clone: {expected_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloning: {expected_symlink_dpath}")
    subprocess_run(
        f"./clone_tpch_kit.sh {dbgym_workspace.dbgym_this_run_path}",
        cwd=dbgym_workspace.base_dbgym_repo_dpath / "benchmark" / "tpch",
    )
    symlink_dpath = dbgym_workspace.link_result(
        dbgym_workspace.dbgym_this_run_path / TPCH_KIT_DIRNAME
    )
    assert expected_symlink_dpath.samefile(symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloned: {expected_symlink_dpath}")


def _generate_tpch_queries(
    dbgym_workspace: DBGymWorkspace, seed_start: int, seed_end: int, scale_factor: float
) -> None:
    tpch_kit_dpath = dbgym_workspace.dbgym_cur_symlinks_path / (
        TPCH_KIT_DIRNAME + ".link"
    )
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generating queries: [{seed_start}, {seed_end}]"
    )
    for seed in range(seed_start, seed_end + 1):
        expected_queries_symlink_dpath = dbgym_workspace.dbgym_cur_symlinks_path / (
            _get_queries_dirname(seed, scale_factor) + ".link"
        )
        if expected_queries_symlink_dpath.exists():
            continue

        queries_parent_path = (
            dbgym_workspace.dbgym_this_run_path
            / _get_queries_dirname(seed, scale_factor)
        )
        queries_parent_path.mkdir(parents=False, exist_ok=False)
        for i in range(1, NUM_TPCH_QUERIES + 1):
            target_sql = (queries_parent_path / f"{i}.sql").resolve()
            subprocess_run(
                f"DSS_QUERY=./queries ./qgen {i} -r {seed} -s {scale_factor} > {target_sql}",
                cwd=tpch_kit_dpath / "dbgen",
                verbose=False,
            )
        queries_symlink_dpath = dbgym_workspace.link_result(queries_parent_path)
        assert queries_symlink_dpath.samefile(expected_queries_symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generated queries: [{seed_start}, {seed_end}]"
    )


def _generate_tpch_tables(dbgym_workspace: DBGymWorkspace, scale_factor: float) -> None:
    tpch_kit_dpath = dbgym_workspace.dbgym_cur_symlinks_path / (
        TPCH_KIT_DIRNAME + ".link"
    )
    expected_tables_symlink_dpath = get_tables_symlink_path(
        dbgym_workspace.dbgym_workspace_path, "tpch", scale_factor
    )
    if expected_tables_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping generation: {expected_tables_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generating: {expected_tables_symlink_dpath}"
    )
    subprocess_run(f"./dbgen -vf -s {scale_factor}", cwd=tpch_kit_dpath / "dbgen")
    tables_parent_path = dbgym_workspace.dbgym_this_run_path / get_tables_dirname(
        "tpch", scale_factor
    )
    tables_parent_path.mkdir(parents=False, exist_ok=False)
    subprocess_run(f"mv ./*.tbl {tables_parent_path}", cwd=tpch_kit_dpath / "dbgen")

    tables_symlink_dpath = dbgym_workspace.link_result(tables_parent_path)
    assert tables_symlink_dpath.samefile(expected_tables_symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generated: {expected_tables_symlink_dpath}"
    )


def _generate_tpch_workload(
    dbgym_workspace: DBGymWorkspace,
    seed_start: int,
    seed_end: int,
    query_subset: str,
    scale_factor: float,
) -> None:
    workload_name = get_workload_dirname(
        "tpch",
        scale_factor,
        get_workload_suffix(
            "tpch", seed_start=seed_start, seed_end=seed_end, query_subset=query_subset
        ),
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
        query_names = [f"{i}" for i in range(1, NUM_TPCH_QUERIES + 1)]
    elif query_subset == "even":
        query_names = [f"{i}" for i in range(1, NUM_TPCH_QUERIES + 1) if i % 2 == 0]
    elif query_subset == "odd":
        query_names = [f"{i}" for i in range(1, NUM_TPCH_QUERIES + 1) if i % 2 == 1]
    else:
        assert False

    with open(workload_path / "order.txt", "w") as f:
        for seed in range(seed_start, seed_end + 1):
            queries_parent_path = dbgym_workspace.dbgym_cur_symlinks_path / (
                _get_queries_dirname(seed, scale_factor) + ".link"
            )

            for qname in query_names:
                sql_fpath = fully_resolve_path(queries_parent_path / f"{qname}.sql")
                assert is_fully_resolved(
                    sql_fpath
                ), "We should only write existent real absolute paths to a file"
                f.write(f"S{seed}-Q{qname},{sql_fpath}\n")

    workload_symlink_path = dbgym_workspace.link_result(workload_path)
    assert workload_symlink_path == expected_workload_symlink_path
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"Generated: {expected_workload_symlink_path}"
    )
