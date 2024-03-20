import logging
import click

from misc.utils import DBGymConfig
from util.shell import subprocess_run
from util.sql import *

benchmark_tpch_logger = logging.getLogger("benchmark/tpch")
benchmark_tpch_logger.setLevel(logging.INFO)

TPCH_SCHEMA_FNAME = "tpch_schema.sql"
TPCH_CONSTRAINTS_FNAME = "tpch_constraints.sql"


@click.group(name="tpch")
@click.pass_obj
def tpch_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("tpch")


@tpch_group.command(name="generate-data")
@click.argument("scale-factor", type=float)
@click.pass_obj
# The reason generate-data is separate from create-pgdata is because generate-data is generic
#   to all DBMSs while create-pgdata is specific to Postgres.
def tpch_generate_data(dbgym_cfg: DBGymConfig, scale_factor: float):
    _clone(dbgym_cfg)
    _generate_data(dbgym_cfg, scale_factor)


@tpch_group.command(name="generate-workload")
@click.argument("workload-name", type=str)
@click.argument("seed-start", type=int)
@click.argument("seed-end", type=int)
@click.option(
    "--generate_type",
    type=click.Choice(["sequential", "even", "odd"]),
    default="sequential",
)
@click.pass_obj
def tpch_generate_workload(
    dbgym_cfg: DBGymConfig,
    workload_name: str,
    seed_start: int,
    seed_end: int,
    generate_type: str,
):
    _clone(dbgym_cfg)
    _generate_queries(dbgym_cfg, seed_start, seed_end)
    _generate_workload(dbgym_cfg, workload_name, seed_start, seed_end, generate_type)


def _clone(dbgym_cfg: DBGymConfig):
    symlink_dir = dbgym_cfg.cur_symlinks_build_path("tpch-kit")
    if symlink_dir.exists():
        benchmark_tpch_logger.info(f"Skipping clone: {symlink_dir}")
        return

    benchmark_tpch_logger.info(f"Cloning: {symlink_dir}")
    real_build_path = dbgym_cfg.cur_task_runs_build_path()
    subprocess_run(
        f"./tpch_setup.sh {real_build_path}", cwd=dbgym_cfg.cur_source_path()
    )
    subprocess_run(
        f"ln -s {real_build_path / 'tpch-kit'} {dbgym_cfg.cur_symlinks_build_path(mkdir=True)}"
    )
    benchmark_tpch_logger.info(f"Cloned: {symlink_dir}")


def _generate_queries(dbgym_cfg, seed_start, seed_end):
    build_path = dbgym_cfg.cur_symlinks_build_path()
    assert build_path.exists()

    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    benchmark_tpch_logger.info(
        f"Generating queries: {data_path} [{seed_start}, {seed_end}]"
    )
    for seed in range(seed_start, seed_end + 1):
        symlinked_seed = data_path / f"queries_{seed}"
        if symlinked_seed.exists():
            continue

        real_dir = dbgym_cfg.cur_task_runs_data_path(f"queries_{seed}", mkdir=True)
        for i in range(1, 22 + 1):
            target_sql = (real_dir / f"{i}.sql").resolve()
            subprocess_run(
                f"DSS_QUERY=./queries ./qgen {i} -r {seed} > {target_sql}",
                cwd=build_path / "tpch-kit" / "dbgen",
                verbose=False,
            )
        subprocess_run(f"ln -s {real_dir} {data_path}", verbose=False)
    benchmark_tpch_logger.info(
        f"Generated queries: {data_path} [{seed_start}, {seed_end}]"
    )


def _generate_data(dbgym_cfg: DBGymConfig, scale_factor: float):
    build_path = dbgym_cfg.cur_symlinks_build_path()
    assert build_path.exists()

    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    symlink_dir = data_path / f"tables_sf{scale_factor}"
    if symlink_dir.exists():
        benchmark_tpch_logger.info(f"Skipping generation: {symlink_dir}")
        return

    benchmark_tpch_logger.info(f"Generating: {symlink_dir}")
    subprocess_run(
        f"./dbgen -vf -s {scale_factor}", cwd=build_path / "tpch-kit" / "dbgen"
    )
    real_dir = dbgym_cfg.cur_task_runs_data_path(f"tables_sf{scale_factor}", mkdir=True)
    subprocess_run(f"mv ./*.tbl {real_dir}", cwd=build_path / "tpch-kit" / "dbgen")

    subprocess_run(f"ln -s {real_dir} {data_path}")
    benchmark_tpch_logger.info(f"Generated: {symlink_dir}")


def _generate_workload(
    dbgym_cfg: DBGymConfig,
    workload_name: str,
    seed_start: int,
    seed_end: int,
    generate_type: str,
):
    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    workload_path = data_path / f"workload_{workload_name}"
    if workload_path.exists():
        benchmark_tpch_logger.error(f"Workload directory exists: {workload_path}")
        raise RuntimeError(f"Workload directory exists: {workload_path}")

    benchmark_tpch_logger.info(f"Generating: {workload_path}")
    real_dir = dbgym_cfg.cur_task_runs_data_path(
        f"workload_{workload_name}", mkdir=True
    )

    queries = None
    if generate_type == "sequential":
        queries = [f"{i}" for i in range(1, 22 + 1)]
    elif generate_type == "even":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 0]
    elif generate_type == "odd":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 1]

    with open(real_dir / "order.txt", "w") as f:
        for seed in range(seed_start, seed_end + 1):
            for qnum in queries:
                sqlfile = data_path / f"queries_{seed}" / f"{qnum}.sql"
                assert sqlfile.exists()
                output = ",".join([f"S{seed}-Q{qnum}", str(sqlfile)])
                print(output, file=f)
                # TODO(WAN): add option to deep-copy the workload.
    subprocess_run(f"ln -s {real_dir} {data_path}")
    benchmark_tpch_logger.info(f"Generated: {workload_path}")
