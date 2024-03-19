import logging
from pathlib import Path

import click
from sqlalchemy import create_engine

from misc.utils import DBGymConfig
from util.shell import subprocess_run
from util.sql import *

benchmark_tpch_logger = logging.getLogger("benchmark/tpch")
benchmark_tpch_logger.setLevel(logging.INFO)


@click.group(name="tpch")
@click.pass_obj
def tpch_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("tpch")


@tpch_group.command(name="generate-sf")
@click.argument("sf", type=int)
@click.pass_obj
def tpch_generate_sf(dbgym_cfg: DBGymConfig, sf: int):
    clone(dbgym_cfg)
    generate_tables(dbgym_cfg, sf)


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
    clone(dbgym_cfg)
    generate_queries(dbgym_cfg, seed_start, seed_end)
    generate_workload(dbgym_cfg, workload_name, seed_start, seed_end, generate_type)


@tpch_group.command(name="load-sf")
@click.argument("sf", type=int)
@click.argument("dbms", type=str)
@click.argument("dbname", type=str)
@click.pass_obj
def tpch_load_tables(dbgym_cfg: DBGymConfig, sf: int, dbms: str, dbname: str):
    clone(dbgym_cfg)
    generate_tables(dbgym_cfg, sf)
    load_tables(dbgym_cfg, sf, dbms, dbname)


def clone(dbgym_cfg: DBGymConfig):
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


def generate_queries(dbgym_cfg, seed_start, seed_end):
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


def generate_tables(dbgym_cfg: DBGymConfig, sf: int):
    build_path = dbgym_cfg.cur_symlinks_build_path()
    assert build_path.exists()

    data_path = dbgym_cfg.cur_symlinks_data_path(mkdir=True)
    symlink_dir = data_path / f"tables_sf{sf}"
    if symlink_dir.exists():
        benchmark_tpch_logger.info(f"Skipping generation: {symlink_dir}")
        return

    benchmark_tpch_logger.info(f"Generating: {symlink_dir}")
    subprocess_run(f"./dbgen -vf -s {sf}", cwd=build_path / "tpch-kit" / "dbgen")
    real_dir = dbgym_cfg.cur_task_runs_data_path(f"tables_sf{sf}", mkdir=True)
    subprocess_run(f"mv ./*.tbl {real_dir}", cwd=build_path / "tpch-kit" / "dbgen")

    subprocess_run(f"ln -s {real_dir} {data_path}")
    benchmark_tpch_logger.info(f"Generated: {symlink_dir}")


def generate_workload(
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


def _loaded(conn: Connection):
    # l_sk_pk is the last index that we create.
    sql = "SELECT * FROM pg_indexes WHERE indexname = 'l_sk_pk'"
    res = conn_execute(conn, sql).fetchall()
    return len(res) > 0


def _load(dbgym_cfg: DBGymConfig, conn: Connection, sf: int):
    schema_root = dbgym_cfg.cur_source_path()
    data_root = dbgym_cfg.cur_symlinks_data_path()

    tables = [
        "region",
        "nation",
        "part",
        "supplier",
        "partsupp",
        "customer",
        "orders",
        "lineitem",
    ]

    sql_file_execute(conn, schema_root / "tpch_schema.sql")
    for table in tables:
        conn_execute(conn, f"TRUNCATE {table} CASCADE")
    for table in tables:
        table_path = data_root / f"tables_sf{sf}" / f"{table}.tbl"

        with open(table_path, "r") as table_csv:
            with conn.connection.dbapi_connection.cursor() as cur:
                with cur.copy(f"COPY {table} FROM STDIN CSV DELIMITER '|'") as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)
    sql_file_execute(conn, schema_root / "tpch_constraints.sql")


def load_tables(dbgym_cfg: DBGymConfig, sf: int, dbms: str, dbname: str):
    # TODO(WAN): repetition and slight break of abstraction here.
    dbms_yaml = dbgym_cfg.root_yaml["dbms"][dbms]
    dbms_user = dbms_yaml["user"]
    dbms_pass = dbms_yaml["pass"]
    dbms_port = dbms_yaml["port"]

    if dbms == "postgres":
        connstr = f"postgresql+psycopg://{dbms_user}:{dbms_pass}@localhost:{dbms_port}/{dbname}"
    else:
        raise RuntimeError(f"Unknown DBMS: {dbms}")

    engine: Engine = create_engine(
        connstr,
        execution_options={"isolation_level": "AUTOCOMMIT"},
    )
    with engine.connect() as conn:
        if _loaded(conn):
            benchmark_tpch_logger.info(f"Skipping load: TPC-H SF {sf}")
        else:
            benchmark_tpch_logger.info(f"Loading: TPC-H SF {sf}")
            _load(dbgym_cfg, conn, sf)
    benchmark_tpch_logger.info(f"Loaded: TPC-H SF {sf}")
