import logging
from pathlib import Path

import click
from sqlalchemy import create_engine

from util.shell import subprocess_run
from util.sql import *

benchmark_tpch_logger = logging.getLogger("benchmark/tpch")
benchmark_tpch_logger.setLevel(logging.INFO)


@click.group(name="tpch")
@click.pass_obj
def tpch_group(config):
    config.append_group("tpch")


@tpch_group.command(name="generate-sf")
@click.argument("sf", type=int)
@click.pass_obj
def tpch_generate_sf(config, sf):
    clone(config)
    generate_tables(config, sf)


@tpch_group.command(name="generate-workload")
@click.argument("output-dir", type=str)
@click.argument("seed-start", type=int)
@click.argument("seed-end", type=int)
@click.option(
    "--generate_type",
    type=click.Choice(["sequential", "even", "odd"]),
    default="sequential",
)
@click.pass_obj
def tpch_generate_workload(config, output_dir, seed_start, seed_end, generate_type):
    clone(config)
    generate_queries(config, seed_start, seed_end)
    generate_workload(config, output_dir, seed_start, seed_end, generate_type)


@tpch_group.command(name="load-sf")
@click.argument("sf", type=int)
@click.argument("dbms", type=str)
@click.argument("dbname", type=str)
@click.pass_obj
def tpch_load_tables(config, sf, dbms, dbname):
    clone(config)
    generate_tables(config, sf)
    load_tables(config, sf, dbms, dbname)


def clone(config):
    if config.cur_build_path.exists():
        benchmark_tpch_logger.info(f"Skipping clone: {config.cur_build_path}")
        return

    benchmark_tpch_logger.info(f"Cloning: {config.cur_build_path}")
    build_path = (config.cur_build_path / "..").resolve()
    subprocess_run(f"./tpch_setup.sh {config.cur_run_path}", cwd=config.cur_path)
    build_path.mkdir(parents=True, exist_ok=True)
    subprocess_run(f"ln -s {config.cur_run_path} {build_path}")
    benchmark_tpch_logger.info(f"Cloned: {config.cur_build_path}")


def generate_queries(config, seed_start, seed_end):
    assert config.cur_build_path.exists()

    data_path_parent = config.cur_data_path / "queries" / "seed"
    data_path_parent.mkdir(parents=True, exist_ok=True)
    benchmark_tpch_logger.info(
        f"Generating queries: {data_path_parent} [{seed_start}, {seed_end}]"
    )
    for seed in range(seed_start, seed_end + 1):
        data_path = data_path_parent / f"{seed}"
        if data_path.exists():
            continue

        target_dir = config.cur_run_path / "queries" / "seed" / f"{seed}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 22 + 1):
            target_sql = (target_dir / f"{i}.sql").resolve()
            subprocess_run(
                f"DSS_QUERY=./queries ./qgen {i} -r {seed} > {target_sql}",
                cwd=config.cur_build_path / "tpch-kit" / "dbgen",
                verbose=False,
            )
        subprocess_run(f"ln -s {target_dir} {data_path_parent}", verbose=False)


def generate_tables(config, sf):
    assert config.cur_build_path.exists()
    data_path = config.cur_data_path / "tables" / "sf" / f"{sf}"

    if data_path.exists():
        benchmark_tpch_logger.info(f"Skipping generation: {data_path}")
        return

    benchmark_tpch_logger.info(f"Generating: {data_path}")
    subprocess_run(
        f"./dbgen -vf -s {sf}", cwd=config.cur_build_path / "tpch-kit" / "dbgen"
    )
    target_dir = config.cur_run_path / "tables" / "sf" / f"{sf}"
    subprocess_run(f"mkdir -p {target_dir}")
    subprocess_run(
        f"mv ./*.tbl {target_dir}", cwd=config.cur_build_path / "tpch-kit" / "dbgen"
    )

    data_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess_run(f"mkdir -p {data_path.parent}")
    subprocess_run(f"ln -s {target_dir} {data_path.parent}")
    benchmark_tpch_logger.info(f"Generated: {data_path}")


def generate_workload(config, output_dir, seed_start, seed_end, generate_type):
    data_path = config.cur_data_path / "workloads"
    output_dir = (data_path / output_dir).resolve().absolute()

    if output_dir.exists():
        benchmark_tpch_logger.error(f"Workload directory exists: {output_dir}")
        raise RuntimeError(f"Workload directory exists: {output_dir}")

    benchmark_tpch_logger.info(f"Generating: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = None
    if generate_type == "sequential":
        queries = [f"{i}" for i in range(1, 22 + 1)]
    elif generate_type == "even":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 0]
    elif generate_type == "odd":
        queries = [f"{i}" for i in range(1, 22 + 1) if i % 2 == 1]

    with open(output_dir / "order.txt", "w") as f:
        for seed in range(seed_start, seed_end + 1):
            for qnum in queries:
                sqlfile = (
                    config.cur_data_path
                    / "queries"
                    / "seed"
                    / str(seed)
                    / f"{qnum}.sql"
                )
                assert sqlfile.exists()
                output = ",".join([f"S{seed}-Q{qnum}", str(sqlfile)])
                print(output, file=f)
                # TODO(WAN): add option to deep-copy the workload.


def _loaded(conn: Connection):
    # l_sk_pk is the last index that we create.
    sql = "SELECT * FROM pg_indexes WHERE indexname = 'l_sk_pk'"
    res = conn_execute(conn, sql).fetchall()
    return len(res) > 0


def _load(config, conn: Connection, sf: int):
    schema_root = config.cur_path
    data_root = config.cur_data_path

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
        table_path = data_root / "tables" / "sf" / str(sf) / f"{table}.tbl"

        with open(table_path, "r") as table_csv:
            with conn.connection.dbapi_connection.cursor() as cur:
                with cur.copy(f"COPY {table} FROM STDIN CSV DELIMITER '|'") as copy:
                    while data := table_csv.read(8192):
                        copy.write(data)
    sql_file_execute(conn, schema_root / "tpch_constraints.sql")


def load_tables(config, sf, dbms, dbname):
    # TODO(WAN): repetition and slight break of abstraction here.
    dbms_yaml = config.root_yaml["dbms"][dbms]
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
            _load(config, conn, sf)
    benchmark_tpch_logger.info(f"Loaded: TPC-H SF {sf}")
