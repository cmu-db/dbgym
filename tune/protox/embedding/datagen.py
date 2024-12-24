import copy
import gc
import logging
import math
import os
import random
import shutil
import time
from itertools import chain, combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any, NewType, Optional, cast

import click
import numpy as np
import pandas as pd
import psycopg
import yaml
from sklearn.preprocessing import quantile_transform

from benchmark.constants import DEFAULT_SCALE_FACTOR
from dbms.postgres.cli import start_postgres, stop_postgres
from tune.protox.embedding.loss import COST_COLUMNS
from tune.protox.env.space.primitive_space.index_space import IndexSpace
from tune.protox.env.types import (
    QuerySpec,
    QueryType,
    TableAttrAccessSetsMap,
    TableAttrListMap,
)
from tune.protox.env.workload import Workload
from util.log import DBGYM_LOGGER_NAME
from util.pg import create_psycopg_conn
from util.shell import subprocess_run
from util.workspace import (
    BENCHMARK_NAME_PLACEHOLDER,
    SCALE_FACTOR_PLACEHOLDER,
    WORKLOAD_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    DBGymConfig,
    fully_resolve_path,
    get_default_benchmark_config_path,
    get_default_dbdata_parent_dpath,
    get_default_pgbin_path,
    get_default_pristine_dbdata_snapshot_path,
    get_default_traindata_fname,
    get_default_workload_name_suffix,
    get_default_workload_path,
    get_workload_name,
    is_fully_resolved,
    is_ssd,
    link_result,
    open_and_save,
    save_file,
)

# FUTURE(oltp)
# try:
#     sys.path.append("/home/wz2/noisepage-pilot")
#     from behavior.utils.prepare_ou_data import clean_input_data
# except:
#     pass


QueryBatches = NewType(
    "QueryBatches", list[tuple[str, list[tuple[QueryType, str]], Any]]
)


# click steup
@click.command()
@click.pass_obj

# generic args
@click.argument("benchmark-name")
@click.option(
    "--workload-name-suffix",
    type=str,
    default=None,
    help=f"The suffix of the workload name (the part after the scale factor).",
)
@click.option(
    "--scale-factor",
    type=float,
    default=DEFAULT_SCALE_FACTOR,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--pgbin-path",
    type=Path,
    default=None,
    help=f"The path to the bin containing Postgres executables. The default is {get_default_pgbin_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
# TODO(phw2): need to run pgtune before gathering data
@click.option(
    "--pristine-dbdata-snapshot-path",
    type=Path,
    default=None,
    help=f"The path to the .tgz snapshot of the dbdata directory to build an embedding space over. The default is {get_default_pristine_dbdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
@click.option(
    "--intended-dbdata-hardware",
    type=click.Choice(["hdd", "ssd"]),
    default="hdd",
    help=f"The intended hardware dbdata should be on. Used as a sanity check for --dbdata-parent-dpath.",
)
@click.option(
    "--dbdata-parent-dpath",
    type=Path,
    default=None,
    help=f"The path to the parent directory of the dbdata which will be actively tuned. The default is {get_default_pristine_dbdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
@click.option(
    "--benchmark-config-path",
    type=Path,
    default=None,
    help=f"The path to the .yaml config file for the benchmark. The default is {get_default_benchmark_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--workload-path",
    type=Path,
    default=None,
    help=f"The path to the directory that specifies the workload (such as its queries and order of execution). The default is {get_default_workload_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
)

# dir gen args
@click.option(
    "--leading-col-tbls",
    type=str,
    default=None,
    help='All tables included here will have indexes created s.t. each column is represented equally often as the "leading column" of the index.',
)
# TODO(wz2): what if we sample tbl_sample_limit / len(cols) for tables in leading_col_tbls? this way, tbl_sample_limit will always represent the total # of indexes created on that table. currently the description of the param is a bit weird as you can see
@click.option(
    "--default-sample-limit",
    type=int,
    default=2048,
    help="The default sample limit of all tables, used unless override sample limit is specified. If the table is in --leading-col-tbls, sample limit is # of indexes to sample per column for that table table. If the table is in --leading-col-tbls, sample limit is the # of indexes to sample total for that table.",
)
@click.option(
    "--override-sample-limits",
    type=str,
    default=None,
    help='Override the sample limit for specific tables. An example input would be "lineitem,32768,orders,4096".',
)
# TODO(wz2): if I'm just outputting out.parquet instead of the full directory, do we even need file limit at all?
@click.option(
    "--file-limit",
    type=int,
    default=1024,
    help="The max # of data points (one data point = one hypothetical index) per file",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="The max # of concurrent threads that will be creating hypothetical indexes. The default is `nproc`.",
)
# TODO(wz2): when would we not want to generate costs?
@click.option("--no-generate-costs", is_flag=True, help="Turn off generating costs.")

# file gen args
@click.option("--table-shape", is_flag=True, help="TODO(wz2)")
@click.option("--dual-class", is_flag=True, help="TODO(wz2)")
@click.option("--pad-min", type=int, default=None, help="TODO(wz2)")
@click.option("--rebias", type=float, default=0, help="TODO(wz2)")
def datagen(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    workload_name_suffix: Optional[str],
    scale_factor: float,
    pgbin_path: Optional[Path],
    pristine_dbdata_snapshot_path: Optional[Path],
    intended_dbdata_hardware: str,
    dbdata_parent_dpath: Optional[Path],
    benchmark_config_path: Optional[Path],
    workload_path: Optional[Path],
    seed: Optional[int],
    leading_col_tbls: str,
    default_sample_limit: int,
    override_sample_limits: Optional[str],
    file_limit: int,
    max_concurrent: Optional[int],
    no_generate_costs: bool,
    table_shape: bool,
    dual_class: bool,
    pad_min: int,
    rebias: float,
) -> None:
    """
    Samples the effects of indexes on the workload as estimated by HypoPG.
    Outputs all this data as a .parquet file in the run_*/ dir.
    Updates the symlink in the data/ dir to point to the new .parquet file.
    """
    # Set args to defaults programmatically (do this before doing anything else in the function).
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if workload_name_suffix is None:
        workload_name_suffix = get_default_workload_name_suffix(benchmark_name)
    workload_name = get_workload_name(scale_factor, workload_name_suffix)
    if workload_path is None:
        workload_path = get_default_workload_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if benchmark_config_path is None:
        benchmark_config_path = get_default_benchmark_config_path(benchmark_name)
    if pgbin_path is None:
        pgbin_path = get_default_pgbin_path(dbgym_cfg.dbgym_workspace_path)
    if pristine_dbdata_snapshot_path is None:
        pristine_dbdata_snapshot_path = get_default_pristine_dbdata_snapshot_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, scale_factor
        )
    if dbdata_parent_dpath is None:
        dbdata_parent_dpath = get_default_dbdata_parent_dpath(
            dbgym_cfg.dbgym_workspace_path
        )
    if max_concurrent is None:
        max_concurrent = os.cpu_count()
        assert max_concurrent is not None
    if seed is None:
        seed = random.randint(0, int(1e8))

    # Fully resolve all input paths.
    workload_path = fully_resolve_path(dbgym_cfg, workload_path)
    benchmark_config_path = fully_resolve_path(dbgym_cfg, benchmark_config_path)
    pgbin_path = fully_resolve_path(dbgym_cfg, pgbin_path)
    pristine_dbdata_snapshot_path = fully_resolve_path(
        dbgym_cfg, pristine_dbdata_snapshot_path
    )
    dbdata_parent_dpath = fully_resolve_path(dbgym_cfg, dbdata_parent_dpath)

    # Check assertions on args
    if intended_dbdata_hardware == "hdd":
        assert not is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is HDD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an SSD"
    elif intended_dbdata_hardware == "ssd":
        assert is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is SSD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an HDD"
    else:
        assert False

    # Process the "data structure" args
    leading_col_tbls_parsed: list[str] = (
        [] if leading_col_tbls is None else leading_col_tbls.split(",")
    )
    # I chose to only use the "," delimiter in override_sample_limits_str, so the dictionary is encoded as [key],[value],[key],[value]
    # I felt this was better than introducing a new delimiter which might conflict with the name of a table
    override_sample_limits_parsed: dict[str, int] = dict()
    if override_sample_limits is not None:
        override_sample_limits_str_split = override_sample_limits.split(",")
        assert (
            len(override_sample_limits_str_split) % 2 == 0
        ), f'override_sample_limits ("{override_sample_limits}") does not have an even number of values'
        for i in range(0, len(override_sample_limits_str_split), 2):
            tbl = override_sample_limits_str_split[i]
            limit = int(override_sample_limits_str_split[i + 1])
            override_sample_limits_parsed[tbl] = limit

    # Group args together to reduce the # of parameters we pass into functions
    # I chose to group them into separate objects instead because it felt hacky to pass a giant args object into every function
    generic_args = EmbeddingDatagenGenericArgs(
        benchmark_name,
        workload_name,
        scale_factor,
        benchmark_config_path,
        seed,
        workload_path,
        pristine_dbdata_snapshot_path,
        dbdata_parent_dpath,
    )
    dir_gen_args = EmbeddingDirGenArgs(
        leading_col_tbls_parsed,
        default_sample_limit,
        override_sample_limits_parsed,
        file_limit,
        max_concurrent,
        no_generate_costs,
    )
    file_gen_args = EmbeddingFileGenArgs(table_shape, dual_class, pad_min, rebias)

    # run all steps
    start_time = time.time()
    dbdata_dpath = untar_snapshot(
        dbgym_cfg,
        generic_args.pristine_dbdata_snapshot_path,
        generic_args.dbdata_parent_dpath,
    )
    pgbin_path = fully_resolve_path(
        dbgym_cfg, get_default_pgbin_path(dbgym_cfg.dbgym_workspace_path)
    )
    start_postgres(dbgym_cfg, pgbin_path, dbdata_dpath)
    _gen_traindata_dpath(dbgym_cfg, generic_args, dir_gen_args)
    _combine_traindata_dpath_into_parquet(dbgym_cfg, generic_args, file_gen_args)
    datagen_duration = time.time() - start_time
    with open(f"{dbgym_cfg.dbgym_this_run_path}/datagen_time.txt", "w") as f:
        f.write(f"{datagen_duration}")
    stop_postgres(dbgym_cfg, pgbin_path, dbdata_dpath)


def untar_snapshot(
    dbgym_cfg: DBGymConfig, dbdata_snapshot_fpath: Path, dbdata_parent_dpath: Path
) -> Path:
    # It should be an absolute path and it should exist
    assert is_fully_resolved(
        dbdata_snapshot_fpath
    ), f"untar_snapshot(): dbdata_snapshot_fpath ({dbdata_snapshot_fpath}) either doesn't exist or is not absolute"
    save_file(dbgym_cfg, dbdata_snapshot_fpath)
    dbdata_dpath = dbdata_parent_dpath / "dbdata"
    # Make the parent dir and the dbdata dir. Note how we require that dbdata_dpath does not exist while it's ok if the parent does.
    dbdata_parent_dpath.mkdir(parents=True, exist_ok=True)
    if dbdata_dpath.exists():
        shutil.rmtree(dbdata_dpath)
    dbdata_dpath.mkdir(parents=False, exist_ok=False)
    subprocess_run(f"tar -xzf {dbdata_snapshot_fpath} -C {dbdata_dpath}")
    return dbdata_dpath


class EmbeddingDatagenGenericArgs:
    """
    I made Embedding*Args classes to reduce the # of parameters we pass into functions
    I wanted to use classes over dictionaries to enforce which fields are allowed to be present
    I wanted to make multiple classes instead of just one to conceptually separate the different args
    """

    def __init__(
        self,
        benchmark_name: str,
        workload_name: str,
        scale_factor: float,
        benchmark_config_path: Path,
        seed: int,
        workload_path: Path,
        pristine_dbdata_snapshot_path: Path,
        dbdata_parent_dpath: Path,
    ):
        self.benchmark_name = benchmark_name
        self.workload_name = workload_name
        self.scale_factor = scale_factor
        self.benchmark_config_path = benchmark_config_path
        self.seed = seed
        self.workload_path = workload_path
        self.pristine_dbdata_snapshot_path = pristine_dbdata_snapshot_path
        self.dbdata_parent_dpath = dbdata_parent_dpath


class EmbeddingDirGenArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        leading_col_tbls: list[str],
        default_sample_limit: int,
        override_sample_limits: dict[str, int],
        file_limit: int,
        max_concurrent: int,
        no_generate_costs: bool,
    ):
        self.leading_col_tbls = leading_col_tbls
        self.default_sample_limit = default_sample_limit
        self.override_sample_limits = override_sample_limits
        self.file_limit = file_limit
        self.max_concurrent = max_concurrent
        self.no_generate_costs = no_generate_costs


class EmbeddingFileGenArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self, table_shape: bool, dual_class: bool, pad_min: int, rebias: float
    ):
        self.table_shape = table_shape
        self.dual_class = dual_class
        self.pad_min = pad_min
        self.rebias = rebias


def get_traindata_dpath(dbgym_cfg: DBGymConfig) -> Path:
    return dbgym_cfg.cur_task_runs_data_path("traindata", mkdir=True)


def _gen_traindata_dpath(
    dbgym_cfg: DBGymConfig,
    generic_args: EmbeddingDatagenGenericArgs,
    dir_gen_args: EmbeddingDirGenArgs,
) -> None:
    with open_and_save(dbgym_cfg, generic_args.benchmark_config_path, "r") as f:
        benchmark_config = yaml.safe_load(f)

    max_num_columns: int = benchmark_config["protox"]["max_num_columns"]
    tables: list[str] = benchmark_config["protox"]["tables"]
    attributes: TableAttrListMap = benchmark_config["protox"]["attributes"]
    query_spec: QuerySpec = benchmark_config["protox"]["query_spec"]

    workload = Workload(
        dbgym_cfg, tables, attributes, query_spec, generic_args.workload_path, pid=None
    )
    modified_attrs = workload.column_usages()
    traindata_dpath = get_traindata_dpath(dbgym_cfg)

    with Pool(dir_gen_args.max_concurrent) as pool:
        results = []
        job_id = 0
        for tbl in tables:
            cols: list[Optional[str]] = (
                [None]
                if tbl not in dir_gen_args.leading_col_tbls
                else cast(list[Optional[str]], modified_attrs[tbl])
            )
            for colidx, col in enumerate(cols):
                if col is None:
                    output = traindata_dpath / tbl
                else:
                    output = traindata_dpath / tbl / col
                output.mkdir(parents=True, exist_ok=True)

                tbl_sample_limit = dir_gen_args.override_sample_limits.get(
                    tbl, dir_gen_args.default_sample_limit
                )
                num_slices = math.ceil(tbl_sample_limit / dir_gen_args.file_limit)

                for _ in range(0, num_slices):
                    results.append(
                        pool.apply_async(
                            _produce_index_data,
                            args=(
                                dbgym_cfg,
                                tables,
                                attributes,
                                query_spec,
                                generic_args.workload_path,
                                max_num_columns,
                                generic_args.seed,
                                not dir_gen_args.no_generate_costs,
                                min(tbl_sample_limit, dir_gen_args.file_limit),
                                tbl,  # target
                                colidx if col is not None else None,
                                col,
                                job_id,
                                output,
                            ),
                        )
                    )
                    job_id += 1

        pool.close()
        pool.join()

        for result in results:
            result.get()


def _combine_traindata_dpath_into_parquet(
    dbgym_cfg: DBGymConfig,
    generic_args: EmbeddingDatagenGenericArgs,
    file_gen_args: EmbeddingFileGenArgs,
) -> None:
    tbl_dirs = {}
    with open_and_save(dbgym_cfg, generic_args.benchmark_config_path, "r") as f:
        benchmark_config = yaml.safe_load(f)
        benchmark_config = benchmark_config[[k for k in benchmark_config.keys()][0]]
        tables = benchmark_config["tables"]
        for i, tbl in enumerate(tables):
            tbl_dirs[tbl] = i

    traindata_dpath = get_traindata_dpath(dbgym_cfg)
    files = [f for f in Path(traindata_dpath).rglob("*.parquet")]

    def read(file: Path) -> pd.DataFrame:
        tbl = Path(file).parts[-2]
        if tbl not in tbl_dirs:
            tbl = Path(file).parts[-3]
        df = pd.read_parquet(file)
        df["tbl_index"] = tbl_dirs[tbl]

        if file_gen_args.pad_min is not None:
            if df.shape[0] < file_gen_args.pad_min:
                df = pd.concat([df] * int(file_gen_args.pad_min / df.shape[0]))
        return df

    df = pd.concat(map(read, files))

    if "reference_cost" in df.columns:
        target_cost = df.target_cost

        # This expression is the improvement expression.
        act_cost = df.reference_cost - (df.table_reference_cost - target_cost)
        mult = df.reference_cost / act_cost
        rel = (df.reference_cost - act_cost) / act_cost
        mult_tbl = df.table_reference_cost / target_cost
        rel_tbl = (df.table_reference_cost - target_cost) / target_cost

        if file_gen_args.table_shape:
            df["quant_mult_cost_improvement"] = quantile_transform(
                mult_tbl.to_numpy().reshape(-1, 1),
                n_quantiles=100000,
                subsample=df.shape[0],
            )
            df["quant_rel_cost_improvement"] = quantile_transform(
                rel_tbl.to_numpy().reshape(-1, 1),
                n_quantiles=100000,
                subsample=df.shape[0],
            )
        else:
            df["quant_mult_cost_improvement"] = quantile_transform(
                mult.to_numpy().reshape(-1, 1),
                n_quantiles=min(100000, df.shape[0]),
                subsample=df.shape[0],
            )
            df["quant_rel_cost_improvement"] = quantile_transform(
                rel.to_numpy().reshape(-1, 1),
                n_quantiles=min(100000, df.shape[0]),
                subsample=df.shape[0],
            )

        df.drop(
            columns=["reference_cost", "table_reference_cost", "target_cost"],
            inplace=True,
            errors="ignore",
        )

    if file_gen_args.dual_class:
        df["real_idx_class"] = df["idx_class"]
        df["idx_class"] = df["real_idx_class"] * df.col0.max() + df.col1

    df.drop(columns=["table"], inplace=True)
    df.fillna(0, inplace=True)
    # Only int-ify non-cost columns.
    columns = [
        c
        for c in df.columns
        if c not in COST_COLUMNS and "idx_class" not in c and "cmd" != c
    ]
    df[columns] = df[columns].astype(int)

    if file_gen_args.rebias > 0:
        groups = (
            df.groupby(by=["tbl_index", "idx_class"])
            .quant_mult_cost_improvement.describe()
            .sort_values(by=["max"], ascending=False)
        )
        datum = []
        cur_bias = 1.0
        sep_bias = file_gen_args.rebias
        for g in groups.itertuples():
            d = df[
                (df.tbl_index == g.Index[0])  # type: ignore
                & (df.idx_class == g.Index[1])  # type: ignore
                & (df.quant_mult_cost_improvement >= g._6)
            ].copy()
            d["quant_mult_cost_improvement"] = cur_bias - (file_gen_args.rebias / 2)
            datum.append(d)
            cur_bias -= sep_bias
        df = pd.concat(datum, ignore_index=True)

    traindata_path = dbgym_cfg.cur_task_runs_data_path(
        mkdir=True
    ) / get_default_traindata_fname(
        generic_args.benchmark_name, generic_args.workload_name
    )
    df.to_parquet(traindata_path)
    link_result(dbgym_cfg, traindata_path)


_INDEX_SERVER_COUNTS: dict[str, int] = {}


def _fetch_server_indexes(connection: psycopg.Connection[Any]) -> None:
    global _INDEX_SERVER_COUNTS
    query = """
        SELECT t.relname as table_name, i.relname as index_name
        FROM pg_class t, pg_class i, pg_index ix
        WHERE t.oid = ix.indrelid
        and i.oid = ix.indexrelid
    """

    r = [r for r in connection.execute(query)]
    _INDEX_SERVER_COUNTS = {}
    for rr in r:
        if rr[0] not in _INDEX_SERVER_COUNTS:
            _INDEX_SERVER_COUNTS[rr[0]] = 0
        _INDEX_SERVER_COUNTS[rr[0]] += 1


# FUTURE(oltp)
# def load_ou_models(dbgym_cfg: DBGymConfig, model_dir):
#     models = {}
#     for f in Path(model_dir).rglob("*.pkl"):
#         ou_name = str(f.parts[-1]).split(".")[0]
#         with open_and_save(dbgym_cfg, f, "rb") as model:
#             models[ou_name] = pickle.load(model)
#     return models


def _write(data: list[dict[str, Any]], output_dir: Path, batch_num: int) -> None:
    df = pd.DataFrame(data)
    cols = [c for c in df.columns if "col" in c and "str" not in c]
    df[cols] = df[cols].astype(int)
    df.to_parquet(output_dir / f"{batch_num}.parquet")
    del df


def _augment_query_data(workload: Workload, data: dict[str, float]) -> dict[str, float]:
    for qstem, value in workload.queries_mix.items():
        if qstem in data:
            data[qstem] *= value
    return data


def _execute_explains(
    cursor: psycopg.Cursor[Any], batches: QueryBatches, models: Optional[dict[Any, Any]]
) -> dict[str, float]:
    data: dict[str, float] = {}
    ou_model_data: dict[str, list[Any]] = {}

    def acquire_model_data(q: str, plan: dict[str, Any]) -> None:
        nonlocal ou_model_data
        node_tag = plan["Node Type"]
        node_tag = node_tag.replace(" ", "")
        if node_tag == "ModifyTable":
            assert "Operation" in plan
            node_tag = {
                "Insert": "ModifyTableInsert",
                "Update": "ModifyTableUpdate",
                "Delete": "ModifyTableDelete",
            }[plan["Operation"]]
        elif node_tag == "Aggregate":
            node_tag = "Agg"
        elif node_tag == "NestedLoop":
            node_tag = "NestLoop"

        if node_tag == "ModifyTableInsert" or node_tag == "ModifyTableUpdate":
            assert "Relation Name" in plan
            global _INDEX_SERVER_COUNTS
            tbl_name = plan["Relation Name"]
            num_indexes = _INDEX_SERVER_COUNTS.get(tbl_name, 0)

            if num_indexes > 0:
                if "ModifyTableIndexInsert" not in ou_model_data:
                    ou_model_data["ModifyTableIndexInsert"] = []

                for _ in range(num_indexes):
                    ou_model_data["ModifyTableIndexInsert"].append(
                        {
                            "startup_cost": 0,
                            "total_cost": 0,
                            "q": q,
                        }
                    )

        if node_tag not in ou_model_data:
            ou_model_data[node_tag] = []

        d = {"q": q}
        for k, v in plan.items():
            if k == "Plan" or k == "Plans":
                if isinstance(v, dict):
                    acquire_model_data(q, v)
                else:
                    assert isinstance(v, list)
                    for vv in v:
                        acquire_model_data(q, vv)
            else:
                d[k] = v
        d.update({"startup_cost": d["Startup Cost"], "total_cost": d["Total Cost"]})
        ou_model_data[node_tag].append(d)

    for q, sqls, tbl_aliases in batches:
        data[q] = 0.0
        for qtype, sql in sqls:
            if qtype != QueryType.SELECT and qtype != QueryType.INS_UPD_DEL:
                cursor.execute(sql)
            else:
                ssql = "EXPLAIN (FORMAT JSON) {sql}".format(sql=sql)
                explain = [r for r in cursor.execute(ssql, prepare=False)][0][0]
                if models is None:
                    data[q] += explain[0]["Plan"]["Total Cost"]
                else:
                    acquire_model_data(q, explain[0]["Plan"])

    # FUTURE(oltp)
    # if models is not None and len(ou_model_data) > 0:
    #     holistic_results = []
    #     for ou_type, ou_data in ou_model_data.items():
    #         if ou_type in models:
    #             df = pd.DataFrame(ou_data)
    #             df = clean_input_data(df, separate_indkey_features=False, is_train=True)
    #             preds = pd.Series(models[ou_type].predict(df).reshape(-1), name="elapsed_us")
    #             holistic_results.append(pd.concat([preds, df.q], axis=1))
    #         else:
    #             continue

    #     holistic_results = pd.concat(holistic_results).reset_index()
    #     holistic_results = holistic_results.groupby(by=["q"]).sum().reset_index()
    #     for t in holistic_results.itertuples():
    #         if t.q not in data:
    #             data[t.q] = 0
    #         data[t.q] += t.elapsed_us
    return data


def _extract_refs(
    generate_costs: bool,
    target: Optional[str],
    cursor: psycopg.Cursor[Any],
    workload: Workload,
    models: Optional[dict[Any, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    ref_qs = {}
    table_ref_qs = {}
    if generate_costs:
        # Get reference costs.
        batches = QueryBatches(
            [
                (q, workload.queries[q], workload.query_aliases[q])
                for q in workload.queries.keys()
            ]
        )
        ref_qs = _execute_explains(cursor, batches, models)
        ref_qs = _augment_query_data(workload, ref_qs)

        # Get reference costs specific to the table.
        if target is None:
            table_ref_qs = ref_qs
        else:
            qs = workload.queries_for_table(target)
            batches = QueryBatches(
                [(q, workload.queries[q], workload.query_aliases[q]) for q in qs]
            )
            table_ref_qs = _execute_explains(cursor, batches, models)
            table_ref_qs = _augment_query_data(workload, table_ref_qs)
    return ref_qs, table_ref_qs


def _produce_index_data(
    dbgym_cfg: DBGymConfig,
    tables: list[str],
    attributes: TableAttrListMap,
    query_spec: QuerySpec,
    workload_path: Path,
    max_num_columns: int,
    seed: int,
    generate_costs: bool,
    sample_limit: int,
    target: Optional[str],
    leading_col: Optional[int],
    leading_col_name: Optional[str],
    p: int,
    output: Path,
) -> None:
    models = None
    # FUTURE(oltp)
    # if model_dir is not None:
    #     models = load_ou_models(model_dir)

    # Construct workload.
    workload = Workload(dbgym_cfg, tables, attributes, query_spec, workload_path, pid=p)
    modified_attrs = workload.column_usages()

    np.random.seed(seed)
    random.seed(seed)

    # TODO: In theory we want to bias the sampling towards longer length.
    # Since the options grow exponentially from there...
    idxs = IndexSpace(
        tables,
        max_num_columns,
        max_indexable_attributes=workload.max_indexable(),
        seed=seed,
        rel_metadata=copy.deepcopy(modified_attrs),
        attributes_overwrite=copy.deepcopy(modified_attrs),
        tbl_include_subsets=TableAttrAccessSetsMap({}),
        index_space_aux_type=False,
        index_space_aux_include=False,
        deterministic_policy=False,
    )

    table_idx = 0
    if target is not None:
        for i, tbl in enumerate(tables):
            if tbl == target:
                table_idx = i
                break

        if len(modified_attrs[target]) == 0:
            # there are no indexes to generate.
            return

    with create_psycopg_conn() as connection:
        _fetch_server_indexes(connection)
        if generate_costs:
            try:
                connection.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
            except:
                pass

        with connection.cursor() as cursor:
            reference_qs, table_reference_qs = _extract_refs(
                generate_costs, target, cursor, workload, models
            )
            accum_data: list[dict[str, Any]] = []

            # Repeatedly...
            for i in range(sample_limit):
                if (i % 1024) == 0:
                    logging.getLogger(DBGYM_LOGGER_NAME).info(
                        f"{target} {leading_col_name} {p} progress update: {i} / {sample_limit}."
                    )

                act = idxs.sample(
                    mask={
                        "table_idx": None if target is None else table_idx,
                        "col_idx": leading_col,
                    }
                )
                ia = idxs.to_action(act)

                accum: dict[str, Any] = {
                    "table": ia.tbl_name,
                }
                if generate_costs:
                    index_size = 0
                    # Only try to build if we actually need the cost information.
                    ia = idxs.to_action(act)
                    cmds = []
                    if ia.is_valid:
                        # Always try to add the index.
                        cmds = [ia.sql(add=True)]

                    if len(cmds) > 0:
                        # Use hypopg to create the index.
                        r = [
                            r
                            for r in cursor.execute(
                                f"SELECT * FROM hypopg_create_index('{cmds[0]}')"
                            )
                        ]
                        if len(r) == 0:
                            assert False

                        global _INDEX_SERVER_COUNTS
                        if ia.tbl_name not in _INDEX_SERVER_COUNTS:
                            _INDEX_SERVER_COUNTS[ia.tbl_name] = 0
                        _INDEX_SERVER_COUNTS[ia.tbl_name] += 1

                        indexrelid = r[0][0]
                        if models is None:
                            qs_for_tbl = workload.queries_for_table_col(
                                ia.tbl_name, ia.columns[0]
                            )
                        else:
                            qs_for_tbl = workload.queries_for_table(ia.tbl_name)

                        batches = QueryBatches(
                            [
                                (q, workload.queries[q], workload.query_aliases[q])
                                for q in qs_for_tbl
                            ]
                        )
                        data = _execute_explains(cursor, batches, models)
                        data = _augment_query_data(workload, data)
                        if models is None:
                            if len(data) != len(table_reference_qs):
                                # Fold the stuff we aren't aware of.
                                for k, v in table_reference_qs.items():
                                    if k not in data:
                                        data[k] = v
                            assert set(data.keys()) == set(table_reference_qs.keys())
                        else:
                            assert len(data) == len(table_reference_qs)

                        _INDEX_SERVER_COUNTS[ia.tbl_name] -= 1

                        # Get the index size.
                        index_size = [
                            r
                            for r in cursor.execute(
                                f"SELECT * FROM hypopg_relation_size({indexrelid})"
                            )
                        ][0][0]
                        cursor.execute(f"SELECT hypopg_drop_index({indexrelid})")
                        accum["cmd"] = cmds[0]

                    accum_elem = {
                        "reference_cost": np.sum([v for v in reference_qs.values()]),
                        "table_reference_cost": np.sum(
                            [v for v in table_reference_qs.values()]
                        ),
                        "target_cost": np.sum([v for v in data.values()]),
                        "index_size": index_size,
                    }
                    accum.update(accum_elem)

                # Put a bias on the fact that 0 is a "stop"/"invalid" token.
                for i in range(max_num_columns):
                    accum[f"col{i}"] = 0

                assert ia.col_idxs is not None
                for i, col_idx in enumerate(ia.col_idxs):
                    accum[f"col{i}"] = col_idx + 1

                # Fetch and install the class.
                idx_class = idxs.get_index_class(act)
                assert idx_class != "-1"
                accum["idx_class"] = int(idx_class)
                accum_data.append(accum)

            if len(accum_data) > 0:
                _write(accum_data, output, p)
                gc.collect()
                gc.collect()

    # Log that we finished.
    logging.getLogger(DBGYM_LOGGER_NAME).info(
        f"{target} {p} progress update: {sample_limit} / {sample_limit}."
    )
