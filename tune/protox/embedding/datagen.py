import click
import math
from itertools import chain, combinations
import pandas as pd
import numpy as np
import gc
import yaml
from pathlib import Path
import psycopg
from multiprocessing import Pool
import random
import numpy as np
import os
import time

from tune.protox.env.workload import Workload
from tune.protox.env.workload_utils import QueryType
from tune.protox.env.space.index_space import IndexSpace, IndexRepr

from misc.utils import open_and_save, default_benchmark_config_relpath

# FUTURE(oltp)
# try:
#     sys.path.append("/home/wz2/noisepage-pilot")
#     from behavior.utils.prepare_ou_data import clean_input_data
# except:
#     pass

def _all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

_INDEX_SERVER_COUNTS = {}

def _fetch_server_indexes(connection):
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
# def load_ou_models(model_dir):
#     models = {}
#     for f in Path(model_dir).rglob("*.pkl"):
#         ou_name = str(f.parts[-1]).split(".")[0]
#         with open(f, "rb") as model:
#             models[ou_name] = pickle.load(model)
#     return models

def _write(data, output_dir, batch_num):
    df = pd.DataFrame(data)
    cols = [c for c in df if "col" in c and "str" not in c]
    df[cols] = df[cols].astype(int)
    df.to_parquet(f"{output_dir}/{batch_num}.parquet")
    del df


def _augment_query_data(workload, data):
    for qstem, value in workload.queries_mix.items():
        if qstem in data:
            data[qstem] *= value
    return data


def _execute_explains(cursor, batches, models):
    data = {}
    ou_model_data = {}
    def acquire_model_data(q, plan):
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
                    ou_model_data["ModifyTableIndexInsert"].append({
                        "startup_cost": 0,
                        "total_cost": 0,
                        "q": q,
                    })

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
        for (qtype, sql) in sqls:
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


def _extract_refs(generate_costs, target, cursor, workload, models):
    ref_qs = {}
    table_ref_qs = {}
    if generate_costs:
        # Get reference costs.
        batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in workload.queries.keys()]
        ref_qs = _execute_explains(cursor, batches, models)
        ref_qs = _augment_query_data(workload, ref_qs)

        # Get reference costs specific to the table.
        if target is None:
            table_ref_qs = ref_qs
        else:
            qs = workload.check_queries_for_table(target)
            batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in qs]
            table_ref_qs = _execute_explains(cursor, batches, models)
            table_ref_qs = _augment_query_data(workload, table_ref_qs)
    return ref_qs, table_ref_qs


def _produce_index_data(
    config_path,
    benchmark_config,
    connection,
    tables,
    attributes,
    query_spec,
    max_num_columns,
    seed,
    generate_costs,
    model_dir,
    sample_limit,
    target,
    leading_col,
    leading_col_name,
    truncate_target,
    p,
    output):

    models = None
    # FUTURE(oltp)
    # if model_dir is not None:
    #     models = load_ou_models(model_dir)

    # Construct workload.
    workload = Workload(tables, attributes, query_spec, pid=str(p))
    modified_attrs = workload.process_column_usage()

    seed = (os.getpid() * int(time.time())) % 123456789
    np.random.seed(seed)
    random.seed(seed)

    # TODO: In theory we want to bias the sampling towards longer length.
    # Since the options grow exponentially from there...
    idxs = IndexSpace(
        "wolp",
        tables,
        max_num_columns,
        IndexRepr.ONE_HOT.name,
        seed=seed,
        latent_dim=0,
        attributes_overwrite=modified_attrs)

    table_idx = 0
    if target is not None:
        for i, tbl in enumerate(tables):
            if tbl == target:
                table_idx = i
                break

        if len(modified_attrs[target]) == 0:
            # there are no indexes to generate.
            return

    with psycopg.connect(connection, autocommit=True, prepare_threshold=None) as connection:
        _fetch_server_indexes(connection)
        idxs.reset(connection=connection)
        if generate_costs:
            try:
                connection.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
            except:
                pass

        with connection.cursor() as cursor:
            reference_qs, table_reference_qs = _extract_refs(generate_costs, target, cursor, workload, models)
            cached_refs = {}
            accum_data = []

            # Repeatedly...
            for i in range(sample_limit):
                if (i % 1024) == 0:
                    print(f"{target} {leading_col_name} {p} progress update: {i} / {sample_limit}.")

                act = idxs.random_action_table(None if target is None else table_idx, leading_col, truncate_target)
                ia = idxs.construct_indexaction(act)

                accum = { "table": ia.tbl_name, }
                if generate_costs:
                    index_size = 0
                    # Only try to build if we actually need the cost information.
                    ia = idxs.construct_indexaction(act)
                    cmds = []
                    if ia.is_valid:
                        # Always try to add the index.
                        cmds = [ia.sql(add=True)]

                    if len(cmds) > 0:
                        # Use hypopg to create the index.
                        r = [r for r in cursor.execute(f"SELECT * FROM hypopg_create_index('{cmds[0]}')")]
                        if len(r) == 0:
                            print(cmds)
                            assert False

                        global _INDEX_SERVER_COUNTS
                        if ia.tbl_name not in _INDEX_SERVER_COUNTS:
                            _INDEX_SERVER_COUNTS[ia.tbl_name] = 0
                        _INDEX_SERVER_COUNTS[ia.tbl_name] += 1

                        indexrelid = r[0][0]
                        if models is None:
                            qs_for_tbl = workload.check_queries_for_table_col(ia.tbl_name, ia.columns[0])
                        else:
                            qs_for_tbl = workload.check_queries_for_table(ia.tbl_name)

                        batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in qs_for_tbl]
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
                        index_size = [r for r in cursor.execute(f"SELECT * FROM hypopg_relation_size({indexrelid})")][0][0]
                        cursor.execute(f"SELECT hypopg_drop_index({indexrelid})")
                        accum["cmd"] = cmds[0]

                    accum_elem = {
                        "reference_cost": np.sum([v for v in reference_qs.values()]),
                        "table_reference_cost": np.sum([v for v in table_reference_qs.values()]),
                        "target_cost": np.sum([v for v in data.values()]),
                        "index_size":  index_size,
                    }
                    accum.update(accum_elem)

                # Put a bias on the fact that 0 is a "stop"/"invalid" token.
                for i in range(max_num_columns):
                    accum[f"col{i}"] = 0

                for i, col_idx in enumerate(ia.col_idxs):
                    accum[f"col{i}"] = (col_idx + 1)

                # Fetch and install the class.
                idx_class = idxs.get_index_class(act)
                assert idx_class != "-1"
                accum["idx_class"] = int(idx_class)
                accum_data.append(accum)

            if len(accum_data) > 0:
                _write(accum_data, Path(output), p)
                gc.collect()
                gc.collect()
    # Log that we finished.
    print(f"{target} {p} progress update: {sample_limit} / {sample_limit}.")


def create_datagen_parser(subparser):
    parser = subparser.add_parser("generate")
    parser.add_argument("--config", type=Path, default="configs/config.yaml")
    parser.add_argument("--benchmark-config", type=Path, default="configs/benchmark/tpch.yaml")
    parser.add_argument("--generate-costs", default=False, action="store_true")
    parser.add_argument("--sample-limit", default=100000, type=int)
    parser.add_argument("--batch-limit", type=str)
    parser.add_argument("--num-processes", default=1, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--connection", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per-table", default=False, action="store_true")
    parser.add_argument("--table", default=None, type=str)
    parser.add_argument("--per-leading-col", default="", type=str)
    parser.add_argument("--model-dir", default=None, type=Path)
    parser.add_argument("--truncate-target", default=None, type=int)
    parser.set_defaults(func=datagen)


# click setup
@click.command()
@click.pass_context

# args
@click.argument("benchmark")
@click.option("--benchmark-config-path", default=None, type=str, help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath(BENCHMARK_PLACEHOLDER)}.")
@click.option("--leading-col-tbls", default=None, type=str, help="If the table is in --leading-col-tbls, it is # of indexes to sample per column for that table table. If the table is in --leading-col-tbls, it is the # of indexes to sample total for that table.")
# TODO(wz2): what if we sample tbl_batch_limit / len(cols) for tables in leading_col_tbls? this way, tbl_batch_limit will always represent the total # of indexes created on that table. currently the description of the param is a bit weird as you can see
@click.option("--batch-limits", default="2048", type=str, help="If the table is in --leading-col-tbls, it is # of indexes to sample per column for that table table. If the table is in --leading-col-tbls, it is the # of indexes to sample total for that table.")
@click.option("--max-concurrent", default=None, type=int, help="The max # of concurrent threads that will be creating hypothetical indexes. The default is `nproc`.")
@click.option("--seed", default=None, type=int, help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.")

def datagen(ctx, benchmark, benchmark_config_path, batch_limits, max_concurrent, seed):
    '''
    Samples the effects of indexes on the workload as estimated by HypoPG.
    Outputs all this data as a .parquet file in the run_*/ dir.
    Updates the symlink in the data/ dir to point to the new .parquet file.
    '''
    # set args to defaults programmatically (do this BEFORE creating Embedding*Args objects)
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_relpath(benchmark)
    if max_concurrent == None:
        max_concurrent = os.cpu_count()
    if seed == None:
        seed = random.randint(0, 1e8)

    # process the "array-like" args
    leading_col_tbls = [] if leading_col_tbls == None else leading_col_tbls.split(",")
    batch_limits = str(batch_limits)
    if "," in batch_limits:
        batch_limits = [int(bl) for bl in batch_limits.split(",")]
    else:
        batch_limits = int(batch_limits)

    # function start
    with open_and_save(ctx, benchmark_config_path, "r") as f:
        benchmark_config = yaml.safe_load(f)

    max_num_columns = benchmark_config["mythril"]["max_num_columns"]
    tables = benchmark_config["mythril"]["tables"]
    attributes = benchmark_config["mythril"]["attributes"]
    query_spec = benchmark_config["mythril"]["query_spec"]

    workload = Workload(tables, attributes, query_spec, pid=None)
    modified_attrs = workload.process_column_usage()

    start_time = time.time()
    with Pool(max_concurrent) as pool:
        results = []
        if args.per_table:
            tbls = tables
        else:
            assert args.table is not None
            tbls = args.table.split(",")
            for tbl in tbls:
                assert tbl in tables

        job_id = 0
        for tbli, tbl in enumerate(tbls):
            cols = [None] if tbl not in leading_col_tbls else modified_attrs[tbl]
            for colidx, col in enumerate(cols):
                if col is None:
                    output = args.output_dir / tbl
                else:
                    output = args.output_dir / tbl / col
                Path(output).mkdir(parents=True, exist_ok=True)

                tbl_batch_limit = None
                if isinstance(batch_limits, list):
                    num_slices = math.ceil(batch_limits[tbli] / args.sample_limit)
                    tbl_batch_limit = batch_limits[tbli]
                else:
                    num_slices = math.ceil(batch_limits / args.sample_limit)
                    tbl_batch_limit = batch_limits

                for _ in range(0, num_slices):
                    results.append(pool.apply_async(
                        _produce_index_data,
                        args=(
                            args.config,
                            args.benchmark_config,
                            args.connection,
                            tables,
                            attributes,
                            query_spec,
                            max_num_columns,
                            seed,
                            args.generate_costs,
                            args.model_dir,
                            min(tbl_batch_limit, args.sample_limit),
                            tbl, # target
                            colidx if col is not None else None,
                            col,
                            args.truncate_target,
                            job_id,
                            output)))
                    job_id += 1

        pool.close()
        pool.join()

        for result in results:
            result.get()

    duration = time.time() - start_time
    with open(f"{args.output_dir}/time.txt", "w") as f:
        f.write(f"{duration}")