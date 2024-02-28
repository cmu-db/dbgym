import click
import math
from itertools import chain, combinations
import pandas as pd
import numpy as np
import gc
import yaml
from pathlib import Path
import psycopg

from tune.protox.env.workload import Workload
from tune.protox.env.workload_utils import QueryType
from tune.protox.env.space.index_space import IndexSpace, IndexRepr
from multiprocessing import Pool
import random
import numpy as np
import os
import time

# FUTURE(oltp)
# try:
#     sys.path.append("/home/wz2/noisepage-pilot")
#     from behavior.utils.prepare_ou_data import clean_input_data
# except:
#     pass

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

INDEX_SERVER_COUNTS = {}

def fetch_server_indexes(connection):
    global INDEX_SERVER_COUNTS
    query = """
        SELECT t.relname as table_name, i.relname as index_name
        FROM pg_class t, pg_class i, pg_index ix
        WHERE t.oid = ix.indrelid
        and i.oid = ix.indexrelid
    """

    r = [r for r in connection.execute(query)]
    INDEX_SERVER_COUNTS = {}
    for rr in r:
        if rr[0] not in INDEX_SERVER_COUNTS:
            INDEX_SERVER_COUNTS[rr[0]] = 0
        INDEX_SERVER_COUNTS[rr[0]] += 1

# FUTURE(oltp)
# def load_ou_models(model_dir):
#     models = {}
#     for f in Path(model_dir).rglob("*.pkl"):
#         ou_name = str(f.parts[-1]).split(".")[0]
#         with open(f, "rb") as model:
#             models[ou_name] = pickle.load(model)
#     return models

def write(data, output_dir, batch_num):
    df = pd.DataFrame(data)
    cols = [c for c in df if "col" in c and "str" not in c]
    df[cols] = df[cols].astype(int)
    df.to_parquet(f"{output_dir}/{batch_num}.parquet")
    del df


def augment_query_data(workload, data):
    for qstem, value in workload.queries_mix.items():
        if qstem in data:
            data[qstem] *= value
    return data


def execute_explains(cursor, batches, models):
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
            global INDEX_SERVER_COUNTS
            tbl_name = plan["Relation Name"]
            num_indexes = INDEX_SERVER_COUNTS.get(tbl_name, 0)

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


def extract_refs(generate_costs, target, cursor, workload, models):
    ref_qs = {}
    table_ref_qs = {}
    if generate_costs:
        # Get reference costs.
        batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in workload.queries.keys()]
        ref_qs = execute_explains(cursor, batches, models)
        ref_qs = augment_query_data(workload, ref_qs)

        # Get reference costs specific to the table.
        if target is None:
            table_ref_qs = ref_qs
        else:
            qs = workload.check_queries_for_table(target)
            batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in qs]
            table_ref_qs = execute_explains(cursor, batches, models)
            table_ref_qs = augment_query_data(workload, table_ref_qs)
    return ref_qs, table_ref_qs


def produce_index_data(
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
        fetch_server_indexes(connection)
        idxs.reset(connection=connection)
        if generate_costs:
            try:
                connection.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
            except:
                pass

        with connection.cursor() as cursor:
            reference_qs, table_reference_qs = extract_refs(generate_costs, target, cursor, workload, models)
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

                        global INDEX_SERVER_COUNTS
                        if ia.tbl_name not in INDEX_SERVER_COUNTS:
                            INDEX_SERVER_COUNTS[ia.tbl_name] = 0
                        INDEX_SERVER_COUNTS[ia.tbl_name] += 1

                        indexrelid = r[0][0]
                        if models is None:
                            qs_for_tbl = workload.check_queries_for_table_col(ia.tbl_name, ia.columns[0])
                        else:
                            qs_for_tbl = workload.check_queries_for_table(ia.tbl_name)

                        batches = [(q, workload.queries[q], workload.query_aliases[q]) for q in qs_for_tbl]
                        data = execute_explains(cursor, batches, models)
                        data = augment_query_data(workload, data)
                        if models is None:
                            if len(data) != len(table_reference_qs):
                                # Fold the stuff we aren't aware of.
                                for k, v in table_reference_qs.items():
                                    if k not in data:
                                        data[k] = v
                            assert set(data.keys()) == set(table_reference_qs.keys())
                        else:
                            assert len(data) == len(table_reference_qs)

                        INDEX_SERVER_COUNTS[ia.tbl_name] -= 1

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
                write(accum_data, Path(output), p)
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


@click.command
def datagen(args):
    leading_col_tbls = args.per_leading_col.split(",")
    seed = args.seed if args.seed else int(random.uniform(0, 1e6))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    batch_limit = str(args.batch_limit)
    if "," in batch_limit:
        batch_limit = [int(bl) for bl in batch_limit.split(",")]
    else:
        batch_limit = int(batch_limit)

    benchmark_config = args.benchmark_config
    with open(benchmark_config, "r") as f:
        benchmark_config = yaml.safe_load(f)

    max_num_columns = benchmark_config["mythril"]["max_num_columns"]
    tables = benchmark_config["mythril"]["tables"]
    attributes = benchmark_config["mythril"]["attributes"]
    query_spec = benchmark_config["mythril"]["query_spec"]

    workload = Workload(tables, attributes, query_spec, pid=None)
    modified_attrs = workload.process_column_usage()

    start_time = time.time()
    with Pool(args.num_processes) as pool:
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

                limit = None
                if isinstance(batch_limit, list):
                    num_slices = math.ceil(batch_limit[tbli] / args.sample_limit)
                    limit = batch_limit[tbli]
                else:
                    num_slices = math.ceil(batch_limit / args.sample_limit)
                    limit = batch_limit

                for _ in range(0, num_slices):
                    results.append(pool.apply_async(
                        produce_index_data,
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
                            min(limit, args.sample_limit),
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
