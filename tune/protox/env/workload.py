import copy
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pglast
import pglast.ast
from plumbum import local

from misc.utils import open_and_save
from tune.protox.env import is_binary_enum, is_knob_enum, regress_ams, regress_qid_knobs
from tune.protox.env.space.utils import fetch_server_knobs
from tune.protox.env.workload_utils import (
    QueryType,
    acquire_metrics_around_query,
    execute_serial_variations,
    extract_aliases,
    extract_columns,
    extract_sqltypes,
    parse_access_method,
)


class Workload(object):
    benchbase: bool = True
    allow_per_query: bool = False
    workload_eval_mode: str = "all"
    workload_eval_inverse: bool = False
    tbl_include_subsets_prune: bool = False

    early_workload_kill: bool = False
    workload_timeout: float = 0
    workload_timeout_penalty: float = 1.0

    order: list[str] = None
    queries: Dict[str, List[Tuple[QueryType, str]]] = None
    queries_mix: Dict[str, float] = None
    query_aliases: dict[str, List[str]] = None
    tbl_queries_usage: dict[str, set[str]] = None
    tbl_filter_queries_usage: dict[str, set[str]] = None

    readonly_workload = False
    # Map order[i] -> (per-query-knobs, metric)
    best_observed = {}

    logger = None

    def _crunch(self, all_attributes, sqls, pid):
        self.order = []
        self.queries = {}
        self.tbl_queries_usage = {}
        self.tbl_filter_queries_usage = {}

        # Build the SQL and table usage information.
        self.queries_mix = {}
        self.query_aliases = {}
        self.query_usages = {t: [] for t in self.tables}
        tbl_include_subsets = {tbl: set() for tbl in self.attributes.keys()}

        self.tbl_wheres = {tbl: set() for tbl in self.attributes.keys()}

        sql_mapping = {}
        for stem, sql_file, ratio in sqls:
            assert stem not in self.queries
            self.order.append(stem)
            self.queries_mix[stem] = ratio
            sql_mapping[stem] = sql_file

            with open_and_save(self.dbgym_cfg, sql_file, "r") as q:
                sql = q.read()
                assert not sql.startswith("/*")

                # TODO(WAN): HACK HACK HACK
                if Path(sql_file).name == "15.sql" and "benchmark_tpch" in str(
                    Path(sql_file).absolute()
                ):
                    sql = sql.replace("revenue0", "revenue0_PID")

                stmts = pglast.parse_sql(sql)

                # Extract aliases.
                self.query_aliases[stem] = extract_aliases(stmts)

                # Extract sql and query types.
                self.queries[stem] = extract_sqltypes(stmts, pid)

                # Construct table query usages.
                for tbl in self.query_aliases[stem]:
                    if tbl not in self.tbl_queries_usage:
                        self.tbl_queries_usage[tbl] = set()
                    self.tbl_queries_usage[tbl].add(stem)

                for stmt in stmts:
                    tbl_col_usages, all_refs = extract_columns(
                        stmt, self.tables, all_attributes, self.query_aliases[stem]
                    )
                    tbl_col_usages = {
                        t: [a for a in atts if a in self.attributes[t]]
                        for t, atts in tbl_col_usages.items()
                    }
                    for tbl, atts in tbl_col_usages.items():
                        for att in atts:
                            if (tbl, att) not in self.tbl_filter_queries_usage:
                                self.tbl_filter_queries_usage[(tbl, att)] = set()
                            self.tbl_filter_queries_usage[(tbl, att)].add(stem)

                            if att not in self.query_usages[tbl]:
                                self.query_usages[tbl].append(att)

                        if len(atts) > 0:
                            self.tbl_wheres[tbl].add(tuple(sorted(set(atts))))

                    all_refs = set(all_refs)
                    all_qref_sets = set([r[0] for r in all_refs])
                    all_qref_sets = {
                        k: tuple(sorted([r[1] for r in all_refs if r[0] == k]))
                        for k in all_qref_sets
                    }
                    for k, s in all_qref_sets.items():
                        tbl_include_subsets[k].add(s)

        for k, v in self.tbl_wheres.items():
            self.tbl_wheres[k] = [
                k
                for k, kk in zip(
                    v, [not any(set(v0) <= set(v1) for v1 in v if v0 != v1) for v0 in v]
                )
                if kk
            ]

        # Do this so query_usages is actually in the right order.
        self.query_usages = {
            tbl: [a for a in atts if a in self.query_usages[tbl]]
            for tbl, atts in self.attributes.items()
        }

        if self.tbl_include_subsets_prune:
            self.tbl_include_subsets = {}
            for k, v in tbl_include_subsets.items():
                self.tbl_include_subsets[k] = [
                    k
                    for k, kk in zip(
                        v,
                        [
                            not any(set(v0) <= set(v1) for v1 in v if v0 != v1)
                            for v0 in v
                        ],
                    )
                    if kk
                ]

            if self.tbl_fold_subsets:
                tbl_include_subsets = copy.deepcopy(self.tbl_include_subsets)
                for tbl, subsets in tbl_include_subsets.items():
                    subsets = sorted(subsets, key=lambda x: len(x))
                    for _ in range(self.tbl_fold_iterations):
                        for i in range(len(subsets)):
                            s0 = set(subsets[i])
                            for j in range(i + 1, len(subsets)):
                                s1 = set(subsets[j])
                                if len(s0 - s1) <= self.tbl_fold_delta:
                                    subsets[i] = None
                                    subsets[j] = tuple(
                                        sorted(set(subsets[j]).union(s0))
                                    )
                        subsets = [s for s in subsets if s is not None]
                        subsets = sorted(subsets, key=lambda x: len(x))
                    self.tbl_include_subsets[tbl] = subsets
        else:
            self.tbl_include_subsets = tbl_include_subsets

        self.readonly_workload = not any(
            [
                q == QueryType.INS_UPD_DEL
                for _, sqls in self.queries.items()
                for (q, _) in sqls
            ]
        )
        self.sql_files = {k: str(v) for (k, v, _) in sqls}

    def __init__(
        self,
        dbgym_cfg,
        tables: list[str],
        attributes: dict[str, list[str]],
        query_spec: dict,
        workload_path: Path,
        pid=None,
        workload_eval_mode="all",
        workload_eval_inverse=False,
        workload_timeout=0,
        workload_timeout_penalty=1.0,
        logger=None,
    ):

        self.dbgym_cfg = dbgym_cfg
        self.workload_eval_mode = workload_eval_mode
        self.workload_eval_inverse = workload_eval_inverse
        # Whether we should use benchbase or not.
        self.benchbase = query_spec["benchbase"]
        self.allow_per_query = query_spec["allow_per_query"]
        self.early_workload_kill = query_spec["early_workload_kill"]
        self.workload_timeout = workload_timeout
        self.workload_timeout_penalty = workload_timeout_penalty
        self.tbl_include_subsets_prune = query_spec.get(
            "tbl_include_subsets_prune", False
        )

        self.tbl_fold_subsets = query_spec.get("tbl_fold_subsets", False)
        self.tbl_fold_delta = query_spec.get("tbl_fold_delta", False)
        self.tbl_fold_iterations = query_spec.get("tbl_fold_iterations", False)

        self.logger = logger
        if self.logger is not None:
            self.logger.info(f"Initialized with workload timeout {workload_timeout}")

        self.tables = tables
        self.attributes = attributes

        # Mapping from attribute -> table that has it.
        all_attributes = {}
        for tbl, cols in self.attributes.items():
            for col in cols:
                if col not in all_attributes:
                    all_attributes[col] = []
                all_attributes[col].append(tbl)

        # Get the order in which we should execute in.
        sqls = []
        workload_order_file = workload_path / "order.txt"
        with open_and_save(self.dbgym_cfg, workload_order_file, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                contents = line.split(",")
                if len(contents) == 2:
                    key, sqlfile, weight = contents[0], contents[1], 1
                elif len(contents) == 3:
                    key, sqlfile, weight = contents[0], contents[1], float(contents[2])
                else:
                    raise RuntimeError(
                        f"Invalid workload order.txt: {workload_order_file}"
                    )
                sqls.append((key, sqlfile, weight))

        self._crunch(all_attributes, sqls, pid)

        query_usages = copy.deepcopy(self.query_usages)
        tbl_include_subsets = copy.deepcopy(self.tbl_include_subsets)

        if "execute_query_order" in query_spec:
            # TODO(WAN): can this be folded into workload_path?
            with open_and_save(self.dbgym_cfg, query_spec["execute_query_order"], "r") as f:
                lines = f.read().splitlines()
                sqls = [
                    (
                        line.split(",")[0],
                        Path(query_spec["execute_query_directory"])
                        / line.split(",")[1],
                        1,
                    )
                    for line in lines
                ]

            # Re-crunch with the new data.
            self._crunch(all_attributes, sqls, pid)
            self.query_usages = query_usages
            self.tbl_include_subsets = tbl_include_subsets

    def process_column_usage(self):
        return self.query_usages

    def reset(
        self,
        env_spec=None,
        connection=None,
        reward_utility=None,
        query_timeout=None,
        accum_metric=None,
    ):
        if connection is not None and accum_metric is not None and self.allow_per_query:
            # Construct an empty action.
            action = (
                env_spec.action_space.get_knob_space().get_state(None).copy(),
                env_spec.action_space.get_index_space().null_action(),
            )

            # Replace per-query with the "known best".
            for qid, (qid_set, _) in self.best_observed.items():
                for knob, val in qid_set:
                    action[0][knob.name()] = val
            assert env_spec.action_space.contains(action)

            results = f"{env_spec.benchbase_path}/results{env_spec.postgres_port}"
            shutil.rmtree(results, ignore_errors=True)

            # Determine whether the "known best" is good or not.
            success, mutilated, q_timeout, accum_metric = self._execute_psycopg(
                env_spec,
                results,
                connection,
                action,
                query_timeout,
                reset_eval=True,
                reset_accum_metric=accum_metric,
            )
            assert success

            # Forcefully reconstruct based on the output.
            args = {
                "connection": connection,
                "results": results,
                "action": mutilated if mutilated is not None else action,
            }
            state = env_spec.observation_space.construct_offline(**args)
            metric, _ = reward_utility(
                result_dir=results, update=False, did_error=False
            )
            env_spec.action_space.get_knob_space().advance(
                args["action"][0], connection=connection, workload=self
            )
            logging.debug(
                f"[permute_via_reset]: {env_spec.action_space.get_knob_space().get_state(None)}"
            )
            return state, metric

        return None, None

    def set_workload_timeout(self, metric):
        if self.workload_timeout == 0:
            self.workload_timeout = metric
        else:
            self.workload_timeout = min(self.workload_timeout, metric)
        logging.info(f"Workload timeout set to: {self.workload_timeout}")

    def check_queries_for_table(self, table):
        return [q for q in self.order if q in self.tbl_queries_usage[table]]

    def check_queries_for_table_col(self, table, col):
        if (table, col) not in self.tbl_filter_queries_usage:
            return []
        return [
            q for q in self.order if q in self.tbl_filter_queries_usage[(table, col)]
        ]

    def _execute_benchbase(self, env_spec, results):
        with local.cwd(f"{env_spec.benchbase_path}"):
            code, _, _ = local["java"][
                "-jar",
                "benchbase.jar",
                "-b",
                env_spec.benchmark_name,
                "-c",
                env_spec.benchbase_config_path,
                "-d",
                results,
                "--execute=true",
            ].run(retcode=None)

            if code != 0:
                return False
        return True

    def _execute_workload(
        self,
        connection,
        workload_timeout,
        ql_knobs={},
        output_file=None,
        workload_qdir=None,
        env_spec=None,
        disable_pg_hint=False,
        blocklist=[],
    ):
        # Get the knobs.
        real_knobs = {}
        if env_spec is not None:
            real_knobs = fetch_server_knobs(
                connection,
                tables=env_spec.action_space.get_knob_space().tables,
                knobs=env_spec.action_space.get_knob_space().knobs,
            )

        workload_time = 0
        time_left = workload_timeout

        if workload_qdir is not None and workload_qdir[0] is not None:
            workload_qdir, workload_qlist = workload_qdir
            with open_and_save(self.dbgym_cfg, workload_qlist, "r") as f:
                psql_order = [
                    (f"Q{i+1}", workload_qdir / l.strip())
                    for i, l in enumerate(f.readlines())
                ]

            actual_order = [p[0] for p in psql_order]
            actual_sql_files = {k: str(v) for (k, v) in psql_order}
            actual_queries = {}
            for qid, qpat in psql_order:
                with open_and_save(self.dbgym_cfg, qpat, "r") as f:
                    query = f.read()
                actual_queries[qid] = [(QueryType.SELECT, query)]
        else:
            actual_order = self.order
            actual_sql_files = self.sql_files
            actual_queries = self.queries

        for qid in actual_order:
            queries = actual_queries[qid]
            if any([b in actual_sql_files[qid] for b in blocklist]):
                continue

            for sql_type, query in queries:
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    connection.execute(query)
                    continue

                if time_left > 0:
                    # Relevant qid knobs.
                    qid_knobs = [
                        ql_knobs[knob] for knob in ql_knobs.keys() if f"{qid}_" in knob
                    ]

                    undo_disable = None
                    if disable_pg_hint:
                        # Alter the session first.
                        disable = ";".join(
                            [
                                f"SET {knob.knob_name} = OFF"
                                for (knob, value) in qid_knobs
                                if value == 0
                            ]
                        )
                        connection.execute(disable)

                        undo_disable = ";".join(
                            [
                                f"SET {knob.knob_name} = ON"
                                for (knob, value) in qid_knobs
                                if value == 0
                            ]
                        )
                    else:
                        query = (
                            "/*+ "
                            + " ".join(
                                [
                                    knob.resolve_per_query_knob(
                                        value, all_knobs=real_knobs
                                    )
                                    for (knob, value) in qid_knobs
                                ]
                            )
                            + " */"
                            + query
                        )

                    if output_file is not None:
                        query = "EXPLAIN (ANALYZE, TIMING OFF) " + query

                    _, qid_runtime, query_timeout, explain = acquire_metrics_around_query(
                        "", None, connection, qid, query, time_left, metrics=False
                    )

                    if disable_pg_hint:
                        # Now undo the session.
                        connection.execute(undo_disable)

                    time_left -= qid_runtime / 1e6
                    workload_time += qid_runtime / 1e6

                    if output_file is not None and explain is not None:
                        pqkk = [(knob.name(), val) for (knob, val) in qid_knobs]
                        output_file.write(f"{qid}\n")
                        output_file.write(f"PerQuery: {pqkk}\n")
                        output_file.write("\n".join(explain))
                        output_file.write("\n")
                        output_file.write("\n")
        return workload_time

    def _execute_psycopg(
        self,
        env_spec,
        results,
        connection,
        action,
        query_timeout,
        reset_eval=False,
        reset_accum_metric=None,
        baseline=False,
    ):
        # Values of knobs that were really utilized.
        ks = env_spec.action_space.get_knob_space()
        real_knobs = fetch_server_knobs(
            connection, tables=ks.tables if ks else [], knobs=ks.knobs if ks else {}
        )
        # Prior knob stae.
        all_knobs = ks.get_state(None) if ks else {}
        # Current action's per-query knobs.
        ql_knobs = (
            env_spec.action_space.get_query_level_knobs(action)
            if action is not None
            else {}
        )

        # Setup the mutilated action.
        mutilated = None
        need_metric = env_spec.observation_space.metrics()

        query_explain_data = []
        qid_runtime_data = {}
        running_time = 0
        stop_running = False
        for qid_index, qid in enumerate(self.order):
            if stop_running:
                # Break out if we should stop running.
                # Revert the queries that we don't know about...

                # This should be back to the "previous might be good" -- or the selected one at least...
                # Previous might not actually be good -- but better than an unknown current one that we
                # have no intuition about i guess...
                for qqid_index in range(qid_index, len(self.order)):
                    # Relevant qid knobs.
                    qid = self.order[qqid_index]
                    qid_knobs = [
                        ql_knobs[knob] for knob in ql_knobs.keys() if f"{qid}_" in knob
                    ]
                    qid_global = [
                        (knob, all_knobs[knob.name()]) for (knob, _) in qid_knobs
                    ]
                    for knob, val in qid_global:
                        action[env_spec.action_space.knob_space_ind][knob.name()] = val
                    mutilated = action

                    queries = self.queries[qid]
                    for sql_type, query in queries:
                        assert sql_type != QueryType.UNKNOWN
                        if sql_type != QueryType.SELECT:
                            assert sql_type != QueryType.INS_UPD_DEL
                            connection.execute(query)
                break

            # Relevant qid knobs.
            qid_knobs = [
                ql_knobs[knob] for knob in ql_knobs.keys() if f"{qid}_" in knob
            ]
            inverse_knobs = [
                (knob, knob.invert(val))
                for knobname, (knob, val) in ql_knobs.items()
                if f"{qid}_" in knobname
            ]
            # Rebuild the previous global per-query knob setting.
            qid_global = [(knob, all_knobs[knob.name()]) for (knob, _) in qid_knobs]

            queries = self.queries[qid]
            qid_runtime = 0
            for qidx, (sql_type, query) in enumerate(queries):
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    connection.execute(query)
                    continue

                # Construct the baseline! by regressing it into knob form.
                ams, explain = self._parse_single_access_method(
                    connection, qid, ignore=True
                )
                qid_default = regress_qid_knobs(qid_knobs, real_knobs, ams, explain)

                # Start time.
                start_time = time.time()

                # Construct runs.
                runs = []
                if not reset_eval:
                    if (
                        self.workload_eval_mode in ["prev_dual", "all", "all_enum"]
                        and len(qid_global) > 0
                    ):
                        # If reset_eval, then we basically want to evaluate with what we are tracking.
                        # In the reset_eval case, tracking is the previous known-best!
                        runs.append(("PrevDual", qid_global))

                    if self.workload_eval_mode in [
                        "global_dual",
                        "all",
                        "all_enum",
                    ] or (
                        self.workload_eval_mode == "prev_dual" and len(qid_global) == 0
                    ):
                        if len(runs) == 0 or [v[1] for v in qid_global] != [
                            v[1] for v in qid_default
                        ]:
                            # Run only if the knobs don't match prev dual.
                            runs.append(("GlobalDual", qid_default))

                    if (
                        self.workload_eval_mode == "all_enum"
                        and len(qid_knobs) > 0
                        and any([is_knob_enum(k) for k, _ in qid_knobs])
                    ):
                        # Run the "complete" binary version.
                        top_enums = []
                        bottom_enums = []
                        for k, v in qid_knobs:
                            if not is_knob_enum(k):
                                top_enums.append((k, v))
                                bottom_enums.append((k, v))
                            elif is_binary_enum(k):
                                top_enums.append((k, 1))
                                bottom_enums.append((k, 0))
                            else:
                                top_enums.append((k, int(k.sample_uniform())))
                                bottom_enums.append((k, int(k.sample_uniform())))
                        runs.append(("TopEnum", top_enums))
                        runs.append(("BottomEnum", bottom_enums))

                    if self.workload_eval_mode == "default":
                        # No flags.
                        runs.append(("Default", []))

                if len(qid_knobs) > 0:
                    # This with a sleight of hand is always the action under investigation.
                    runs.append(("PerQuery", qid_knobs))
                    if self.workload_eval_inverse and not reset_eval:
                        runs.append(("PerQueryInverse", inverse_knobs))
                elif len(runs) == 0:
                    runs.append(("GlobalDual", qid_default))

                assert len(runs) > 0

                if (
                    reset_eval
                    and [v[1] for v in qid_global] == [v[1] for v in qid_knobs]
                    and qid in reset_accum_metric
                ):
                    # Case where the per-query and global match each other.
                    assert qid in reset_accum_metric
                    best_metric, best_time, best_query_timeout = reset_accum_metric[qid]
                    runs_idx = ("PrevDual", qid_global, False)

                    # Note that we skipped.
                    logging.debug(f"reset_eval re-using prior computation for {qid}")
                else:
                    (
                        best_metric,
                        best_time,
                        best_query_timeout,
                        best_explain_data,
                        runs_idx,
                    ) = execute_serial_variations(
                        env_spec=env_spec,
                        connection=connection,
                        query_timeout=min(query_timeout, self.workload_timeout - running_time + 1),
                        logger=self.logger,
                        qid=qid,
                        query=query,
                        runs=runs,
                        real_knobs=real_knobs,
                    )

                if not reset_eval and best_explain_data is not None:
                    query_explain_data.append(
                        (qid, runs_idx[0], runs_idx[1], best_explain_data)
                    )

                # Compare against the old if we have a record available.
                # Note that PerQuery refers to the best observed record -- PrevDual is best of state.
                if (
                    reset_eval
                    and runs_idx[0] == "PerQuery"
                    and qid in reset_accum_metric
                ):
                    # Note this is a little awkward if we're doing "best-of" ness.
                    # Old record is PrevDual which is the state being restored.
                    # PerQuery in action refers to the best observed that is being tested.

                    # Preventing old record from drifting and pre-maturely timing out -- we only run the PerQuery
                    # and see how well it compares to (saved) PrevDual. Hopefully the cache warm-ness doesn't burn
                    # us too much here.

                    # If the "new" timed out or if "old" is better, use the old metric.
                    _, old_best_time, _ = reset_accum_metric[qid]
                    if best_query_timeout or old_best_time < best_time:
                        best_metric, best_time, best_query_timeout = reset_accum_metric[qid]
                        runs_idx = ("PrevDual", qid_global)

                assert qid not in qid_runtime_data
                assert best_metric is not None or (not need_metric)
                assert runs_idx[0] in [
                    "TopEnum",
                    "BottomEnum",
                    "Default",
                    "PrevDual",
                    "GlobalDual",
                    "PerQuery",
                    "PerQueryInverse",
                ]
                if reset_eval:
                    assert runs_idx[0] in ["PrevDual", "PerQuery"]
                qid_runtime_data[qid] = {
                    "start": start_time,
                    "runtime": best_time,
                    "metric": best_metric,
                    "query_timeout": best_query_timeout,
                    "prefix": runs_idx[0],
                }

                if runs_idx[0] == "PrevDual" and not best_query_timeout:
                    # Overwrite the action decision with the prior decision.
                    for knob, val in qid_global:
                        action[env_spec.action_space.knob_space_ind][knob.name()] = val
                    mutilated = action

                    if reset_eval:
                        logging.debug(f"best_observe action deemed not valuable.")
                elif runs_idx[0] == "GlobalDual" and not best_query_timeout:
                    assert not reset_eval
                    for knob, val in qid_default:
                        action[env_spec.action_space.knob_space_ind][knob.name()] = val
                    mutilated = action
                elif runs_idx[0] == "PerQueryInverse" and not best_query_timeout:
                    assert not reset_eval
                    for knob, val in inverse_knobs:
                        action[env_spec.action_space.knob_space_ind][knob.name()] = val
                    mutilated = action
                elif runs_idx[0] in ["TopEnum", "BottomEnum"] and not best_query_timeout:
                    assert not reset_eval
                    enum_knobs = runs_idx[1]

                    # Get the actual access method utilized.
                    pqk_query = (
                        "/*+ "
                        + " ".join(
                            [
                                knob.resolve_per_query_knob(value, all_knobs=real_knobs)
                                for (knob, value) in enum_knobs
                            ]
                        )
                        + " */"
                        + query
                    )
                    pqk_query = "EXPLAIN (FORMAT JSON) " + pqk_query
                    ams, explain = self._parse_query_am(connection, pqk_query)
                    # Regress the access methods.
                    qid_enum = regress_ams(enum_knobs, ams, explain)

                    for knob, val in qid_enum:
                        action[env_spec.action_space.knob_space_ind][knob.name()] = val
                    mutilated = action

                if (
                    qid not in self.best_observed
                    or (best_time < self.best_observed[qid][1] and not best_query_timeout)
                    or reset_eval
                ):
                    # Knobs that are actually set.
                    qid_set = [
                        (
                            knob,
                            action[env_spec.action_space.knob_space_ind][knob.name()],
                        )
                        for (knob, _) in qid_knobs
                    ]
                    self.best_observed[qid] = (qid_set, best_time)
                    logging.debug(
                        f"[best_observe] {qid}: {best_time / 1.0e6} ({reset_eval})"
                    )

                # Break if we've exceeded the minimum time.
                # Note that runtime is in microseconds.
                running_time += qid_runtime_data[qid]["runtime"] / 1.0e6
                if (
                    self.early_workload_kill
                    and self.workload_timeout > 0
                    and running_time > self.workload_timeout
                ):
                    logging.info("Aborting workload early.")
                    # Don't penalize for now since it should anyways have negative reward.
                    stop_running = True

                    for qqidx in range(qidx + 1, len(queries)):
                        # Finish the remaining maintenance queries.
                        sql_type, query = queries[qqidx]
                        assert sql_type != QueryType.UNKNOWN
                        if sql_type != QueryType.SELECT:
                            assert sql_type != QueryType.INS_UPD_DEL
                            connection.execute(query)
                    break

        # Get the timeouts flag.
        query_timeouts = [v["query_timeout"] for _, v in qid_runtime_data.items()]
        # Get the accumulated metrics.
        accum_metric = {
            q: (v["metric"], v["runtime"], v["query_timeout"])
            for q, v in qid_runtime_data.items()
        }

        results_dir = Path(results)
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "run.plans", "w") as f:
            for qid, prefix, pqk, explain in query_explain_data:
                pqkk = [(knob.name(), val) for (knob, val) in pqk]

                f.write(f"{qid}\n")
                f.write(f"{prefix}: {pqkk}\n")
                f.write(json.dumps(explain))
                f.write("\n")
                f.write("\n")

        if need_metric:
            accum_data = [v["metric"] for _, v in qid_runtime_data.items()]
            accum_stats = env_spec.observation_space.merge_data(accum_data)

            with open(results_dir / "run.metrics.json", "w") as f:

                def flatten(d):
                    flat = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            flat[k] = flatten(v)
                        elif isinstance(v, np.ndarray):
                            flat[k] = float(v[0])
                        elif isinstance(v, np.ScalarType):
                            if isinstance(v, str):
                                flat[k] = v
                            else:
                                flat[k] = float(v)
                        else:
                            flat[k] = v
                    return flat

                output = flatten(accum_stats)
                output["flattened"] = True
                f.write(json.dumps(output, indent=4))

        with open(results_dir / "run.raw.csv", "w") as f:
            f.write(
                "Transaction Type Index,Transaction Name,Start Time (microseconds),Latency (microseconds),Worker Id (start number),Phase Id (index in config file)\n"
            )
            for i, qid in enumerate(self.order):
                if qid in qid_runtime_data:
                    start = qid_runtime_data[qid]["start"]
                    run = qid_runtime_data[qid]["runtime"]
                    prefix = qid_runtime_data[qid]["prefix"]
                    f.write(f"{i+1},{qid},{start},{run},0,{prefix}\n")

            if stop_running and self.workload_timeout_penalty > 1:
                # Get the penalty.
                penalty = (
                    self.workload_timeout * self.workload_timeout_penalty - running_time
                )
                if not baseline:
                    # Always make it a bit worse...
                    penalty = (penalty + 1.05) * 1e6
                else:
                    penalty = penalty * 1e6
            elif stop_running:
                if not baseline:
                    # Always degrade it a little if we've timed out.
                    penalty = 3e6
                else:
                    penalty = 0
            else:
                penalty = 0

            if penalty > 0:
                f.write(f"{len(self.order)},P,{time.time()},{penalty},0,PENALTY\n")

        return True, mutilated, (any(query_timeouts) or stop_running), accum_metric

    def execute(
        self,
        connection,
        reward_utility,
        env_spec,
        query_timeout=None,
        action=None,
        current_state=None,
        update=True,
    ):
        success = True
        logging.info("Starting to run benchmark...")

        if self.benchbase:
            # Attach a timeout if necessary to prevent really bad OLAP configs from running.
            if query_timeout is not None and query_timeout > 0:
                connection.execute(
                    f"ALTER SYSTEM SET statement_timeout = {query_timeout * 1000}"
                )
                connection.execute("SELECT pg_reload_conf()")

            # Purge results directory first.
            results = f"{env_spec.benchbase_path}/results{env_spec.postgres_port}"
            shutil.rmtree(results, ignore_errors=True)

            # Execute benchbase if specified.
            success = self._execute_benchbase(env_spec, results)
            if success:
                # We can only create a state if we succeeded.
                args = {
                    "connection": connection,
                    "results": results,
                    "action": action,
                }
                success = env_spec.observation_space.check_benchbase(**args)
                if success:
                    state = env_spec.observation_space.construct_offline(**args)
                else:
                    logging.error("Benchbase failed to produce a valid output result.")
                    assert current_state is not None
                    state = current_state
            else:
                assert current_state is not None
                state = current_state

            metric, reward = None, None
            if reward_utility is not None:
                metric, reward = reward_utility(
                    result_dir=results, update=update, did_error=not success
                )

            if query_timeout is not None and query_timeout > 0:
                connection.execute("ALTER SYSTEM SET statement_timeout = 0")
                connection.execute("SELECT pg_reload_conf()")

            # Benchbase can't mutilate.
            mutilated = None
            q_timeout = False
            accum_metric = None
        elif self.allow_per_query:
            # Purge results directory first.
            results = f"{env_spec.benchbase_path}/results{env_spec.postgres_port}"
            shutil.rmtree(results, ignore_errors=True)
            baseline = current_state is None

            success, mutilated, q_timeout, accum_metric = self._execute_psycopg(
                env_spec, results, connection, action, query_timeout, baseline=baseline
            )
            assert success

            args = {
                "connection": connection,
                "results": results,
                "action": mutilated if mutilated is not None else action,
            }
            state = env_spec.observation_space.construct_offline(**args)
            assert success, logging.error(
                "Invalid benchbase results information created from per-query run."
            )

            metric, reward = None, None
            if reward_utility is not None:
                metric, reward = reward_utility(
                    result_dir=results, update=update, did_error=not success
                )
        else:
            assert False, "Currently don't support not running with benchbase."

        logging.info(
            f"Benchmark iteration with metric {metric} (reward: {reward}) (q_timeout: {q_timeout})"
        )
        return (
            success,
            metric,
            reward,
            results,
            state,
            mutilated,
            q_timeout,
            accum_metric,
        )

    def _parse_query_am(self, connection, query):
        data = [r for r in connection.execute(query)][0][0]
        data = data[0]
        return parse_access_method(data), data

    def _parse_single_access_method(self, connection, qid, ignore=False):
        qams = {}
        explain = None
        queries = self.queries[qid]
        for sql_type, query in queries:
            if sql_type != QueryType.SELECT:
                assert sql_type != QueryType.INS_UPD_DEL
                if not ignore:
                    connection.execute(query)
                continue

            explain = "EXPLAIN (FORMAT JSON) " + query
            qams_delta, explain = self._parse_query_am(connection, explain)
            qams.update(qams_delta)
        return qams, explain

    def parse_all_access_methods(self, connection):
        q_ams = {}
        for qid in self.order:
            q_ams[qid] = self._parse_single_access_method(connection, qid)[0]
        return q_ams

    def save_state(self):
        kv = {}
        for attr in dir(self):
            if (
                attr == "logger"
                or attr.startswith("_")
                or callable(getattr(self, attr))
            ):
                continue
            kv[attr] = getattr(self, attr)
        return kv

    def load_state(self, d):
        for attr in dir(self):
            if attr in d:
                setattr(self, attr, d[attr])
