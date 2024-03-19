import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import pglast # type: ignore
from plumbum import local
from psycopg import Connection

from envs.logger import Logger, time_record
from envs.spaces.holon_space import HolonSpace
from envs.spaces.primitive_spaces import KnobSpace, QuerySpace
from envs.spaces.primitives.knob import CategoricalKnob, Knob
from envs.spaces.latent_spaces import LatentKnobSpace, LatentQuerySpace
from envs.spaces.state.space import StateSpace
from envs.utils.execute import _acquire_metrics_around_query, execute_variations
from envs.utils.postgres import PostgresConn
from envs.utils.reward import RewardUtility
from envs.utils.workload_analysis import (
    extract_aliases,
    extract_columns,
    extract_sqltypes,
)
from envs.types import (
    QueryType,
    KnobSpaceAction,
    QuerySpaceAction,
    QuerySpaceKnobAction,
    HolonAction,
    QueryRun,
    BestQueryRun,
    AttrTableListMap,
    QuerySpec,
    QueryMap,
    TableColTuple,
    TableAttrListMap,
    TableAttrAccessSetsMap,
    TableAttrSetMap,
)


class Workload(object):
    def _crunch(
        self,
        all_attributes: AttrTableListMap,
        sqls: list[Tuple[str, Path, float]],
        pid: Optional[int],
        query_spec: QuerySpec,
    ) -> None:
        do_tbl_include_subsets_prune = query_spec["tbl_include_subsets_prune"]
        self.order = []
        self.queries = QueryMap({})
        # Map table -> set(queries that use it)
        self.tbl_queries_usage: dict[str, set[str]] = {}
        # Map (table, column) -> set(queries that use it)
        self.tbl_filter_queries_usage: dict[TableColTuple, set[str]] = {}

        # Build the SQL and table usage information.
        self.queries_mix = {}
        self.query_aliases = {}
        self.query_usages = TableAttrListMap({t: [] for t in self.tables})
        tbl_include_subsets = TableAttrAccessSetsMap({
            tbl: set() for tbl in self.attributes.keys()
        })
        for stem, sql_file, ratio in sqls:
            assert stem not in self.queries
            self.order.append(stem)
            self.queries_mix[stem] = ratio

            with open(sql_file, "r") as q:
                sql = q.read()
                assert not sql.startswith("/*")

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
                    # Get all columns that appear in the predicates.
                    # Get all columns that appear together (all_refs).
                    tbl_col_usages, all_refs = extract_columns(
                        stmt, self.tables, all_attributes, self.query_aliases[stem]
                    )
                    tbl_col_usages = TableAttrSetMap({
                        t: set([a for a in atts if a in self.attributes[t]])
                        for t, atts in tbl_col_usages.items()
                    })

                    for tbl, atts in tbl_col_usages.items():
                        for att in atts:
                            # Update the (tbl, col) query references.
                            if (tbl, att) not in self.tbl_filter_queries_usage:
                                self.tbl_filter_queries_usage[TableColTuple((tbl, att))] = set()
                            self.tbl_filter_queries_usage[TableColTuple((tbl, att))].add(stem)

                            # Update query_usages (reflects predicate usage).
                            if att not in self.query_usages[tbl]:
                                self.query_usages[tbl].append(att)

                    # Compute table -> unique set of columns used from that table
                    all_qref_sets = {
                        k: tuple(sorted([r[1] for r in set(all_refs) if r[0] == k]))
                        for k in set([r[0] for r in all_refs])
                    }
                    for k, s in all_qref_sets.items():
                        tbl_include_subsets[k].add(s)

        # Do this so query_usages is actually in the right order.
        # Order based on the original attribute list.
        self.query_usages = TableAttrListMap({
            tbl: [a for a in atts if a in self.query_usages[tbl]]
            for tbl, atts in self.attributes.items()
        })

        if do_tbl_include_subsets_prune:
            self.tbl_include_subsets = {}
            # First prune any "fully enclosed".
            for tbl, attrsets in tbl_include_subsets.items():
                self.tbl_include_subsets[tbl] = set(
                    tbl
                    for tbl, not_enclosed in zip(
                        attrsets,
                        [
                            # Basically:
                            # for v0 in attrsets:
                            #   v0_not_enclosed = True
                            #   for v1 in attrsets:
                            #     if v0 <= v1:
                            #       v0_not_enclosed = False
                            not any(set(v0) <= set(v1) for v1 in attrsets if v0 != v1)
                            for v0 in attrsets
                        ],
                    )
                    if not_enclosed
                )

            if query_spec["tbl_fold_subsets"]:
                tis = copy.deepcopy(self.tbl_include_subsets)
                for tbl, subsets in tis.items():
                    # Sort by length...these are not fully enclosed.
                    sorted_subsets = sorted(subsets, key=lambda x: len(x))
                    for _ in range(query_spec["tbl_fold_iterations"]):
                        for i in range(len(sorted_subsets)):
                            s0 = set(sorted_subsets[i])
                            for j in range(i + 1, len(sorted_subsets)):
                                s1 = set(sorted_subsets[j])
                                # If the difference is small enough, merge them.
                                if len(s0 - s1) <= query_spec["tbl_fold_delta"]:
                                    sorted_subsets[i] = tuple()
                                    sorted_subsets[j] = tuple(sorted(s1.union(s0)))
                        # Filter out the sets that no longer exist.
                        sorted_subsets = sorted(
                            [s for s in sorted_subsets if len(s) > 0],
                            key=lambda x: len(x),
                        )
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
        tables: list[str],
        attributes: TableAttrListMap,
        query_spec: QuerySpec,
        pid: Optional[int] = None,
        workload_timeout: float = 0,
        workload_timeout_penalty: float = 1.0,
        logger: Optional[Logger] = None,
    ) -> None:

        # Whether we should use benchbase or not.
        self.benchbase = query_spec["benchbase"]
        self.oltp_workload = query_spec["oltp_workload"]
        self.workload_timeout = workload_timeout
        self.workload_timeout_penalty = workload_timeout_penalty
        self.logger = logger
        if self.logger:
            self.logger.get_logger(__name__).info(
                f"Initialized with workload timeout {workload_timeout}"
            )

        self.tables: list[str] = tables
        self.attributes: TableAttrListMap = attributes

        # Mapping from attribute -> table that has it.
        all_attributes = AttrTableListMap({})
        for tbl, cols in self.attributes.items():
            for col in cols:
                if col not in all_attributes:
                    all_attributes[col] = []
                all_attributes[col].append(tbl)

        # Get the order in which we should execute in.
        sqls = []
        if "query_order" in query_spec:
            with open(query_spec["query_order"], "r") as f:
                lines = f.read().splitlines()
                sqls = [
                    (
                        line.split(",")[0],
                        Path(query_spec["query_directory"]) / line.split(",")[1],
                        1.0,
                    )
                    for line in lines
                ]

        if "query_transactional" in query_spec:
            with open(query_spec["query_transactional"], "r") as f:
                lines = f.read().splitlines()
                splits = [line.split(",") for line in lines]
                sqls = [
                    (
                        split[0],
                        Path(query_spec["query_directory"]) / split[1],
                        float(split[2]),
                    )
                    for split in splits
                ]

        self._crunch(all_attributes, sqls, pid, query_spec)
        query_usages = copy.deepcopy(self.query_usages)
        tbl_include_subsets = copy.deepcopy(self.tbl_include_subsets)

        if "execute_query_order" in query_spec:
            with open(query_spec["execute_query_order"], "r") as f:
                lines = f.read().splitlines()
                sqls = [
                    (
                        line.split(",")[0],
                        Path(query_spec["execute_query_directory"])
                        / line.split(",")[1],
                        1.0,
                    )
                    for line in lines
                ]

            # Re-crunch with the new data.
            self._crunch(all_attributes, sqls, pid, query_spec)
            self.query_usages = query_usages
            self.tbl_include_subsets = tbl_include_subsets

    def set_workload_timeout(self, metric: float) -> None:
        if self.workload_timeout == 0:
            self.workload_timeout = metric
        else:
            self.workload_timeout = min(self.workload_timeout, metric)

        if self.logger:
            self.logger.get_logger(__name__).info(
                f"Workload timeout set to: {self.workload_timeout}"
            )

    def queries_for_table(self, table: str) -> list[str]:
        return [q for q in self.order if q in self.tbl_queries_usage[table]]

    def queries_for_table_col(self, table: str, col: str) -> list[str]:
        if (table, col) not in self.tbl_filter_queries_usage:
            return []
        return [
            q for q in self.order if q in self.tbl_filter_queries_usage[TableColTuple((table, col))]
        ]

    def column_usages(self) -> TableAttrListMap:
        return copy.deepcopy(self.query_usages)

    def max_indexable(self) -> int:
        return max([len(cols) for _, cols in self.query_usages.items()])

    @time_record("execute")
    def _execute_workload(
        self,
        pgconn: PostgresConn,
        actions: list[HolonAction] = [],
        actions_names: list[str] = [],
        results: Optional[Union[str, Path]] = None,
        obs_space: Optional[StateSpace] = None,
        action_space: Optional[HolonSpace] = None,
        reset_metrics: Optional[dict[str, BestQueryRun]] = None,
        override_workload_timeout: Optional[float] = None,
        pqt: Optional[int] = None,
        workload_qdir: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
        disable_pg_hint: bool = False,
        blocklist: list[str] = [],
        first: bool = False,
    ) -> Union[float, Tuple[bool, bool, dict[str, Any]]]:
        workload_timeout = (
            self.workload_timeout
            if not override_workload_timeout
            else override_workload_timeout
        )
        assert len(actions) == len(actions_names)

        # Do we need metrics.
        need_metric = False if not obs_space else obs_space.require_metrics()

        sysknobs = KnobSpaceAction({})
        ql_knobs = []
        if len(actions) > 0:
            assert action_space

            sysknobs = cast(KnobSpaceAction, [
                v
                for t, v in action_space.split_action(actions[0])
                if isinstance(t, LatentKnobSpace)
            ][0])
            ql_knobs = cast(list[Tuple[LatentQuerySpace, QuerySpaceAction]], [
                [
                    (t, v)
                    for t, v in action_space.split_action(action)
                    if isinstance(t, LatentQuerySpace)
                ][0]
                for action in actions
            ])

        # Figure out workload to execute.
        if workload_qdir is not None and workload_qdir[0] is not None:
            # Load actual queries to execute.
            workload_dir, workload_qlist = workload_qdir
            with open(workload_qlist, "r") as f:
                psql_order = [
                    (f"Q{i+1}", Path(workload_dir) / l.strip())
                    for i, l in enumerate(f.readlines())
                ]

            actual_order = [p[0] for p in psql_order]
            actual_sql_files = {k: str(v) for (k, v) in psql_order}
            actual_queries = {}
            for qid, qpat in psql_order:
                with open(qpat, "r") as f:
                    query = f.read()
                actual_queries[qid] = [(QueryType.SELECT, query)]
        else:
            actual_order = self.order
            actual_sql_files = self.sql_files
            actual_queries = self.queries

        # Now let us start executing.
        workload_time = 0.0
        time_left = workload_timeout
        qid_runtime_data = {}
        stop_running = False

        for execute_idx, qid in enumerate(actual_order):
            if stop_running:
                break

            queries = actual_queries[qid]
            if any([b in actual_sql_files[qid] for b in blocklist]):
                # Skip any query in blocklist.
                continue

            for sql_type, query in queries:
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    pgconn.conn().execute(query)
                    continue

                if disable_pg_hint:
                    assert len(ql_knobs) == 1
                    ql_knob = ql_knobs[0]
                    qid_knobs = {
                        ql_knob[0].knobs[k]: ql_knob[1][k]
                        for k in ql_knob[1].keys()
                        if f"{qid}_" in k
                    }

                    # Alter the session first.
                    disable = ";".join(
                        [
                            f"SET {knob.knob_name} = OFF"
                            for knob, value in qid_knobs.items()
                            if value == 0
                        ]
                    )
                    pgconn.conn().execute(disable)

                    qid_runtime, _, _, _ = _acquire_metrics_around_query(
                        self.logger,
                        f"{qid}",
                        pgconn.conn(),
                        query,
                        pqt=time_left,
                        obs_space=None,
                    )

                    undo_disable = ";".join(
                        [
                            f"SET {knob.knob_name} = ON"
                            for knob, value in qid_knobs.items()
                            if value == 0
                        ]
                    )
                    pgconn.conn().execute(undo_disable)

                else:
                    # De-duplicate the runs.
                    runs: list[QueryRun] = []
                    zruns: list[QueryRun] = [
                        QueryRun(
                            act_name,
                            f"{act_name}_{qid}",
                            QuerySpaceKnobAction({
                                ql_knob[0].knobs[k]: ql_knob[1][k]
                                for k in ql_knob[1].keys()
                                if f"{qid}_" in k
                            }),
                        )
                        for ql_knob, act_name in zip(ql_knobs, actions_names)
                    ]
                    for r in zruns:
                        if r[2] not in [rr[2] for rr in runs]:
                            runs.append(r)

                    target_pqt = pqt if pqt else workload_timeout
                    skip_execute = False
                    if (
                        reset_metrics is not None
                        and qid in reset_metrics
                        and not reset_metrics[qid].timeout
                    ):
                        # If we have a reset metric, use it's timeout and convert to seconds.
                        truntime = reset_metrics[qid].runtime
                        assert truntime is not None
                        target_pqt = truntime / 1.e6

                        # If we've seen this exact before, skip it.
                        rmetrics = reset_metrics[qid]
                        skip_execute = (rmetrics.query_run is not None) and (rmetrics.query_run.qknobs is not None) and (rmetrics.query_run.qknobs == runs[-1].qknobs)

                    if not skip_execute:
                        best_run: BestQueryRun = execute_variations(
                            connection=pgconn.conn(),
                            runs=runs,
                            query=query,
                            pqt=min(target_pqt, workload_timeout - workload_time + 1),
                            logger=self.logger,
                            sysknobs=sysknobs,
                            obs_space=obs_space,
                        )
                    else:
                        assert reset_metrics
                        best_run = reset_metrics[qid]

                    if reset_metrics is not None and qid in reset_metrics:
                        # Old one is actually better so let's use that.
                        rmetric = reset_metrics[qid]
                        if best_run.timeout or (best_run.runtime and rmetric.runtime and rmetric.runtime < best_run.runtime):
                            best_run = rmetric

                    assert best_run.runtime
                    qid_runtime_data[qid] = best_run
                    qid_runtime = best_run.runtime

                time_left -= qid_runtime / 1e6
                workload_time += qid_runtime / 1e6
                if time_left < 0:
                    stop_running = True
                    break

        # Undo any necessary state changes.
        for qqid_index in range(execute_idx, len(actual_order)):
            queries = self.queries[qid]
            for sql_type, query in queries:
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    pgconn.conn().execute(query)

        if results is not None:
            # Make the result directory.
            results_dir = Path(results)
            if not results_dir.exists():
                results_dir.mkdir(parents=True, exist_ok=True)

            with open(results_dir / "run.plans", "w") as f:
                # Output the explain data.
                for qid, run in qid_runtime_data.items():
                    if run.explain_data is not None:
                        assert run.query_run and run.query_run.qknobs is not None
                        pqkk = [
                            (knob.name(), val) for knob, val in run.query_run.qknobs.items()
                        ]
                        f.write(f"{qid}\n{run.query_run.prefix}: {pqkk}\n")
                        f.write(json.dumps(run.explain_data))
                        f.write("\n\n")

            if obs_space and obs_space.require_metrics():
                # Create the metrics.
                # Log the metrics data as a flattened.
                accum_data = cast(list[dict[str, Any]], [v.metric_data for _, v in qid_runtime_data.items()])
                accum_stats = obs_space.merge_deltas(accum_data)
                with open(results_dir / "run.metrics.json", "w") as f:
                    # Flatten it.
                    def flatten(d: dict[str, Any]) -> dict[str, Any]:
                        flat: dict[str, Any] = {}
                        for k, v in d.items():
                            if isinstance(v, dict):
                                flat[k] = flatten(v)
                            elif isinstance(v, np.ndarray):
                                flat[k] = float(v[0])
                            elif isinstance(v, np.ScalarType):
                                if isinstance(v, str):
                                    flat[k] = v
                                else:
                                    flat[k] = float(cast(float, v))
                            else:
                                flat[k] = v
                        return flat

                    output = flatten(accum_stats)
                    output["flattened"] = True
                    f.write(json.dumps(output, indent=4))

            with open(results_dir / "run.raw.csv", "w") as f:
                # Write the raw query data.
                f.write(
                    "Transaction Type Index,Transaction Name,Start Time (microseconds),Latency (microseconds),Worker Id (start number),Phase Id (index in config file)\n"
                )

                start = 0.
                for i, qid in enumerate(self.order):
                    if qid in qid_runtime_data:
                        data = qid_runtime_data[qid]
                        assert data and data.runtime and data.query_run
                        rtime = data.runtime
                        pfx = data.query_run.prefix
                        f.write(f"{i+1},{qid},{start},{rtime},0,{pfx}\n")
                        start += rtime / 1e6

                # Write a penalty term if needed.
                penalty = 0.0
                if stop_running and self.workload_timeout_penalty > 1:
                    # Get the penalty.
                    penalty = (
                        workload_timeout * self.workload_timeout_penalty - workload_time
                    )
                    penalty = (penalty + 1.05) * 1e6 if not first else penalty * 1e6
                elif stop_running and not first:
                    # Always degrade it a little if we've timed out.
                    penalty = 3.0e6

                if penalty > 0:
                    f.write(f"{len(self.order)},P,{time.time()},{penalty},0,PENALTY\n")

            # Get all the timeouts.
            timeouts = [v.timeout for _, v in qid_runtime_data.items()]
            return True, (any(timeouts) or stop_running), qid_runtime_data

        return workload_time

    @time_record("execute")
    def _execute_benchbase(
        self, benchbase_config: dict[str, Any], results: Union[str, Path]
    ) -> bool:
        bb_path = benchbase_config["benchbase_path"]
        with local.cwd(bb_path):
            code, _, _ = local["java"][
                "-jar",
                "benchbase.jar",
                "-b",
                benchbase_config["benchmark"],
                "-c",
                benchbase_config["benchbase_config_path"],
                "-d",
                results,
                "--execute=true",
            ].run(retcode=None)

            if code != 0:
                return False
        return True

    def execute(
        self,
        pgconn: PostgresConn,
        reward_utility: RewardUtility,
        obs_space: StateSpace,
        action_space: HolonSpace,
        actions: list[HolonAction],
        actions_names: list[str],
        benchbase_config: dict[str, Any],
        pqt: Optional[int] = None,
        reset_metrics: Optional[dict[str, BestQueryRun]] = None,
        update: bool = True,
        first: bool = False,
    ) -> Tuple[bool, float, float, Union[str, Path], bool, dict[str, BestQueryRun]]:
        success = True
        if self.logger:
            self.logger.get_logger(__name__).info("Starting to run benchmark...")

        # Purge results directory first.
        assert "benchbase_path" in benchbase_config
        bb_path = benchbase_config["benchbase_path"]
        results = f"{bb_path}/results{pgconn.postgres_port}"
        shutil.rmtree(results, ignore_errors=True)

        if self.benchbase:
            # Execute benchbase if specified.
            success = self._execute_benchbase(benchbase_config, results)
            # We can only create a state if we succeeded.
            success = obs_space.check_benchbase(results)
        else:
            ret = self._execute_workload(
                pgconn,
                actions=actions,
                actions_names=actions_names,
                results=results,
                obs_space=obs_space,
                action_space=action_space,
                reset_metrics=reset_metrics,
                override_workload_timeout=self.workload_timeout,
                pqt=pqt,
                workload_qdir=None,
                disable_pg_hint=False,
                blocklist=[],
                first=first,
            )
            assert isinstance(ret, tuple)
            success, q_timeout, query_metric_data = ret[0], ret[1], ret[2]
            assert success

        metric, reward = None, None
        if reward_utility is not None:
            metric, reward = reward_utility(
                result_dir=results, update=update, did_error=not success
            )

        if self.logger:
            self.logger.get_logger(__name__).info(
                f"Benchmark iteration with metric {metric} (reward: {reward}) (q_timeout: {q_timeout})"
            )
        return success, metric, reward, results, q_timeout, query_metric_data
