import math
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast
import tempfile
import numpy as np
import pglast  # type: ignore
from plumbum import local

from misc.utils import DBGymConfig, open_and_save
from tune.protox.env.logger import Logger, time_record
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.latent_space import LatentKnobSpace, LatentQuerySpace
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.types import (
    AttrTableListMap,
    BestQueryRun,
    HolonAction,
    KnobSpaceAction,
    QueryMap,
    QueryRun,
    QuerySpaceAction,
    QuerySpaceKnobAction,
    QuerySpec,
    QueryType,
    TableAttrAccessSetsMap,
    TableAttrListMap,
    TableAttrSetMap,
    TableColTuple,
)
from tune.protox.env.util.execute import (
    _acquire_metrics_around_query,
    execute_variations,
)
from tune.protox.env.util.pg_conn import PostgresConn
from tune.protox.env.util.reward import RewardUtility
from tune.protox.env.util.workload_analysis import (
    extract_aliases,
    extract_columns,
    extract_sqltypes,
)


class Workload(object):
    # Usually, we want to call open_and_save() when opening a file for reading
    # However, when creating a Workload object for unittesting, we just want to call open()
    def _open_for_reading(
        self,
        path,
        mode="r",
    ):
        # when opening for writing we always use open() so we don't need this function, which is
        # why we assert here
        # I still chose to make mode an argument just to make the interface identical to open()/open_and_save()
        assert mode == "r"
        if self.dbgym_cfg != None:
            return open_and_save(self.dbgym_cfg, path)
        else:
            return open(path)

    def _crunch(
        self,
        all_attributes: AttrTableListMap,
        sqls: list[Tuple[str, Path, float]],
        pid: Optional[int],
        query_spec: QuerySpec,
    ) -> None:
        assert all(sql[1].exists() and not sql[1].is_symlink() and sql[1].is_absolute() for sql in sqls), f"sqls ({sqls}) should only contain existent real absolute paths."
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
        tbl_include_subsets = TableAttrAccessSetsMap(
            {tbl: set() for tbl in self.attributes.keys()}
        )
        for stem, sql_file, ratio in sqls:
            assert stem not in self.queries
            self.order.append(stem)
            self.queries_mix[stem] = ratio

            with self._open_for_reading(sql_file, "r") as q:
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
                    # Get all columns that appear in the predicates.
                    # Get all columns that appear together (all_refs).
                    tbl_col_usages, all_refs = extract_columns(
                        stmt, self.tables, all_attributes, self.query_aliases[stem]
                    )
                    tbl_col_usages = TableAttrSetMap(
                        {
                            t: set([a for a in atts if a in self.attributes[t]])
                            for t, atts in tbl_col_usages.items()
                        }
                    )

                    for tbl, atts in tbl_col_usages.items():
                        for att in atts:
                            # Update the (tbl, col) query references.
                            if (tbl, att) not in self.tbl_filter_queries_usage:
                                self.tbl_filter_queries_usage[
                                    TableColTuple((tbl, att))
                                ] = set()
                            self.tbl_filter_queries_usage[
                                TableColTuple((tbl, att))
                            ].add(stem)

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
        self.query_usages = TableAttrListMap(
            {
                tbl: [a for a in atts if a in self.query_usages[tbl]]
                for tbl, atts in self.attributes.items()
            }
        )

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
        dbgym_cfg: DBGymConfig,
        tables: list[str],
        attributes: TableAttrListMap,
        query_spec: QuerySpec,
        workload_path: Path,
        pid: Optional[int] = None,
        workload_timeout: float = 0,
        workload_timeout_penalty: float = 1.0,
        logger: Optional[Logger] = None,
    ) -> None:

        self.dbgym_cfg = dbgym_cfg
        self.workload_path = workload_path
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
        order_or_txn_fname = "txn.txt" if self.oltp_workload else "order.txt"
        workload_order_or_txn_fpath = self.workload_path / order_or_txn_fname
        with self._open_for_reading(workload_order_or_txn_fpath, "r") as f:
            lines = f.read().splitlines()
            sqls = [
                (
                    line.split(",")[0],
                    Path(line.split(",")[1]),
                    1.0,
                )
                for line in lines
            ]

        # TODO(phw2): pass "query_transactional" somewhere other than query_spec, just like "query_order" is
        if "query_transactional" in query_spec:
            with self._open_for_reading(query_spec["query_transactional"], "r") as f:
                lines = f.read().splitlines()
                splits = [line.split(",") for line in lines]
                sqls = [
                    (
                        split[0],
                        Path(split[1]),
                        float(split[2]),
                    )
                    for split in splits
                ]

        self._crunch(all_attributes, sqls, pid, query_spec)
        query_usages = copy.deepcopy(self.query_usages)
        tbl_include_subsets = copy.deepcopy(self.tbl_include_subsets)

        # TODO(phw2): pass "execute_query_order" somewhere other than query_spec, just like "query_order" is
        if "execute_query_order" in query_spec:
            with open_and_save(dbgym_cfg, query_spec["execute_query_order"], "r") as f:
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
            q
            for q in self.order
            if q in self.tbl_filter_queries_usage[TableColTuple((table, col))]
        ]

    def column_usages(self) -> TableAttrListMap:
        return copy.deepcopy(self.query_usages)

    def max_indexable(self) -> int:
        return max([len(cols) for _, cols in self.query_usages.items()])

    @staticmethod
    def compute_total_workload_runtime(qid_runtime_data: dict[str, BestQueryRun]) -> float:
        return sum(best_run.runtime for best_run in qid_runtime_data.values()) / 1.0e6

    @time_record("execute")
    def execute_workload(
        self,
        pg_conn: PostgresConn,
        actions: list[HolonAction] = [],
        variation_names: list[str] = [],
        results: Optional[Union[str, Path]] = None,
        observation_space: Optional[StateSpace] = None,
        action_space: Optional[HolonSpace] = None,
        reset_metrics: Optional[dict[str, BestQueryRun]] = None,
        override_workload_timeout: Optional[float] = None,
        query_timeout: Optional[int] = None,
        workload_qdir: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
        blocklist: list[str] = [],
        first: bool = False,
    ) -> Tuple[int, bool, dict[str, Any]]:
        this_execution_workload_timeout = (
            self.workload_timeout
            if not override_workload_timeout
            else override_workload_timeout
        )
        assert len(actions) == len(variation_names)

        sysknobs = KnobSpaceAction({})
        ql_knobs = []
        if len(actions) > 0:
            assert action_space

            sysknobs = cast(
                KnobSpaceAction,
                [
                    v
                    for t, v in action_space.split_action(actions[0])
                    if isinstance(t, LatentKnobSpace)
                ][0],
            )
            ql_knobs = cast(
                list[Tuple[LatentQuerySpace, QuerySpaceAction]],
                [
                    [
                        (t, v)
                        for t, v in action_space.split_action(action)
                        if isinstance(t, LatentQuerySpace)
                    ][0]
                    for action in actions
                ],
            )
        
        # Figure out workload to execute.
        if workload_qdir is not None and workload_qdir[0] is not None:
            # Load actual queries to execute.
            workload_dir, workload_qlist = workload_qdir
            with self._open_for_reading(workload_qlist, "r") as f:
                psql_order = [
                    (f"Q{i+1}", Path(workload_dir) / l.strip())
                    for i, l in enumerate(f.readlines())
                ]

            actual_order = [p[0] for p in psql_order]
            actual_sql_files = {k: str(v) for (k, v) in psql_order}
            actual_queries = {}
            for qid, qpat in psql_order:
                with self._open_for_reading(qpat, "r") as f:
                    query = f.read()
                actual_queries[qid] = [(QueryType.SELECT, query)]
        else:
            actual_order = self.order
            actual_sql_files = self.sql_files
            actual_queries = self.queries

        # Now let us start executing.
        qid_runtime_data: dict[str, BestQueryRun] = {}
        workload_timed_out = False

        for execute_idx, qid in enumerate(actual_order):
            if workload_timed_out:
                break

            queries = actual_queries[qid]
            if any([b in actual_sql_files[qid] for b in blocklist]):
                # Skip any query in blocklist.
                continue

            for qidx, (sql_type, query) in enumerate(queries):
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    # This is a sanity check because any OLTP workload should be run through benchbase, and any OLAP workload should not have INS_UPD_DEL queries. 
                    assert sql_type != QueryType.INS_UPD_DEL
                    pg_conn.conn().execute(query)
                    continue

                # De-duplicate the runs.
                runs: list[QueryRun] = []
                zruns: list[QueryRun] = [
                    QueryRun(
                        act_name,
                        f"{act_name}_{qid}",
                        QuerySpaceKnobAction(
                            {
                                ql_knob[0].knobs[k]: ql_knob[1][k]
                                for k in ql_knob[1].keys()
                                if f"{qid}_" in k
                            }
                        ),
                    )
                    for ql_knob, act_name in zip(ql_knobs, variation_names)
                ]
                for r in zruns:
                    if r[2] not in [rr[2] for rr in runs]:
                        runs.append(r)

                target_pqt = query_timeout if query_timeout else this_execution_workload_timeout
                skip_execute = False
                if (
                    reset_metrics is not None
                    and qid in reset_metrics
                    and not reset_metrics[qid].timed_out
                ):
                    # If we have a reset metric, use it's timeout and convert to seconds.
                    truntime = reset_metrics[qid].runtime
                    assert truntime is not None
                    target_pqt = math.ceil(truntime / 1.0e6)

                    # If we've seen the exact same query knobs before, skip it.
                    rmetrics = reset_metrics[qid]
                    skip_execute = (
                        (rmetrics.query_run is not None)
                        and (rmetrics.query_run.qknobs is not None)
                        and (rmetrics.query_run.qknobs == runs[-1].qknobs)
                    )

                if not skip_execute:
                    best_run: BestQueryRun = execute_variations(
                        connection=pg_conn.conn(),
                        runs=runs,
                        query=query,
                        query_timeout=min(target_pqt, this_execution_workload_timeout - Workload.compute_total_workload_runtime(qid_runtime_data) + 1),
                        logger=self.logger,
                        sysknobs=sysknobs,
                        observation_space=observation_space,
                    )
                else:
                    assert reset_metrics
                    best_run = reset_metrics[qid]

                if reset_metrics is not None and qid in reset_metrics:
                    # Old one is actually better so let's use that.
                    rmetric = reset_metrics[qid]
                    if best_run.timed_out or (
                        best_run.runtime
                        and rmetric.runtime
                        and rmetric.runtime < best_run.runtime
                    ):
                        best_run = rmetric

                assert best_run.runtime
                qid_runtime_data[qid] = best_run

                if Workload.compute_total_workload_runtime(qid_runtime_data) > this_execution_workload_timeout:
                    # We need to undo any potential statements after the timed out query.
                    for st, rq in queries[qidx+1:]:
                        if st != QueryType.SELECT:
                            # This is a sanity check because any OLTP workload should be run through benchbase, and any OLAP workload should not have INS_UPD_DEL queries. If we do have INS_UPD_DEL queries, our "undo" logic will likely have to change.
                            assert st != QueryType.INS_UPD_DEL
                            pg_conn.conn().execute(rq)

                    workload_timed_out = True
                    break

        # Undo any necessary state changes.
        for qqid_index in range(execute_idx, len(actual_order)):
            queries = self.queries[qid]
            for sql_type, query in queries:
                assert sql_type != QueryType.UNKNOWN
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    pg_conn.conn().execute(query)

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
                            (knob.name(), val)
                            for knob, val in run.query_run.qknobs.items()
                        ]
                        f.write(f"{qid}\n{run.query_run.prefix}: {pqkk}\n")
                        f.write(json.dumps(run.explain_data))
                        f.write("\n\n")

            if observation_space and observation_space.require_metrics():
                # Create the metrics.
                # Log the metrics data as a flattened.
                accum_data = cast(
                    list[dict[str, Any]],
                    [v.metric_data for _, v in qid_runtime_data.items()],
                )
                accum_stats = observation_space.merge_deltas(accum_data)
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

            # run.raw.csv will essentially contain the information in qid_runtime_data. However, run.raw.csv may have an extra line for the penalty.
            with open(results_dir / "run.raw.csv", "w") as f:
                # Write the raw query data.
                f.write(
                    "Transaction Type Index,Transaction Name,Start Time (microseconds),Latency (microseconds),Timed Out,Worker Id (start number),Phase Id (index in config file)\n"
                )

                start = 0.0
                for i, qid in enumerate(self.order):
                    if qid in qid_runtime_data:
                        best_run = qid_runtime_data[qid]
                        assert best_run and best_run.runtime and best_run.query_run
                        rtime = best_run.runtime
                        pfx = best_run.query_run.prefix
                        f.write(f"{i+1},{qid},{start},{rtime},{best_run.timed_out},0,{pfx}\n")
                        start += rtime / 1e6

                # Write a penalty term if needed.
                penalty = 0.0
                if workload_timed_out and self.workload_timeout_penalty > 1:
                    # Get the penalty.
                    penalty = (
                        this_execution_workload_timeout * self.workload_timeout_penalty - Workload.compute_total_workload_runtime(qid_runtime_data)
                    )
                    penalty = (penalty + 1.05) * 1e6 if not first else penalty * 1e6
                elif workload_timed_out and not first:
                    # Always degrade it a little if we've timed out.
                    penalty = 3.0e6

                if penalty > 0:
                    f.write(f"{len(self.order)},P,{time.time()},{penalty},True,0,PENALTY\n")

        # Get all the timeouts.
        num_timed_out_queries = sum([1 if best_run.timed_out else 0 for _, best_run in qid_runtime_data.items()])
        return num_timed_out_queries, workload_timed_out, qid_runtime_data

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
        pg_conn: PostgresConn,
        reward_utility: RewardUtility,
        observation_space: StateSpace,
        action_space: HolonSpace,
        actions: list[HolonAction],
        variation_names: list[str],
        benchbase_config: dict[str, Any],
        query_timeout: Optional[int] = None,
        reset_metrics: Optional[dict[str, BestQueryRun]] = None,
        update: bool = True,
        first: bool = False,
    ) -> Tuple[bool, float, float, Union[str, Path], bool, dict[str, BestQueryRun]]:
        success = True
        if self.logger:
            self.logger.get_logger(__name__).info("Starting to run benchmark...")

        # Purge results directory first.
        tmp_dir = tempfile.gettempdir()
        results = f"{tmp_dir}/results{pg_conn.pgport}"
        shutil.rmtree(results, ignore_errors=True)

        if self.benchbase:
            # Execute benchbase if specified.
            success = self._execute_benchbase(benchbase_config, results)
            # We can only create a state if we succeeded.
            success = observation_space.check_benchbase(self.dbgym_cfg, results)
        else:
            num_timed_out_queries, did_workload_time_out, query_metric_data = self.execute_workload(
                pg_conn,
                actions=actions,
                variation_names=variation_names,
                results=results,
                observation_space=observation_space,
                action_space=action_space,
                reset_metrics=reset_metrics,
                override_workload_timeout=self.workload_timeout,
                query_timeout=query_timeout,
                workload_qdir=None,
                blocklist=[],
                first=first,
            )
            did_anything_time_out = num_timed_out_queries > 0 or did_workload_time_out
            success = True

        metric, reward = None, None
        if reward_utility is not None:
            metric, reward = reward_utility(
                result_dir=results, update=update, did_error=not success
            )

        if self.logger:
            self.logger.get_logger(__name__).info(
                f"Benchmark iteration with metric {metric} (reward: {reward}) (did_anything_timeout: {did_anything_time_out})"
            )
        return success, metric, reward, results, did_anything_time_out, query_metric_data
