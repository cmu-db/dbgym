import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union, cast

import numpy as np
import psycopg
from gymnasium import spaces
from gymnasium.spaces import Box, Space
from psycopg.rows import dict_row

from misc.utils import DBGymConfig, open_and_save
from tune.protox.env.space.state.space import StateSpace
from util.pg import DBGYM_POSTGRES_DBNAME

# Defines the relevant metrics that we care about from benchbase.
# <filter_db>: whether to filter with the benchbase database.
# <per_table>: whether to process the set of valid_keys per table.
METRICS_SPECIFICATION = {
    "pg_stat_database": {
        "filter_db": True,
        "per_table": False,
        "valid_keys": [
            "temp_files",
            "tup_returned",
            "xact_commit",
            "xact_rollback",
            "conflicts",
            "blks_hit",
            "blks_read",
            "temp_bytes",
            "deadlocks",
            "tup_inserted",
            "tup_fetched",
            "tup_updated",
            "tup_deleted",
        ],
    },
    "pg_stat_bgwriter": {
        "filter_db": False,
        "per_table": False,
        "valid_keys": [
            "checkpoint_write_time",
            "buffers_backend_fsync",
            "buffers_clean",
            "buffers_checkpoint",
            "checkpoints_req",
            "checkpoints_timed",
            "buffers_alloc",
            "buffers_backend",
            "maxwritten_clean",
        ],
    },
    "pg_stat_database_conflicts": {
        "filter_db": True,
        "per_table": False,
        "valid_keys": [
            "confl_deadlock",
            "confl_lock",
            "confl_bufferpin",
            "confl_snapshot",
        ],
    },
    "pg_stat_user_tables": {
        "filter_db": False,
        "per_table": True,
        "valid_keys": [
            "n_tup_ins",
            "n_tup_upd",
            "n_tup_del",
            "n_ins_since_vacuum",
            "n_mod_since_analyze",
            "n_tup_hot_upd",
            "idx_tup_fetch",
            "seq_tup_read",
            "autoanalyze_count",
            "autovacuum_count",
            "n_live_tup",
            "n_dead_tup",
            "seq_scan",
            "idx_scan",
        ],
    },
    "pg_statio_user_tables": {
        "filter_db": False,
        "per_table": True,
        "valid_keys": [
            "heap_blks_hit",
            "heap_blks_read",
            "idx_blks_hit",
            "idx_blks_read",
            "tidx_blks_hit",
            "tidx_blks_read",
            "toast_blks_hit",
            "toast_blks_read",
        ],
    },
}


# A metrics-based state returns the physical metrics (i.e., consequences) of running
# a particular workload in a given configuration. This serves to represent the
# assumption that we should be indifferent/invariant to <workload, configuration>
# pairs that yield the *same* physical metrics.
#
# In the RL state-action-reward-next_state sense:
# The benchmark is executed in the baseline configuration to determine the physical metrics
# as a consequence of the baseline configuration. That is the "previous state".
#
# You then pick an action that produces a new configuration. That configuration is then applied
# to the database. This is the action.
#
# You then run the benchmark again. This yields some "target" metric and also physical database
# metrics. "target" metric is used to determine the reward from the transition. The physical
# database metrics form the "next_state".
#
# In this way, the physical database metrics serves as proxy for the actual configuration at
# a given moment in time. This is arguably a little bit twisted?? Since we are using some
# metrics that are also indirectly a proxy for the actual runtime/tps. But we are banking
# on the metrics containing the relevant data to allow better action selection...
class MetricStateSpace(StateSpace, spaces.Dict):
    @staticmethod
    def construct_key(key: str, metric: str, per_tbl: bool, tbl: Optional[str]) -> str:
        if per_tbl:
            assert tbl
            return f"{key}_{metric}_{tbl}"
        return f"{key}_{metric}"

    def require_metrics(self) -> bool:
        return True

    def __init__(
        self, dbgym_cfg: DBGymConfig, spaces: Mapping[str, spaces.Space[Any]], tables: list[str], seed: int
    ) -> None:
        self.dbgym_cfg = dbgym_cfg
        self.tables = tables
        self.internal_spaces: dict[str, Space[Any]] = {}
        self.internal_spaces.update(spaces)
        for key, spec in METRICS_SPECIFICATION.items():
            for key_metric in cast(list[str], spec["valid_keys"]):
                if spec["per_table"]:
                    for tbl in tables:
                        tbl_metric = MetricStateSpace.construct_key(
                            key, key_metric, True, tbl
                        )
                        assert tbl_metric not in self.internal_spaces
                        self.internal_spaces[tbl_metric] = Box(low=-np.inf, high=np.inf)
                else:
                    metric = MetricStateSpace.construct_key(
                        key, key_metric, False, None
                    )
                    assert metric not in self.internal_spaces
                    self.internal_spaces[metric] = Box(low=-np.inf, high=np.inf)
        super().__init__(self.internal_spaces, seed)

    def check_benchbase(self, dbgym_cfg: DBGymConfig, results_dpath: Union[str, Path]) -> bool:
        assert results_dpath is not None
        assert Path(results_dpath).exists()
        metric_files = [f for f in Path(results_dpath).rglob("*metrics.json")]
        if len(metric_files) != 2:
            return False

        initial = (
            metric_files[0] if "initial" in str(metric_files[0]) else metric_files[1]
        )
        final = metric_files[1] if initial == metric_files[0] else metric_files[0]

        try:
            with open_and_save(dbgym_cfg, initial) as f:
                initial_metrics = json.load(f)

            with open_and_save(dbgym_cfg, final) as f:
                final_metrics = json.load(f)
        except Exception as e:
            return False

        for key, spec in METRICS_SPECIFICATION.items():
            assert key in initial_metrics
            if key not in initial_metrics or key not in final_metrics:
                # Missing key.
                return False

            initial_data = initial_metrics[key]
            final_data = final_metrics[key]
            if spec["filter_db"]:
                initial_data = [d for d in initial_data if d["datname"] == DBGYM_POSTGRES_DBNAME]
                final_data = [d for d in final_data if d["datname"] == DBGYM_POSTGRES_DBNAME]
            elif spec["per_table"]:
                initial_data = sorted(
                    [d for d in initial_data if d["relname"] in self.tables],
                    key=lambda x: x["relname"],
                )
                final_data = sorted(
                    [d for d in final_data if d["relname"] in self.tables],
                    key=lambda x: x["relname"],
                )

            if len(initial_data) == 0 or len(final_data) == 0:
                return False

            for pre, post in zip(initial_data, final_data):
                for metric in cast(list[str], spec["valid_keys"]):
                    if metric not in pre or metric not in post:
                        return False
        return True

    def construct_offline(
        self, connection: psycopg.Connection[Any], data: Any, state_container: Any
    ) -> dict[str, Any]:
        assert data is not None
        assert Path(data).exists()

        # This function computes the metrics state that is used to represent
        # consequence of executing in the current environment.
        metric_files = [f for f in Path(data).rglob("*metrics.json")]
        if len(metric_files) == 1:
            with open_and_save(self.dbgym_cfg, metric_files[0], "r") as f:
                metrics = json.load(f)
                assert "flattened" in metrics
                metrics.pop("flattened")

            def npify(d: dict[str, Any]) -> Any:
                data = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        data[k] = npify(v)
                    else:
                        data[k] = np.array([v], dtype=np.float32)
                return data

            return cast(dict[str, Any], npify(metrics))

        assert len(metric_files) == 2
        initial = (
            metric_files[0] if "initial" in str(metric_files[0]) else metric_files[1]
        )
        final = metric_files[1] if initial == metric_files[0] else metric_files[0]

        with open_and_save(self.dbgym_cfg, initial) as f:
            initial_metrics = json.load(f)

        with open_and_save(self.dbgym_cfg, final) as f:
            final_metrics = json.load(f)

        return self.state_delta(initial_metrics, final_metrics)

    def state_delta(
        self, initial: dict[str, Any], final: dict[str, Any]
    ) -> dict[str, Any]:
        metrics = {}
        for key, spec in METRICS_SPECIFICATION.items():
            assert key in initial
            assert isinstance(spec, dict)

            initial_data = initial[key]
            final_data = final[key]
            if spec["filter_db"]:
                initial_data = [d for d in initial_data if d["datname"] == DBGYM_POSTGRES_DBNAME]
                final_data = [d for d in final_data if d["datname"] == DBGYM_POSTGRES_DBNAME]
            elif spec["per_table"]:
                initial_data = sorted(
                    [d for d in initial_data if d["relname"] in self.tables],
                    key=lambda x: x["relname"],
                )
                final_data = sorted(
                    [d for d in final_data if d["relname"] in self.tables],
                    key=lambda x: x["relname"],
                )

            for pre, post in zip(initial_data, final_data):
                for metric in cast(list[str], spec["valid_keys"]):
                    if pre[metric] is None or post[metric] is None:
                        diff = 0.0
                    else:
                        diff = max(float(post[metric]) - float(pre[metric]), 0.0)

                    metric_key = MetricStateSpace.construct_key(
                        key,
                        metric,
                        bool(spec["per_table"]),
                        pre["relname"] if spec["per_table"] else None,
                    )
                    metrics[metric_key] = np.array([diff], dtype=np.float32)
        return metrics

    def construct_online(self, connection: psycopg.Connection[Any]) -> dict[str, Any]:
        metric_data = {}
        with connection.cursor(row_factory=dict_row) as cursor:
            for key in METRICS_SPECIFICATION.keys():
                records = cursor.execute(f"SELECT * FROM {key}")
                metric_data[key] = [r for r in records]
        return metric_data

    def merge_deltas(self, merge_data: list[dict[str, Any]]) -> dict[str, Any]:
        comb_data = {}
        for datum in merge_data:
            for key, value in datum.items():
                if key not in comb_data:
                    comb_data[key] = value
                else:
                    comb_data[key] += value
        return comb_data
