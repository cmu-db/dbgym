import logging
import math
import time
from typing import Any, Optional, Tuple, Union

import psycopg
from psycopg import Connection
from psycopg.errors import QueryCanceled

from env.pg_conn import PostgresConn
from tune.protox.env.artifact_manager import ArtifactManager
from tune.protox.env.space.primitive.knob import CategoricalKnob, Knob
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.types import (
    BestQueryRun,
    KnobSpaceAction,
    KnobSpaceContainer,
    QueryRun,
    QueryType,
)
from util.log import DBGYM_LOGGER_NAME


def _acquire_metrics_around_query(
    pg_conn: PostgresConn,
    query: str,
    query_timeout: float = 0.0,
    observation_space: Optional[StateSpace] = None,
) -> tuple[float, bool, Optional[dict[str, Any]], Any]:
    pg_conn.force_statement_timeout(0)
    if observation_space and observation_space.require_metrics():
        initial_metrics = observation_space.construct_online(pg_conn.conn())

    qid_runtime, did_time_out, explain_data = pg_conn.time_query(
        query, add_explain=True, timeout=query_timeout
    )

    if observation_space and observation_space.require_metrics():
        final_metrics = observation_space.construct_online(pg_conn.conn())
        diff = observation_space.state_delta(initial_metrics, final_metrics)
    else:
        diff = None

    # qid_runtime is in microseconds.
    return qid_runtime, did_time_out, explain_data, diff


def execute_variations(
    pg_conn: PostgresConn,
    runs: list[QueryRun],
    query: str,
    query_timeout: float = 0,
    artifact_manager: Optional[ArtifactManager] = None,
    sysknobs: Optional[KnobSpaceAction] = None,
    observation_space: Optional[StateSpace] = None,
) -> BestQueryRun:

    # Initial timeout.
    timeout_limit = query_timeout
    # Best run invocation.
    best_qr = BestQueryRun(None, None, True, None, None)

    for qr in runs:
        # Attach the specific per-query knobs.
        pqk_query = (
            "/*+ "
            + " ".join(
                [
                    knob.resolve_per_query_knob(
                        value,
                        all_knobs=sysknobs if sysknobs else KnobSpaceContainer({}),
                    )
                    for knob, value in qr.qknobs.items()
                ]
            )
            + " */"
            + query
        )

        # Log out the knobs that we are using.
        pqkk = [(knob.name(), val) for knob, val in qr.qknobs.items()]
        logging.getLogger(DBGYM_LOGGER_NAME).debug(
            f"{qr.prefix_qid} executing with {pqkk}"
        )

        runtime, did_time_out, explain_data, metric = _acquire_metrics_around_query(
            pg_conn=pg_conn,
            query=pqk_query,
            query_timeout=timeout_limit,
            observation_space=observation_space,
        )

        if not did_time_out:
            new_timeout_limit = math.ceil(runtime / 1e3) / 1.0e3
            if new_timeout_limit < timeout_limit:
                timeout_limit = new_timeout_limit

        if best_qr.runtime is None or runtime < best_qr.runtime:
            assert qr
            best_qr = BestQueryRun(
                qr,
                runtime,
                did_time_out,
                explain_data,
                metric,
            )

        if artifact_manager:
            # Log how long we are executing each query + mode.
            artifact_manager.record(qr.prefix_qid, runtime / 1e6)

    return best_qr
