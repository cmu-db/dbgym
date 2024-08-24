import math
import time
from typing import Any, Optional, Tuple, Union

import psycopg
from psycopg import Connection
from psycopg.errors import QueryCanceled

from tune.protox.env.logger import Logger
from tune.protox.env.space.primitive.knob import CategoricalKnob, Knob
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.types import (BestQueryRun, KnobSpaceAction,
                                   KnobSpaceContainer, QueryRun, QueryType)


def _force_statement_timeout(
    connection: psycopg.Connection[Any], timeout_ms: float
) -> None:
    retry = True
    while retry:
        retry = False
        try:
            connection.execute(f"SET statement_timeout = {timeout_ms}")
        except QueryCanceled:
            retry = True


def _time_query(
    logger: Optional[Logger],
    prefix: str,
    connection: psycopg.Connection[Any],
    query: str,
    timeout: float,
) -> Tuple[float, bool, Any]:
    did_time_out = False
    has_explain = "EXPLAIN" in query
    explain_data = None

    try:
        start_time = time.time()
        cursor = connection.execute(query)
        qid_runtime = (time.time() - start_time) * 1e6

        if has_explain:
            c = [c for c in cursor][0][0][0]
            assert "Execution Time" in c
            qid_runtime = float(c["Execution Time"]) * 1e3
            explain_data = c

        if logger:
            logger.get_logger(__name__).debug(
                f"{prefix} evaluated in {qid_runtime/1e6}"
            )

    except QueryCanceled:
        if logger:
            logger.get_logger(__name__).debug(
                f"{prefix} exceeded evaluation timeout {timeout}"
            )
        qid_runtime = timeout * 1e6
        did_time_out = True
    except Exception as e:
        assert False, print(e)
    # qid_runtime is in microseconds.
    return qid_runtime, did_time_out, explain_data


def _acquire_metrics_around_query(
    logger: Optional[Logger],
    prefix: str,
    connection: psycopg.Connection[Any],
    query: str,
    query_timeout: float = 0.0,
    observation_space: Optional[StateSpace] = None,
) -> Tuple[float, bool, Any, Any]:
    _force_statement_timeout(connection, 0)
    if observation_space and observation_space.require_metrics():
        initial_metrics = observation_space.construct_online(connection)

    if query_timeout > 0:
        _force_statement_timeout(connection, query_timeout * 1000)
    else:
        assert (
            query_timeout == 0
        ), f'Setting query_timeout to 0 indicates "timeout". However, setting query_timeout ({query_timeout}) < 0 is a bug.'

    qid_runtime, did_time_out, explain_data = _time_query(
        logger, prefix, connection, query, query_timeout
    )

    # Wipe the statement timeout.
    _force_statement_timeout(connection, 0)
    if observation_space and observation_space.require_metrics():
        final_metrics = observation_space.construct_online(connection)
        diff = observation_space.state_delta(initial_metrics, final_metrics)
    else:
        diff = None

    # qid_runtime is in microseconds.
    return qid_runtime, did_time_out, explain_data, diff


def execute_variations(
    connection: psycopg.Connection[Any],
    runs: list[QueryRun],
    query: str,
    query_timeout: float = 0,
    logger: Optional[Logger] = None,
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
        # Log the query plan.
        pqk_query = "EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + pqk_query

        # Log out the knobs that we are using.
        pqkk = [(knob.name(), val) for knob, val in qr.qknobs.items()]
        if logger:
            logger.get_logger(__name__).debug(f"{qr.prefix_qid} executing with {pqkk}")

        runtime, did_time_out, explain_data, metric = _acquire_metrics_around_query(
            logger=logger,
            prefix=qr.prefix_qid,
            connection=connection,
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

        if logger:
            # Log how long we are executing each query + mode.
            logger.record(qr.prefix_qid, runtime / 1e6)

    return best_qr
