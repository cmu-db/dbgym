from collections import defaultdict
from pathlib import Path

from env.pg_conn import PostgresConn
from env.tuning_artifacts import TuningArtifactsReader
from env.workload import Workload
from util.pg import DEFAULT_POSTGRES_PORT
from util.workspace import DBGymWorkspace


def replay(
    dbgym_workspace: DBGymWorkspace, tuning_artifacts_path: Path
) -> list[tuple[float, int]]:
    """
    Returns the total runtime and the number of timed out queries for each step.

    The first step will use no configuration changes.
    """
    replay_data: list[tuple[float, int]] = []

    reader = TuningArtifactsReader(tuning_artifacts_path)
    pg_conn = PostgresConn(
        dbgym_workspace,
        DEFAULT_POSTGRES_PORT,
        reader.get_metadata().pristine_dbdata_snapshot_path,
        reader.get_metadata().dbdata_parent_path,
        reader.get_metadata().pgbin_path,
        None,
    )
    workload = Workload(
        dbgym_workspace,
        reader.get_metadata().workload_path,
    )

    pg_conn.restore_pristine_snapshot()
    pg_conn.restart_postgres()
    qknobs: defaultdict[str, list[str]] = defaultdict(list)
    replay_data.append(time_workload(pg_conn, workload, qknobs))

    for delta in reader.get_all_deltas_in_order():
        pg_conn.restart_with_changes(delta.sysknobs)

        for index in delta.indexes:
            pg_conn.psql(index)

        for query, knobs in delta.qknobs.items():
            # TODO: account for deleting a knob if we are representing knobs as deltas.
            qknobs[query].extend(knobs)

        replay_data.append(time_workload(pg_conn, workload, qknobs))

    pg_conn.shutdown_postgres()
    return replay_data


def time_workload(
    pg_conn: PostgresConn, workload: Workload, qknobs: dict[str, list[str]]
) -> tuple[float, int]:
    """
    Returns the total runtime and the number of timed out queries.
    """
    total_runtime: float = 0
    num_timed_out_queries: int = 0

    for qid in workload.get_query_order():
        query = workload.get_query(qid)
        this_query_knobs = qknobs[qid]
        runtime, did_time_out, _ = pg_conn.time_query(
            query, query_knobs=this_query_knobs
        )
        total_runtime += runtime
        if did_time_out:
            num_timed_out_queries += 1

    return total_runtime, num_timed_out_queries
