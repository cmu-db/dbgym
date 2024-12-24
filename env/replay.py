import logging
from pathlib import Path

from env.pg_conn import PostgresConn
from env.tuning_agent import TuningAgentArtifactsReader
from env.workload import Workload
from util.log import DBGYM_OUTPUT_LOGGER_NAME
from util.pg import DEFAULT_POSTGRES_PORT
from util.workspace import DBGymConfig


# TODO: make it return the full replay data.
def replay(dbgym_cfg: DBGymConfig, tuning_agent_artifacts_dpath: Path) -> None:
    reader = TuningAgentArtifactsReader(tuning_agent_artifacts_dpath)

    # TODO: figure out port
    pg_conn = PostgresConn(
        dbgym_cfg,
        DEFAULT_POSTGRES_PORT,
        reader.get_metadata().pristine_dbdata_snapshot_path,
        reader.get_metadata().dbdata_parent_path,
        reader.get_metadata().pgbin_path,
        None,
    )
    workload = Workload(
        dbgym_cfg,
        reader.get_metadata().workload_path,
    )
    total_runtime, num_timed_out_queries = time_workload(pg_conn, workload)
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"Total runtime: {total_runtime} seconds"
    )
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"Number of timed out queries: {num_timed_out_queries}"
    )


def time_workload(pg_conn: PostgresConn, workload: Workload) -> tuple[float, int]:
    """
    It returns the total runtime and the number of timed out queries.
    """
    total_runtime: float = 0
    num_timed_out_queries: int = 0

    for query in workload.get_queries_in_order():
        runtime, did_time_out, _ = pg_conn.time_query(query)
        total_runtime += runtime
        if did_time_out:
            num_timed_out_queries += 1

    return total_runtime, num_timed_out_queries
