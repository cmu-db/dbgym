from collections import defaultdict
from pathlib import Path

from gymlib.pg import DEFAULT_POSTGRES_PORT
from gymlib.pg_conn import PostgresConn
from gymlib.tuning_artifacts import TuningArtifactsReader
from gymlib.workload import Workload
from gymlib.workspace import DBGymWorkspace


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
    replay_data.append(pg_conn.time_workload(workload, qknobs))

    for delta in reader.get_all_deltas_in_order():
        pg_conn.restart_with_changes(delta.sysknobs)

        for index in delta.indexes:
            pg_conn.psql(index)

        for query, knobs in delta.qknobs.items():
            # TODO: account for deleting a knob if we are representing knobs as deltas.
            qknobs[query].extend(knobs)

        replay_data.append(pg_conn.time_workload(workload, qknobs))

    pg_conn.shutdown_postgres()
    return replay_data
