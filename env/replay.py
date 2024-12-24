import logging
from typing import Optional

import click

from benchmark.constants import DEFAULT_SCALE_FACTOR
from env.pg_conn import PostgresConn
from env.workload import Workload
from util.log import DBGYM_OUTPUT_LOGGER_NAME
from util.pg import DEFAULT_POSTGRES_PORT
from util.workspace import (
    DBGymConfig,
    default_workload_path,
    get_default_workload_name_suffix,
    get_workload_name,
)


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.option(
    "--workload-name-suffix",
    type=str,
    default=None,
    help=f"The suffix of the workload name (the part after the scale factor).",
)
@click.option(
    "--scale-factor",
    type=float,
    default=DEFAULT_SCALE_FACTOR,
    help="The scale factor used when generating the data of the benchmark.",
)
def replay(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    workload_name_suffix: Optional[str],
    scale_factor: float,
) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if workload_name_suffix is None:
        workload_name_suffix = get_default_workload_name_suffix(benchmark_name)
    workload_name = get_workload_name(scale_factor, workload_name_suffix)

    # TODO(phw2): Uncomment this.
    # pg_conn = PostgresConn(dbgym_cfg, DEFAULT_POSTGRES_PORT)
    # workload = Workload(
    #     dbgym_cfg,
    #     default_workload_path(
    #         dbgym_cfg.dbgym_workspace_path,
    #         benchmark_name,
    #         workload_name,
    #     ),
    # )
    # total_runtime, num_timed_out_queries = time_workload(pg_conn, workload)
    # logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
    #     f"Total runtime: {total_runtime} seconds"
    # )
    # logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
    #     f"Number of timed out queries: {num_timed_out_queries}"
    # )


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
