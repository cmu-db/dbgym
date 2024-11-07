import click

from benchmark.job.cli import job_group
from benchmark.tpch.cli import tpch_group
from util.workspace import DBGymConfig


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("benchmark")


benchmark_group.add_command(tpch_group)
benchmark_group.add_command(job_group)
