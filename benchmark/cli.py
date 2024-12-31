import click
from gymlib.workspace import DBGymWorkspace

from benchmark.job.cli import job_group
from benchmark.tpch.cli import tpch_group


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(dbgym_workspace: DBGymWorkspace) -> None:
    pass


benchmark_group.add_command(tpch_group)
benchmark_group.add_command(job_group)
