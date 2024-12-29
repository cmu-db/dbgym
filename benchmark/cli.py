import click

from benchmark.job.cli import job_group
from benchmark.tpch.cli import tpch_group
from util.workspace import DBGymWorkspace


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(dbgym_workspace: DBGymWorkspace) -> None:
    dbgym_workspace.append_group("benchmark")


benchmark_group.add_command(tpch_group)
benchmark_group.add_command(job_group)
