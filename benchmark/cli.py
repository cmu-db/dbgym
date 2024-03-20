import click

from misc.utils import DBGymConfig
from benchmark.tpch.cli import tpch_group


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("benchmark")

benchmark_group.add_command(tpch_group)

