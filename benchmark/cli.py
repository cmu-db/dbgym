import click

from misc.utils import DBGymConfig


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("benchmark")
