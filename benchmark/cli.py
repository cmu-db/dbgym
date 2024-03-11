import click

from misc.utils import DBGymConfig


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(config: DBGymConfig):
    config.append_group("benchmark")
