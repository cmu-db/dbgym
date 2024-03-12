import click

from misc.utils import DBGymConfig


@click.group(name="protox")
@click.pass_obj
def protox_group(cfg: DBGymConfig):
    cfg.append_group("protox")
