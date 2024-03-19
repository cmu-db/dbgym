import click

from misc.utils import DBGymConfig


@click.group(name="protox")
@click.pass_obj
def protox_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("protox")
