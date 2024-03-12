import click

from misc.utils import DBGymConfig


@click.group(name="tune")
@click.pass_obj
def tune_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("tune")
