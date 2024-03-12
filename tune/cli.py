import click

from misc.utils import DBGymConfig


@click.group(name="tune")
@click.pass_obj
def tune_group(cfg: DBGymConfig):
    cfg.append_group("tune")
