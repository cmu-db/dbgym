import click

from misc.utils import DBGymConfig


@click.group("embedding")
@click.pass_obj
def embedding_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("embedding")
