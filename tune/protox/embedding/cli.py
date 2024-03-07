import click

from misc.utils import DBGymConfig


@click.group("embedding")
@click.pass_obj
def embedding_group(config: DBGymConfig):
    config.append_group("embedding")
