import click

from util.workspace import DBGymConfig
from tune.protox.embedding.datagen import datagen
from tune.protox.embedding.train import train


@click.group("embedding")
@click.pass_obj
def embedding_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("embedding")


embedding_group.add_command(datagen)
embedding_group.add_command(train)
