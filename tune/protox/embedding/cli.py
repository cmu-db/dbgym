import click

from tune.protox.embedding.datagen import datagen
from tune.protox.embedding.train import train


@click.group("embedding")
def embedding_group():
    pass


embedding_group.add_command(train)
embedding_group.add_command(datagen)
