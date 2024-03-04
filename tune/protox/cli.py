import click

from tune.protox.embedding.cli import embedding_group


@click.group(name="protox")
def protox_group():
    pass


protox_group.add_command(embedding_group)
