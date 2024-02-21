import click

@click.group("embedding")
def embedding_group():
    pass

@embedding_group.command()
def datagen():
    pass

@embedding_group.command()
def train():
    pass