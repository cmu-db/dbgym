import click

@click.group(name="protox")
def protox_group():
    pass

@protox_group.command()
def embedding():
    pass