import click

@click.group(name="expt")
def experiment_group():
    pass

@experiment_group.command()
@click.argument("dir")
def run(dir):
    click.echo(f'dir={dir}')