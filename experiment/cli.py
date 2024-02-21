import click
from misc.utils import conv_inputpath_to_abspath

@click.group(name="expt")
def experiment_group():
    pass

@experiment_group.command()
@click.argument("dir")
def run(dir):
    dir = conv_inputpath_to_abspath(dir)
    click.echo(f'dir={dir}')