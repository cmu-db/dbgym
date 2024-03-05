import logging

import click


@click.group(name="benchmark")
@click.pass_obj
def benchmark_group(config):
    config.append_group("benchmark")
