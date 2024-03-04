import logging
from pathlib import Path
import click
import yaml

# TODO(phw2): save commit, git diff, and run command
# TODO(phw2): remove write permissions on old run_*/ dirs to enforce that they are immutable

from experiment.cli import experiment_group
from tune.protox.cli import protox_group

from misc.utils import DBGymConfig

task_logger = logging.getLogger("task")
task_logger.setLevel(logging.INFO)


@click.group()
@click.option("--config-path", default="config.yaml")
@click.option("--no-startup-check", is_flag=True)
@click.pass_context
def task(ctx, config_path, no_startup_check):
    """💩💩💩 CMU-DB Database Gym: github.com/cmu-db/dbgym 💩💩💩"""
    ctx.obj = DBGymConfig(config_path, startup_check=not no_startup_check)


@click.group(name="config")
def config_group():
    pass


@config_group.command(name="show")
@click.argument("keys", nargs=-1)
@click.pass_obj
def config_show(config, keys):
    config_path = config.path
    config_yaml = config.yaml

    # Traverse the YAML.
    for key in keys:
        config_yaml = config_yaml[key]

    # Pretty-print the requested YAML value.
    output_str = None
    if type(config_yaml) != dict:
        output_str = config_yaml
    else:
        output_str = yaml.dump(config_yaml, default_flow_style=False)
        if len(keys) > 0:
            output_str = "  " + output_str.replace("\n", "\n  ")
        output_str = output_str.rstrip()
    print(output_str)

    task_logger.info(f"Read: {Path(config_path)}")


@config_group.command(name="write")
@click.argument("keys", nargs=-1)
@click.argument("value_type")
@click.argument("value")
@click.pass_obj
def config_write(config, keys, value_type, value):
    config_path = config.path
    config_yaml = config.yaml

    # Traverse the YAML.
    root_yaml = config_yaml
    for key in keys[:-1]:
        config_yaml = config_yaml[key]

    # Modify the requested YAML value and write the YAML file.
    assert type(config_yaml[keys[-1]]) != dict
    config_yaml[keys[-1]] = getattr(__builtins__, value_type)(value)
    new_yaml = yaml.dump(root_yaml, default_flow_style=False).rstrip()
    Path(config_path).write_text(new_yaml)

    task_logger.info(f"Updated: {Path(config_path)}")


@config_group.command(name="standardize")
@click.pass_obj
def config_standardize(config):
    config_path = config.path
    config_yaml = config.yaml

    # Write the YAML file.
    new_yaml = yaml.dump(config_yaml, default_flow_style=False).rstrip()
    Path(config_path).write_text(new_yaml)

    task_logger.info(f"Updated: {Path(config_path)}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s:%(name)s:%(levelname)s - %(message)s")

    task.add_command(config_group)
    task.add_command(experiment_group)
    task.add_command(protox_group)
    task()