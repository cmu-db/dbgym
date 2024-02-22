import logging
from pathlib import Path
import sys
from datetime import datetime

import click
import yaml

from experiment.cli import experiment_group
from tune.protox.cli import protox_group

task_logger = logging.getLogger("task")
task_logger.setLevel(logging.INFO)


class Config:
    def __init__(self, config_path, startup_check=False):
        self.path = config_path
        # Parse the YAML file.
        contents = Path(self.path).read_text()
        self.root_yaml = yaml.safe_load(contents)
        self.cur_path = Path(".")
        self.cur_yaml = self.root_yaml

        # Quickly display options.
        if startup_check:
            msg = (
                "💩💩💩 CMU-DB Database Gym: github.com/cmu-db/dbgym 💩💩💩\n"
                f"\tdbgym_workspace_path: {self.root_yaml['dbgym_workspace_path']}\n"
                "\n"
                "Proceed?"
            )
            if not click.confirm(msg):
                print("Goodbye.")
                sys.exit(0)

        # Set and create paths for storing results.
        self.dbgym_workspace_path = Path(self.root_yaml["dbgym_workspace_path"])
        self.dbgym_workspace_path.mkdir(parents=True, exist_ok=True)

        self.dbgym_bin_path = self.dbgym_workspace_path / "bin"
        self.dbgym_bin_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_build_path = self.dbgym_workspace_path / "build"
        self.dbgym_build_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_data_path = self.dbgym_workspace_path / "data"
        self.dbgym_data_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_runs_path = self.dbgym_workspace_path / "task_runs"
        self.dbgym_runs_path.mkdir(parents=True, exist_ok=True)
        curr_dt = datetime.now()
        self.dbgym_this_run_path = self.dbgym_runs_path / f"run_{curr_dt.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.dbgym_this_run_path.mkdir(parents=True, exist_ok=False)

    def append_group(self, name):
        self.cur_path /= name
        self.cur_yaml = config.cur_yaml.get(name, {})

    @property
    def cur_bin_path(self):
        return self.dbgym_bin_path / self.cur_path

    @property
    def cur_build_path(self):
        return self.dbgym_build_path / self.cur_path

    @property
    def cur_data_path(self):
        return self.dbgym_data_path / self.cur_path


@click.group()
@click.option("--config-path", default="config.yaml")
@click.option("--no-startup-check", is_flag=True)
@click.pass_context
def task(ctx, config_path, no_startup_check):
    """💩💩💩 CMU-DB Database Gym: github.com/cmu-db/dbgym 💩💩💩"""
    ctx.obj = Config(config_path, startup_check=not no_startup_check)


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