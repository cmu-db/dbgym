import click
import yaml
import logging
from pathlib import Path
from misc.utils import DBGymConfig
import os


task_logger = logging.getLogger("task")
task_logger.setLevel(logging.INFO)


@click.group(name="manage")
def manage_group():
    pass


@click.command(name="show")
@click.argument("keys", nargs=-1)
@click.pass_obj
def manage_show(dbgym_cfg, keys):
    config_path = dbgym_cfg.path
    config_yaml = dbgym_cfg.yaml

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


@click.command(name="write")
@click.argument("keys", nargs=-1)
@click.argument("value_type")
@click.argument("value")
@click.pass_obj
def manage_write(dbgym_cfg, keys, value_type, value):
    config_path = dbgym_cfg.path
    config_yaml = dbgym_cfg.yaml

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


@click.command(name="standardize")
@click.pass_obj
def manage_standardize(dbgym_cfg):
    config_path = dbgym_cfg.path
    config_yaml = dbgym_cfg.yaml

    # Write the YAML file.
    new_yaml = yaml.dump(config_yaml, default_flow_style=False).rstrip()
    Path(config_path).write_text(new_yaml)

    task_logger.info(f"Updated: {Path(config_path)}")


@click.command("clean")
@click.pass_obj
@click.option(
    "--mode",
    type=click.Choice(["safe", "aggressive"]),
    default="safe",
    help="The mode to clean the workspace (default=\"safe\"). \"aggressive\" means \"only keep run_*/ folders referenced by a file in symlinks/\". \"safe\" means \"in addition to that, recursively keep any run_*/ folders referenced by any symlinks in run_*/ folders we are keeping.\""
)
def manage_clean(dbgym_cfg: DBGymConfig, mode: str):
    # This queue holds the symlinks that are left to be processed
    symlink_paths_to_process = []

    # Initialize paths to process
    for root_pathstr, _, file_names in os.walk(dbgym_cfg.dbgym_symlinks_path):
        root_path = Path(root_pathstr)
        for file_name in file_names:
            file_path = root_path / file_name
            if file_path.is_symlink():
                symlink_paths_to_process.append(file_path)

    print(f"symlink_paths_to_process={symlink_paths_to_process}")


manage_group.add_command(manage_show)
manage_group.add_command(manage_write)
manage_group.add_command(manage_standardize)
manage_group.add_command(manage_clean)