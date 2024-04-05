import shutil
import click
import yaml
import logging
from pathlib import Path
from misc.utils import DBGymConfig, is_child_path, parent_dir
import os

from misc.utils import get_symlinks_path_from_workspace_path

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
    clean_workspace(dbgym_cfg.dbgym_workspace_path, mode)


def clean_workspace(dbgym_cfg: DBGymConfig, mode: str="safe") -> None:
    '''
    Clean all [workspace]/task_runs/run_*/ directories that are not referenced by any "active symlinks".
    If mode is "aggressive", "active symlinks" means *only* the symlinks directly in [workspace]/symlinks/.
    If mode is "safe", "active symlinks" means the symlinks directly in [workspace]/symlinks/ as well as
      any symlinks referenced in task_runs/run_*/ directories we have already decided to keep.
    '''

    # This stack holds the symlinks that are left to be processed
    symlink_fpaths_to_process = []

    # Initialize paths to process
    for root_pathstr, _, file_names in os.walk(get_symlinks_path_from_workspace_path(dbgym_cfg.dbgym_workspace_path)):
        root_path = Path(root_pathstr)
        for file_name in file_names:
            file_path = root_path / file_name
            if file_path.is_symlink():
                symlink_fpaths_to_process.append(file_path)

    # Go through symlinks, figuring out which "children of task runs" to keep
    # Based on the rules of the framework, "children of task runs" should be run_*/ directories.
    # However, the user's workspace might happen to break these rules by putting directories not
    #   named "run_*/" or files directly in task_runs/. Thus, I use the term "task_run_child_fordpaths"
    #   instead of "run_dpaths".
    task_run_child_fordpaths_to_keep = set()
    while symlink_fpaths_to_process:
        symlink_fpath: Path = symlink_fpaths_to_process.pop()
        assert symlink_fpath.is_symlink()
        real_fordpath = symlink_fpath.resolve()

        # If the file doesn't exist, we'll just ignore it.
        if not real_fordpath.exists():
            continue
        # We're only trying to figure out which direct children of task_runs/ to save. If the file isn't
        #   even a descendant, we don't care about it.
        if not is_child_path(real_fordpath, dbgym_cfg.dbgym_runs_path):
            continue

        assert not os.path.samefile(real_fordpath, dbgym_cfg.dbgym_runs_path)

        # Figure out the task_run_child_fordpath to put into task_run_child_fordpaths_to_keep
        task_run_child_fordpath = None
        if os.path.samefile(parent_dir(real_fordpath), dbgym_cfg.dbgym_runs_path):
            # While it's true that it shouldn't be possible to symlink to a directory directly in task_runs/,
            #   we'll just not delete it if the user happens to have one like this. Even if the user messed up
            #   the structure somehow, it's just a good idea not to delete it.
            task_run_child_fordpath = real_fordpath
        else:
            # Technically, it's not allowed to symlink to any files not in task_runs/run_*/[codebase]/[organization]/.
            #   However, as with above, we won't just nuke files if the workspace doesn't follow this rule for
            #   some reason.
            task_run_child_fordpath = real_fordpath
            while not os.path.samefile(parent_dir(parent_dir(task_run_child_fordpath)), dbgym_cfg.dbgym_runs_path):
                task_run_child_fordpath = parent_dir(task_run_child_fordpath)
        assert task_run_child_fordpath != None
        task_run_child_fordpaths_to_keep.add(task_run_child_fordpath)
            
        # If on safe mode, add symlinks inside the task_run_child_fordpath to be processed
        # TODO(phw2)
    
    # Go through all children of task_runs/*, deleting any that we weren't told to keep
    # It's true that symlinks might link outside of task_runs/*. We'll just not care about those
    for child_fordpath in dbgym_cfg.dbgym_runs_path.iterdir():
        if child_fordpath not in task_run_child_fordpaths_to_keep:
            if child_fordpath.is_dir():
                shutil.rmtree(child_fordpath)
            else:
                os.remove(child_fordpath)


manage_group.add_command(manage_show)
manage_group.add_command(manage_write)
manage_group.add_command(manage_standardize)
manage_group.add_command(manage_clean)