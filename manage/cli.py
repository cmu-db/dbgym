import logging
import os
import shutil
from itertools import chain
from pathlib import Path

import click

from util.log import DBGYM_LOGGER_NAME, DBGYM_OUTPUT_LOGGER_NAME
from util.workspace import (
    DBGymConfig,
    get_runs_path_from_workspace_path,
    get_symlinks_path_from_workspace_path,
    is_child_path,
    parent_dpath_of_path,
)


# This is used in test_clean.py. It's defined here to avoid a circular import.
class MockDBGymConfig:
    def __init__(self, scratchspace_path: Path):
        self.dbgym_workspace_path = scratchspace_path
        self.dbgym_symlinks_path = get_symlinks_path_from_workspace_path(
            scratchspace_path
        )
        self.dbgym_runs_path = get_runs_path_from_workspace_path(scratchspace_path)


@click.group(name="manage")
def manage_group() -> None:
    pass


@click.command("clean")
@click.pass_obj
@click.option(
    "--mode",
    type=click.Choice(["safe", "aggressive"]),
    default="safe",
    help='The mode to clean the workspace (default="safe"). "aggressive" means "only keep run_*/ folders referenced by a file in symlinks/". "safe" means "in addition to that, recursively keep any run_*/ folders referenced by any symlinks in run_*/ folders we are keeping."',
)
def manage_clean(dbgym_cfg: DBGymConfig, mode: str) -> None:
    clean_workspace(dbgym_cfg, mode=mode, verbose=True)


@click.command("count")
@click.pass_obj
def manage_count(dbgym_cfg: DBGymConfig) -> None:
    num_files = _count_files_in_workspace(dbgym_cfg)
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"The workspace ({dbgym_cfg.dbgym_workspace_path}) has {num_files} total files/dirs/symlinks."
    )


def add_symlinks_in_dpath(
    symlinks_stack: list[Path], root_dpath: Path, processed_symlinks: set[Path]
) -> None:
    """
    Will modify symlinks_stack and processed_symlinks.
    """
    for root_pathstr, dir_names, file_names in os.walk(root_dpath):
        root_path = Path(root_pathstr)
        # symlinks can either be files or directories, so we go through both dir_names and file_names
        for file_name in chain(dir_names, file_names):
            file_path = root_path / file_name
            if file_path.is_symlink() and file_path not in processed_symlinks:
                symlinks_stack.append(file_path)
                processed_symlinks.add(file_path)


def _count_files_in_workspace(dbgym_cfg: DBGymConfig | MockDBGymConfig) -> int:
    """
    Counts the number of files (regular file or dir or symlink) in the workspace.
    """
    total_count = 0
    for dirpath, dirnames, filenames in os.walk(
        dbgym_cfg.dbgym_workspace_path, followlinks=False
    ):
        # Check if any of the directories are symbolic links and remove them from dirnames
        dirnames[:] = [
            d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
        ]

        # Count files and directories (non-symlink directories already filtered)
        total_count += len(filenames) + len(dirnames)

    return total_count


def clean_workspace(
    dbgym_cfg: DBGymConfig | MockDBGymConfig, mode: str = "safe", verbose: bool = False
) -> None:
    """
    Clean all [workspace]/task_runs/run_*/ directories that are not referenced by any "active symlinks".
    If mode is "aggressive", "active symlinks" means *only* the symlinks directly in [workspace]/symlinks/.
    If mode is "safe", "active symlinks" means the symlinks directly in [workspace]/symlinks/ as well as
      any symlinks referenced in task_runs/run_*/ directories we have already decided to keep.
    """
    # This stack holds the symlinks that are left to be processed
    symlink_fpaths_to_process: list[Path] = []
    # This set holds the symlinks that have already been processed to avoid infinite loops
    processed_symlinks: set[Path] = set()

    # 1. Initialize paths to process
    if dbgym_cfg.dbgym_symlinks_path.exists():
        add_symlinks_in_dpath(
            symlink_fpaths_to_process, dbgym_cfg.dbgym_symlinks_path, processed_symlinks
        )

    # 2. Go through symlinks, figuring out which "children of task runs" to keep
    # Based on the rules of the framework, "children of task runs" should be run_*/ directories.
    # However, the user's workspace might happen to break these rules by putting directories not
    #   named "run_*/" or files directly in task_runs/. Thus, I use the term "task_run_child_fordpaths"
    #   instead of "run_dpaths".
    task_run_child_fordpaths_to_keep = set()

    if dbgym_cfg.dbgym_runs_path.exists():
        while symlink_fpaths_to_process:
            symlink_fpath: Path = symlink_fpaths_to_process.pop()
            assert symlink_fpath.is_symlink()
            # Path.resolve() resolves all layers of symlinks while os.readlink() only resolves one layer.
            # However, os.readlink() literally reads the string contents of the link. We need to do some
            #   processing on the result of os.readlink() to convert it to an absolute path
            real_fordpath = symlink_fpath.resolve()
            one_layer_resolved_fordpath = os.readlink(symlink_fpath)
            assert str(real_fordpath) == str(
                os.readlink(symlink_fpath)
            ), f"symlink_fpath ({symlink_fpath}) seems to point to *another* symlink. This is difficult to handle, so it is currently disallowed. Please resolve this situation manually."

            # If the file doesn't exist, we'll just ignore it.
            if not real_fordpath.exists():
                continue
            # We're only trying to figure out which direct children of task_runs/ to save. If the file isn't
            #   even a descendant, we don't care about it.
            if not is_child_path(real_fordpath, dbgym_cfg.dbgym_runs_path):
                continue

            assert not real_fordpath.samefile(dbgym_cfg.dbgym_runs_path)

            # Figure out the task_run_child_fordpath to put into task_run_child_fordpaths_to_keep
            task_run_child_fordpath = None
            if parent_dpath_of_path(real_fordpath).samefile(dbgym_cfg.dbgym_runs_path):
                # While it's true that it shouldn't be possible to symlink to a directory directly in task_runs/,
                #   we'll just not delete it if the user happens to have one like this. Even if the user messed up
                #   the structure somehow, it's just a good idea not to delete it.
                task_run_child_fordpath = real_fordpath
            else:
                # Technically, it's not allowed to symlink to any files not in task_runs/run_*/[codebase]/[organization]/.
                #   However, as with above, we won't just nuke files if the workspace doesn't follow this rule for
                #   some reason.
                task_run_child_fordpath = real_fordpath
                while not parent_dpath_of_path(task_run_child_fordpath).samefile(
                    dbgym_cfg.dbgym_runs_path
                ):
                    task_run_child_fordpath = parent_dpath_of_path(
                        task_run_child_fordpath
                    )
            assert task_run_child_fordpath != None
            assert parent_dpath_of_path(task_run_child_fordpath).samefile(
                dbgym_cfg.dbgym_runs_path
            ), f"task_run_child_fordpath ({task_run_child_fordpath}) is not a direct child of dbgym_cfg.dbgym_runs_path"
            task_run_child_fordpaths_to_keep.add(task_run_child_fordpath)

            # If on safe mode, add symlinks inside the task_run_child_fordpath to be processed
            if mode == "safe":
                add_symlinks_in_dpath(
                    symlink_fpaths_to_process,
                    task_run_child_fordpath,
                    processed_symlinks,
                )

    # 3. Go through all children of task_runs/*, deleting any that we weren't told to keep
    # It's true that symlinks might link outside of task_runs/*. We'll just not care about those
    starting_num_files = _count_files_in_workspace(dbgym_cfg)
    if dbgym_cfg.dbgym_runs_path.exists():
        for child_fordpath in dbgym_cfg.dbgym_runs_path.iterdir():
            if child_fordpath not in task_run_child_fordpaths_to_keep:
                if child_fordpath.is_dir():
                    shutil.rmtree(child_fordpath)
                else:
                    os.remove(child_fordpath)
    ending_num_files = _count_files_in_workspace(dbgym_cfg)

    if verbose:
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Removed {starting_num_files - ending_num_files} out of {starting_num_files} files"
        )
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Workspace went from {starting_num_files - ending_num_files} to {starting_num_files}"
        )


manage_group.add_command(manage_clean)
manage_group.add_command(manage_count)
