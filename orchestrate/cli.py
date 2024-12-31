import click
from gymlib.workspace import DBGymWorkspace

from orchestrate.clean import clean_workspace, count_files_in_workspace


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
def manage_clean(dbgym_workspace: DBGymWorkspace, mode: str) -> None:
    clean_workspace(dbgym_workspace, mode=mode, verbose=True)


@click.command("count")
@click.pass_obj
def manage_count(dbgym_workspace: DBGymWorkspace) -> None:
    num_files = count_files_in_workspace(dbgym_workspace)
    print(
        f"The workspace ({dbgym_workspace.dbgym_workspace_path}) has {num_files} total files/dirs/symlinks."
    )


manage_group.add_command(manage_clean)
manage_group.add_command(manage_count)
