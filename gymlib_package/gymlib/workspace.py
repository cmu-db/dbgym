"""
This file contains everything needed to manage the workspace (the dbgym_workspace/ folder).
"""

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

import yaml
from gymlib.symlinks_paths import is_linkname, name_to_linkname

WORKSPACE_PATH_PLACEHOLDER = Path("[workspace]")


# Helper functions that both this file and other files use
def get_symlinks_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "symlinks"


def get_tmp_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "tmp"


def get_runs_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "task_runs"


def get_latest_run_path_from_workspace_path(workspace_path: Path) -> Path:
    return get_runs_path_from_workspace_path(workspace_path) / "latest_run.link"


# Paths of config files in the codebase. These are always relative paths.
# The reason these can be relative paths instead of functions taking in codebase_path as input is because relative paths are relative to the codebase root
DEFAULT_BOOT_CONFIG_PATH = Path("dbms") / "postgres" / "default_boot_config.yaml"


class DBGymWorkspace:
    """
    Global configurations that apply to all parts of DB-Gym
    """

    _num_times_created_this_run: int = 0

    def __init__(self, dbgym_workspace_path: Path):
        # The logic around dbgym_tmp_path assumes that DBGymWorkspace is only constructed once.
        # This is because DBGymWorkspace creates a new run_*/ dir when it's initialized.
        DBGymWorkspace._num_times_created_this_run += 1
        assert (
            DBGymWorkspace._num_times_created_this_run == 1
        ), f"DBGymWorkspace has been created {DBGymWorkspace._num_times_created_this_run} times. It should only be created once per run."

        self.base_dbgym_repo_path = get_base_dbgym_repo_path()
        self.app_name = (
            "dbgym"  # TODO: discover this dynamically. app means dbgym or an agent
        )

        # Set and create paths.
        self.dbgym_workspace_path = dbgym_workspace_path
        self.dbgym_workspace_path.mkdir(parents=True, exist_ok=True)

        # Now that the workspace is guaranteed to be created, we can check if it's fully resolved.
        assert is_fully_resolved(self.dbgym_workspace_path)

        self.dbgym_runs_path = get_runs_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        self.dbgym_runs_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_symlinks_path = get_symlinks_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        self.dbgym_symlinks_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_cur_symlinks_path = self.dbgym_symlinks_path / self.app_name
        # tmp/ is a workspace for this run only
        # One use for it is to place the unzipped dbdata.
        # There's no need to save the actual dbdata dir in run_*/ because we just save a symlink to
        #   the .tgz file we unzipped.
        self.dbgym_tmp_path = get_tmp_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        # The best place to delete the old dbgym_tmp_path is in DBGymWorkspace.__init__().
        # This is better than deleting the dbgym_tmp_path is in DBGymWorkspace.__del__() because DBGymWorkspace may get deleted before execution has completed.
        # Also, by keeping the tmp directory around, you can look at it to debug issues.
        if self.dbgym_tmp_path.exists():
            shutil.rmtree(self.dbgym_tmp_path)
        self.dbgym_tmp_path.mkdir(parents=True, exist_ok=True)

        # Set the path for this task run's results.
        for _ in range(2):
            try:
                self.dbgym_this_run_path = (
                    self.dbgym_runs_path
                    / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                )
                # `exist_ok` is False because we don't want to override a previous task run's data.
                self.dbgym_this_run_path.mkdir(parents=True, exist_ok=False)
                # Break if it succeeds so we don't do it a second time.
                break
            except FileExistsError:
                # In case we call task.py twice in one second, sleeping here will fix it.
                # Waiting one second is enough since we assume there's only one task.py running at a time.
                time.sleep(1)
            except Exception as e:
                raise e

        self.dbgym_latest_run_path = get_latest_run_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        try_remove_file(self.dbgym_latest_run_path)
        try_create_symlink(self.dbgym_this_run_path, self.dbgym_latest_run_path)

    # TODO(phw2): refactor our manual symlinking in postgres/cli.py to use link_result() instead
    def link_result(
        self,
        result_path: Path,
        custom_link_name: Optional[str] = None,
    ) -> Path:
        """
        result_path must be a "result", meaning it was generated inside dbgym_workspace.dbgym_this_run_path.
        Further, result_path must have been generated by this invocation to task.py. This also means that
            result_path itself can be a file or a dir but not a symlink.
        Given a file or directory in task_runs/run_*/[codebase]/[org], this will create a symlink inside
            symlinks/[codebase]/[org]/.
        Will override the old symlink if there is one, so that symlinks/ always contains the latest generated
            version of a file.
        This function will return the path to the symlink that was created.
        """
        assert isinstance(result_path, Path)
        assert is_fully_resolved(
            result_path
        ), f"result_path ({result_path}) should be a fully resolved path"
        assert is_child_path(
            result_path, self.dbgym_this_run_path
        ), "The result must have been generated in *this* run_*/ dir"
        assert not os.path.islink(result_path)

        if type(custom_link_name) is str:
            link_name = custom_link_name
        else:
            if os.path.isfile(result_path):
                link_name = name_to_linkname(basename_of_path(result_path))
            elif os.path.isdir(result_path):
                link_name = name_to_linkname(basename_of_path(result_path))
            else:
                raise AssertionError("result_path must be either a file or dir")

        symlink_parent_path = self.dbgym_symlinks_path / self.app_name
        symlink_parent_path.mkdir(parents=True, exist_ok=True)

        # Remove the old symlink ("old" meaning created in an earlier run) if there is one
        # Note that in a multi-threaded setting, this might remove one created by a process in the same run,
        #   meaning it's not "old" by our definition of "old". However, we'll always end up with a symlink
        #   file of the current run regardless of the order of threads.
        assert is_linkname(
            link_name
        ), f'link_name ({link_name}) should end with ".link"'
        symlink_path = symlink_parent_path / link_name
        try_remove_file(symlink_path)
        try_create_symlink(result_path, symlink_path)

        return symlink_path

    def get_run_path_from_path(self, path: Path) -> Path:
        run_path = path
        while not parent_path_of_path(run_path).samefile(self.dbgym_runs_path):
            run_path = parent_path_of_path(run_path)
        return run_path

    # TODO(phw2): really look at the clean PR to see what it changed
    # TODO(phw2): after merging agent-train, refactor some code in agent-train to use save_file() instead of open_and_save()
    def save_file(self, path: Path) -> None:
        """
        If an external function takes in a file/directory as input, you will not be able to call open_and_save().
            In these situations, just call save_file().
        Like open_and_save(), this function only works with real absolute paths.
        "Saving" can mean either copying the file or creating a symlink to it
        We copy the file if it is a "config", meaning it just exists without having been generated
        We create a symlink if it is a "dependency", meaning a task.py command was run to generate it
            In these cases we create a symlink so we have full provenance for how the dependency was created

        **Notable Behavior**
          - When you save a dependency, it actually creates a link to the outermost directory still inside run_*/.
          - The second save will overwrite the first.
            - If you save the same file twice in the same run, the second save will overwrite the first.
            - If you save two configs with the same name, the second save will overwrite the first.
            - If you save two dependencies with the same *outermost* directory, or two dependencies with the same filename
              both directly inside run_*/, the second save will overwrite the first.
        """
        # validate path
        assert isinstance(path, Path)
        assert not os.path.islink(path), f"path ({path}) should not be a symlink"
        assert os.path.exists(path), f"path ({path}) does not exist"
        assert os.path.isfile(path), f"path ({path}) is not a file"
        assert not is_child_path(
            path, self.dbgym_this_run_path
        ), f"path ({path}) was generated in this task run ({self.dbgym_this_run_path}). You do not need to save it"

        # Save _something_ to dbgym_this_run_path.
        # Save a symlink if the opened file was generated by a run. This is for two reasons:
        #   1. Files or dirs generated by a run are supposed to be immutable so saving a symlink is safe.
        #   2. Files or dirs generated by a run may be very large (up to 100s of GBs) so we don't want to copy them.
        if is_child_path(path, self.dbgym_runs_path):
            # If the path file is directly in run_path, we symlink the file directly.
            run_path = self.get_run_path_from_path(path)
            parent_path = parent_path_of_path(path)
            if parent_path.samefile(run_path):
                fname = basename_of_path(path)
                symlink_path = self.dbgym_this_run_path / name_to_linkname(fname)
                try_remove_file(symlink_path)
                try_create_symlink(path, symlink_path)
            # Otherwise, we know the path file is _not_ directly inside run_path dir.
            # We go as far back as we can while still staying in run_path and symlink that "base" dir.
            # This is because lots of runs create dirs within run_path and it creates too much clutter to symlink every individual file.
            # Further, this avoids an edge case where you both save a file and the dir it's in.
            else:
                # Set base_path such that its parent is run_path.
                base_path = parent_path
                while not parent_path_of_path(base_path).samefile(run_path):
                    base_path = parent_path_of_path(base_path)

                # Create symlink
                open_base_dname = basename_of_path(base_path)
                symlink_path = self.dbgym_this_run_path / name_to_linkname(
                    open_base_dname
                )
                try_remove_file(symlink_path)
                try_create_symlink(base_path, symlink_path)
        # If the file wasn't generated by a run, we can't just symlink it because we don't know that it's immutable.
        else:
            fname = basename_of_path(path)
            # In this case, we want to copy instead of symlinking since it might disappear in the future.
            copy_path = self.dbgym_this_run_path / fname
            shutil.copy(path, copy_path)

    def open_and_save(self, open_path: Path, mode: str = "r") -> IO[Any]:
        """
        Open a file and "save" it to [workspace]/task_runs/run_*/.
        It takes in a str | Path to match the interface of open().
        This file does not work if open_path is a symlink, to make its interface identical to that of open().
            Make sure to resolve all symlinks with fully_resolve_path().
        To avoid confusion, I'm enforcing this function to only work with absolute paths.
        # TODO: maybe make it work on non-fully-resolved paths to better match open()
        See the comment of save_file() for what "saving" means
        If you are generating a "result" for the run, _do not_ use this. Just use the normal open().
            This shouldn't be too hard to remember because this function crashes if open_path doesn't exist,
            and when you write results you're usually opening open_paths which do not exist.
        """
        # Validate open_path
        assert isinstance(open_path, Path)
        assert is_fully_resolved(
            open_path
        ), f"open_and_save(): open_path ({open_path}) should be a fully resolved path"
        assert not os.path.islink(
            open_path
        ), f"open_path ({open_path}) should not be a symlink"
        assert os.path.exists(open_path), f"open_path ({open_path}) does not exist"
        # `open_and_save`` *must* be called on files because it doesn't make sense to open a directory. note that this doesn't mean we'll always save
        #   a file though. we sometimes save a directory (see save_file() for details)
        assert os.path.isfile(open_path), f"open_path ({open_path}) is not a file"

        # Save
        self.save_file(open_path)

        # Open
        return open(open_path, mode=mode)


def get_workspace_path_from_config(dbgym_config_path: Path) -> Path:
    """
    Returns the workspace path (as a fully resolved path) from the config file.
    """
    with open(dbgym_config_path) as f:
        # We do *not* call fully_resolve_path() here because the workspace may not exist yet.
        return Path(yaml.safe_load(f)["dbgym_workspace_path"]).resolve().absolute()


def make_standard_dbgym_workspace() -> DBGymWorkspace:
    """
    The "standard" way to make a DBGymWorkspace using the DBGYM_CONFIG_PATH envvar and the
    default path of dbgym_config.yaml.
    """
    dbgym_config_path = Path(os.getenv("DBGYM_CONFIG_PATH", "dbgym_config.yaml"))
    assert dbgym_config_path == Path(
        "gymlib_package/gymlib/tests/gymlib_integtest_dbgym_config.yaml"
    )
    dbgym_workspace_path = get_workspace_path_from_config(dbgym_config_path)
    dbgym_workspace = DBGymWorkspace(dbgym_workspace_path)
    return dbgym_workspace


def fully_resolve_path(inputpath: os.PathLike[str]) -> Path:
    """
    Fully resolve any path to a real, absolute path.

    For flexibility, we take in any os.PathLike. However, for consistency, we always output a Path object.

    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~.

    Relative paths are relative to the base dbgym repo dir.

    It *does not* check whether the path exists, since the user might be wanting to create a new file/dir.

    Raises RuntimeError for errors.
    """
    # For simplicity, we only process Path objects.
    realabspath = Path(inputpath)
    # `expanduser()` is always "ok" to call first.
    realabspath = realabspath.expanduser()
    # The reason we don't call Path.absolute() is because the path should be relative to get_base_dbgym_repo_path(),
    #   which is not necessary where cwd() points at the time of calling this function.
    if not realabspath.is_absolute():
        realabspath = get_base_dbgym_repo_path() / realabspath
    # `resolve()` has two uses: normalize the path (remove ..) and resolve symlinks.
    # I believe the pathlib library (https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve) does these together this
    #   way to avoid an edge case related to symlinks and normalizing paths (footnote 1 of the linked docs)
    realabspath = realabspath.resolve()
    assert is_fully_resolved(
        realabspath
    ), f"realabspath ({realabspath}) is not fully resolved"
    return realabspath


def get_base_dbgym_repo_path() -> Path:
    path = Path(os.getcwd())
    assert _is_base_dbgym_repo_path(
        path
    ), "This script should be invoked from the root of the dbgym repo."
    return path


def _is_base_dbgym_repo_path(path: Path) -> bool:
    """
    Returns whether we are in the base directory of some git repository
    """
    try:
        git_toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
        ).strip()
        return Path(git_toplevel) == path
    except subprocess.CalledProcessError:
        # This means we are not in _any_ git repo
        return False
    except Exception as e:
        raise e


def is_fully_resolved(path: Path) -> bool:
    """
    Checks if a path is fully resolved (exists, is absolute, and contains no symlinks in its entire ancestry).

    The reason we check for existence is because that's the only way we know that there are no symlinks in its entire ancestry.
    If we didn't check for existence, we could later create a new symlink in the path's ancestry.

    Even if a path exists, is absolute, and is not itself a symlink, it could still contain
    symlinks in its parent directories. For example:
        /home/user/           # Real directory
        /home/user/links/     # Symlink to /data/links
        /home/user/links/file.txt  # Real file

    In this case, "/home/user/links/file.txt" exists and isn't itself a symlink,
    but it's not fully resolved because it contains a symlink in its ancestry.
    The fully resolved path would be "/data/links/file.txt".
    """
    assert isinstance(path, Path)
    resolved_path = path.resolve()

    # Check if the path exists.
    if not resolved_path.exists():
        return False

    # Check if the path contains no symlinks in its entire ancestry.
    # This also checks if the path is absolute because resolved_path is absolute.
    assert (
        resolved_path.is_absolute()
    ), "resolved_path should be absolute (see comment above)"
    # Converting them to strings is the most unambiguously strict way of checking equality.
    # Stuff like Path.__eq__() or Path.samefile() might be more lenient.
    return str(resolved_path) == str(path)


def parent_path_of_path(path: Path) -> Path:
    """
    This function only calls Path.parent, but in a safer way.
    """
    assert isinstance(path, Path)
    assert is_fully_resolved(
        path
    ), f"path must be fully resolved because Path.parent has weird behavior on non-resolved paths (see https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent)"
    parent_path = path.parent
    assert isinstance(parent_path, Path)
    return parent_path


def basename_of_path(path: Path) -> str:
    """
    This function only calls Path.name, but in a safer way.
    """
    assert isinstance(path, Path)
    assert is_fully_resolved(
        path
    ), f'path must be fully resolved because Path.name has weird behavior on non-resolved paths (like giving ".." if the path ends with a "..")'
    path_dirname, path_basename = os.path.split(path)
    # this means the path ended with a '/' so all os.path.split() does is get rid of the slash
    if path_basename == "":
        return os.path.basename(path_dirname)
    else:
        return path_basename


# TODO(phw2): refactor to use Path
def is_child_path(child_path: os.PathLike[str], parent_path: os.PathLike[str]) -> bool:
    """
    Checks whether child_path refers to a file/dir/link that is a child of the dir referred to by parent_path
    If the two paths are equal, this function returns FALSE
    """
    assert os.path.isdir(parent_path)
    if os.path.samefile(child_path, parent_path):
        return False
    else:
        return os.path.samefile(
            os.path.commonpath([parent_path, child_path]), parent_path
        )


def extract_from_task_run_path(
    dbgym_workspace: DBGymWorkspace, task_run_path: Path
) -> tuple[Path, str, Path, str]:
    """
    The task_runs/ folder is organized like task_runs/run_*/[codebase]/[org]/any/path/you/want.
    This function extracts the [codebase] and [org] components
    """
    assert isinstance(task_run_path, Path)
    assert not task_run_path.is_symlink()
    parent_path = task_run_path.parent
    # TODO(phw2): make this a common function
    assert not parent_path.samefile(
        dbgym_workspace.dbgym_runs_path
    ), f"task_run_path ({task_run_path}) should be inside a run_*/ dir instead of directly in dbgym_workspace.dbgym_runs_path ({dbgym_workspace.dbgym_runs_path})"
    assert not parent_path_of_path(parent_path).samefile(
        dbgym_workspace.dbgym_runs_path
    ), f"task_run_path ({task_run_path}) should be inside a run_*/[codebase]/ dir instead of directly in run_*/ ({dbgym_workspace.dbgym_runs_path})"
    assert not parent_path_of_path(parent_path_of_path(parent_path)).samefile(
        dbgym_workspace.dbgym_runs_path
    ), f"task_run_path ({task_run_path}) should be inside a run_*/[codebase]/[organization]/ dir instead of directly in run_*/ ({dbgym_workspace.dbgym_runs_path})"
    # org_path is the run_*/[codebase]/[organization]/ dir that task_run_path is in
    org_path = parent_path
    while not parent_path_of_path(
        parent_path_of_path(parent_path_of_path(org_path))
    ).samefile(dbgym_workspace.dbgym_runs_path):
        org_path = parent_path_of_path(org_path)
    org_dname = basename_of_path(org_path)
    codebase_path = parent_path_of_path(org_path)
    codebase_dname = basename_of_path(codebase_path)

    return codebase_path, codebase_dname, org_path, org_dname


def try_create_symlink(src_path: Path, dst_path: Path) -> None:
    """
    Our functions that create symlinks might be called by multiple processes at once
    during HPO. Thus, this is a thread-safe way to create a symlink.
    """
    assert is_linkname(dst_path.name)
    try:
        os.symlink(src_path, dst_path)
    except FileExistsError:
        # it's ok if it exists
        pass


def try_remove_file(path: Path) -> None:
    """
    Our functions that remove files might be called by multiple processes at once
    during HPO. Thus, this is a thread-safe way to remove a file.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        # it's ok if it doesn't exist
        pass


def is_ssd(path: Path) -> bool:
    try:
        device = (
            subprocess.check_output(["df", path]).decode().split("\n")[1].split()[0]
        )
        device_basename = os.path.basename(device)
        lsblk_output = subprocess.check_output(
            ["lsblk", "-d", "-o", "name,rota"]
        ).decode()
        for line in lsblk_output.split("\n")[1:]:
            parts = line.split()
            if parts and parts[0] == device_basename:
                is_ssd = int(parts[1]) == 0
                return is_ssd
        return False
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False
