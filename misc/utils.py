import os
import shutil
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable, Tuple, Optional

import redis
import yaml

from util.shell import subprocess_run

# Enums
TuningMode = Enum("TuningMode", ["HPO", "TUNE", "REPLAY"])

# Default values
DEFAULT_WORKLOAD_TIMEOUT = 600

# Relative paths of different folders in the codebase
DBMS_PATH = Path("dbms")
POSTGRES_PATH = DBMS_PATH / "postgres"
TUNE_PATH = Path("tune")
PROTOX_PATH = TUNE_PATH / "protox"
PROTOX_EMBEDDING_PATH = PROTOX_PATH / "embedding"
PROTOX_AGENT_PATH = PROTOX_PATH / "agent"
PROTOX_WOLP_PATH = PROTOX_AGENT_PATH / "wolp"

# Paths of different parts of the workspace
# I made these Path objects even though they're not real paths just so they can work correctly with my other helper functions
WORKSPACE_PATH_PLACEHOLDER = Path("[workspace]")


# Helper functions that both this file and other files use
def get_symlinks_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "symlinks"


def get_tmp_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "tmp"


def get_runs_path_from_workspace_path(workspace_path: Path) -> Path:
    return workspace_path / "task_runs"


def get_scale_factor_string(scale_factor: float | str) -> str:
    if type(scale_factor) is str and scale_factor == SCALE_FACTOR_PLACEHOLDER:
        return scale_factor
    else:
        if float(int(scale_factor)) == scale_factor:
            return str(int(scale_factor))
        else:
            return str(scale_factor).replace(".", "point")


def get_dbdata_tgz_name(benchmark_name: str, scale_factor: float | str) -> str:
    return f"{benchmark_name}_sf{get_scale_factor_string(scale_factor)}_pristine_dbdata.tgz"


# Other parameters
BENCHMARK_NAME_PLACEHOLDER: str = "[benchmark_name]"
WORKLOAD_NAME_PLACEHOLDER: str = "[workload_name]"
SCALE_FACTOR_PLACEHOLDER: str = "[scale_factor]"

# Paths of config files in the codebase. These are always relative paths.
# The reason these can be relative paths instead of functions taking in codebase_path as input is because relative paths are relative to the codebase root
DEFAULT_HPO_SPACE_PATH = PROTOX_EMBEDDING_PATH / "default_hpo_space.json"
DEFAULT_SYSKNOBS_PATH = PROTOX_AGENT_PATH / "default_sysknobs.yaml"
DEFAULT_BOOT_CONFIG_FPATH = POSTGRES_PATH / "default_boot_config.yaml"
default_benchmark_config_path: Callable[[str], Path] = (
    lambda benchmark_name: PROTOX_PATH
    / f"default_{benchmark_name}_benchmark_config.yaml"
)
default_benchbase_config_path: Callable[[str], Path] = (
    lambda benchmark_name: PROTOX_PATH
    / f"default_{benchmark_name}_benchbase_config.xml"
)

# Generally useful functions
workload_name_fn: Callable[[float | str, int, int, str], str] = (
    lambda scale_factor, seed_start, seed_end, query_subset: f"workload_sf{get_scale_factor_string(scale_factor)}_{seed_start}_{seed_end}_{query_subset}"
)

# Standard names of files/directories. These can refer to either the actual file/directory or a link to the file/directory.
#   Since they can refer to either the actual or the link, they do not have ".link" in them.
traindata_fname: Callable[[str, str], str] = (
    lambda benchmark_name, workload_name: f"{benchmark_name}_{workload_name}_embedding_traindata.parquet"
)
default_embedder_dname: Callable[[str, str], str] = (
    lambda benchmark_name, workload_name: f"{benchmark_name}_{workload_name}_embedder"
)
default_hpoed_agent_params_fname: Callable[[str, str], str] = (
    lambda benchmark_name, workload_name: f"{benchmark_name}_{workload_name}_hpoed_agent_params.json"
)
default_tuning_steps_dname: Callable[[str, str, bool], str] = (
    lambda benchmark_name, workload_name, boot_enabled_during_tune: f"{benchmark_name}_{workload_name}{'_boot' if boot_enabled_during_tune else ''}_tuning_steps"
)

# Paths of dependencies in the workspace. These are named "*_path" because they will be an absolute path
# The reason these _cannot_ be relative paths is because relative paths are relative to the codebase root, not the workspace root
# Note that it's okay to hardcode the codebase paths (like dbgym_dbms_postgres) here. In the worst case, we'll just break an
#   integration test. The "source of truth" of codebase paths is based on DBGymConfig.cur_source_path(), which will always
#   reflect the actual codebase structure. As long as we automatically enforce getting the right codebase paths when writing, it's
#   ok to have to hardcode them when reading.
# Details
#  - If a name already has the workload_name, I omit scale factor. This is because the workload_name includes the scale factor
#  - By convention, symlinks should end with ".link". The bug that motivated this decision involved replaying a tuning run. When
#    replaying a tuning run, you read the tuning_steps/ folder of the tuning run. Earlier, I created a symlink to that tuning_steps/
#    folder called run_*/dbgym_agent_protox_tune/tuning_steps. However, replay itself generates an output.log file, which goes in
#    run_*/dbgym_agent_protox_tune/tuning_steps/. The bug was that my replay function was overwriting the output.log file of the
#    tuning run. By naming all symlinks "*.link", we avoid the possibility of subtle bugs like this happening.
default_traindata_path: Callable[[Path, str, str], Path] = (
    lambda workspace_path, benchmark_name, workload_name: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / "dbgym_tune_protox_embedding"
    / "data"
    / (traindata_fname(benchmark_name, workload_name) + ".link")
)
default_embedder_path: Callable[[Path, str, str], Path] = (
    lambda workspace_path, benchmark_name, workload_name: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / "dbgym_tune_protox_embedding"
    / "data"
    / (default_embedder_dname(benchmark_name, workload_name) + ".link")
)
default_hpoed_agent_params_path: Callable[[Path, str, str], Path] = (
    lambda workspace_path, benchmark_name, workload_name: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / "dbgym_tune_protox_agent"
    / "data"
    / (default_hpoed_agent_params_fname(benchmark_name, workload_name) + ".link")
)
default_workload_path: Callable[[Path, str, str], Path] = (
    lambda workspace_path, benchmark_name, workload_name: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / f"dbgym_benchmark_{benchmark_name}"
    / "data"
    / (workload_name + ".link")
)
default_pristine_dbdata_snapshot_path: Callable[[Path, str, float | str], Path] = (
    lambda workspace_path, benchmark_name, scale_factor: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / "dbgym_dbms_postgres"
    / "data"
    / (get_dbdata_tgz_name(benchmark_name, scale_factor) + ".link")
)
default_dbdata_parent_dpath: Callable[[Path], Path] = (
    lambda workspace_path: get_tmp_path_from_workspace_path(workspace_path)
)
default_pgbin_path: Callable[[Path], Path] = (
    lambda workspace_path: get_symlinks_path_from_workspace_path(workspace_path)
    / "dbgym_dbms_postgres"
    / "build"
    / "repo.link"
    / "boot"
    / "build"
    / "postgres"
    / "bin"
)
default_tuning_steps_dpath: Callable[[Path, str, str, bool], Path] = (
    lambda workspace_path, benchmark_name, workload_name, boot_enabled_during_tune: get_symlinks_path_from_workspace_path(
        workspace_path
    )
    / "dbgym_tune_protox_agent"
    / "artifacts"
    / (
        default_tuning_steps_dname(
            benchmark_name, workload_name, boot_enabled_during_tune
        )
        + ".link"
    )
)


class DBGymConfig:
    """
    Global configurations that apply to all parts of DB-Gym
    """

    num_times_created_this_run: int = 0

    def __init__(self, dbgym_config_path: Path):
        # The logic around dbgym_tmp_path assumes that DBGymConfig is only constructed once.
        DBGymConfig.num_times_created_this_run += 1
        assert (
            DBGymConfig.num_times_created_this_run == 1
        ), f"DBGymConfig has been created {DBGymConfig.num_times_created_this_run} times. It should only be created once per run."

        assert is_base_git_dir(
            os.getcwd()
        ), "This script should be invoked from the root of the dbgym repo."

        # Parse the YAML file.
        contents: str = dbgym_config_path.read_text()
        yaml_config: dict[str, Any] = yaml.safe_load(contents)

        # Require dbgym_workspace_path to be absolute.
        # All future paths should be constructed from dbgym_workspace_path.
        dbgym_workspace_path = (
            Path(yaml_config["dbgym_workspace_path"]).resolve().absolute()
        )

        self.path: Path = dbgym_config_path
        self.cur_path_list: list[str] = ["dbgym"]
        self.root_yaml: dict[str, Any] = yaml_config
        self.cur_yaml: dict[str, Any] = self.root_yaml

        # Set and create paths.
        self.dbgym_repo_path = Path(os.getcwd())
        self.dbgym_workspace_path = dbgym_workspace_path
        self.dbgym_workspace_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_runs_path = self.dbgym_workspace_path / "task_runs"
        self.dbgym_runs_path.mkdir(parents=True, exist_ok=True)
        self.dbgym_symlinks_path = get_symlinks_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        self.dbgym_symlinks_path.mkdir(parents=True, exist_ok=True)
        # tmp/ is a workspace for this run only
        # One use for it is to place the unzipped dbdata.
        # There's no need to save the actual dbdata dir in run_*/ because we just save a symlink to
        #   the .tgz file we unzipped.
        self.dbgym_tmp_path = get_tmp_path_from_workspace_path(
            self.dbgym_workspace_path
        )
        # The best place to delete the old dbgym_tmp_path is in DBGymConfig.__init__().
        # This is better than deleting the dbgym_tmp_path is in DBGymConfig.__del__() because DBGymConfig may get deleted before execution has completed.
        # Also, by keeping the tmp directory around, you can look at it to debug issues.
        if self.dbgym_tmp_path.exists():
            shutil.rmtree(self.dbgym_tmp_path)
        self.dbgym_tmp_path.mkdir(parents=True, exist_ok=True)

        # Set the path for this task run's results.
        self.dbgym_this_run_path = (
            self.dbgym_runs_path / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        # `exist_ok` is False because we don't want to override a previous task run's data.
        self.dbgym_this_run_path.mkdir(parents=True, exist_ok=False)

    # `append_group()` is used to mark the "codebase path" of an invocation of the CLI. The "codebase path" is
    #   explained further in the documentation.
    def append_group(self, name: str) -> None:
        self.cur_path_list.append(name)
        self.cur_yaml = self.cur_yaml.get(name, {})

    def cur_source_path(self, *dirs: str) -> Path:
        cur_path = self.dbgym_repo_path
        assert self.cur_path_list[0] == "dbgym"
        for folder in self.cur_path_list[1:]:
            cur_path = cur_path / folder
        for dir in dirs:
            cur_path = cur_path / dir
        return cur_path

    def cur_symlinks_path(self, *dirs: str, mkdir: bool=False) -> Path:
        flattened_structure = "_".join(self.cur_path_list)
        cur_path = self.dbgym_symlinks_path / flattened_structure
        for dir in dirs:
            cur_path = cur_path / dir
        if mkdir:
            cur_path.mkdir(parents=True, exist_ok=True)
        return cur_path

    def cur_task_runs_path(self, *dirs: str, mkdir: bool=False) -> Path:
        flattened_structure = "_".join(self.cur_path_list)
        cur_path = self.dbgym_this_run_path / flattened_structure
        for dir in dirs:
            cur_path = cur_path / dir
        if mkdir:
            cur_path.mkdir(parents=True, exist_ok=True)
        return cur_path

    def cur_symlinks_bin_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_symlinks_path("bin", *dirs, mkdir=mkdir)

    def cur_symlinks_build_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_symlinks_path("build", *dirs, mkdir=mkdir)

    def cur_symlinks_data_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_symlinks_path("data", *dirs, mkdir=mkdir)

    def cur_task_runs_build_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_task_runs_path("build", *dirs, mkdir=mkdir)

    def cur_task_runs_data_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_task_runs_path("data", *dirs, mkdir=mkdir)

    def cur_task_runs_artifacts_path(self, *dirs: str, mkdir: bool=False) -> Path:
        return self.cur_task_runs_path("artifacts", *dirs, mkdir=mkdir)


def conv_inputpath_to_realabspath(
    dbgym_cfg: DBGymConfig, inputpath: os.PathLike[str]
) -> Path:
    """
    Convert any user inputted path to a real, absolute path
    For flexibility, we take in any os.PathLike. However, for consistency, we always output a Path object
    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~
    Relative paths are relative to the base dbgym repo dir
    It *does not* check whether the path exists, since the user might be wanting to create a new file/dir
    Raises RuntimeError for errors
    """
    # For simplicity, we only process Path objects.
    realabspath = Path(inputpath)
    # `expanduser()` is always "ok" to call first.
    realabspath = realabspath.expanduser()
    # The reason we don't call Path.absolute() is because the path should be relative to dbgym_cfg.dbgym_repo_path,
    #   which is not necessary where cwd() points at the time of calling this function.
    if not realabspath.is_absolute():
        realabspath = dbgym_cfg.dbgym_repo_path / realabspath
    # `resolve()` has two uses: normalize the path (remove ..) and resolve symlinks.
    # I believe the pathlib library (https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve) does it this
    #   way to avoid an edge case related to symlinks and normalizing paths (footnote 1 of the linked docs)
    realabspath = realabspath.resolve()
    assert (
        realabspath.is_absolute()
    ), f"after being processed, realabspath ({realabspath}) is still not absolute"
    assert (
        realabspath.exists()
    ), f"after being processed, realabspath ({realabspath}) is still a non-existent path"
    return realabspath


def is_base_git_dir(cwd: str) -> bool:
    """
    Returns whether we are in the base directory of some git repository
    """
    try:
        git_toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
        ).strip()
        return git_toplevel == cwd
    except subprocess.CalledProcessError as e:
        # this means we are not in _any_ git repo
        return False


def is_fully_resolved(path: Path) -> bool:
    assert isinstance(path, Path)
    resolved_path = path.resolve()
    # Converting them to strings is the most unambiguously strict way of checking equality.
    # Stuff like Path.__eq__() or Path.samefile() might be more lenient.
    return str(resolved_path) == str(path)


def path_exists_dont_follow_symlinks(path: Path) -> bool:
    """
    As of writing this comment, ray is currently constraining us to python <3.12. However, the "follow_symlinks" option in
    Path.exists() only comes up in python 3.12. Thus, this is the only way to check if a path exists without following symlinks.
    """
    # If the path exists and is a symlink, os.path.islink() will be true (even if the symlink is broken)
    if os.path.islink(path):
        return True
    # Otherwise, we know it's either non-existent or not a symlink, so path.exists() works fine
    else:
        return path.exists()


def parent_dpath_of_path(dpath: Path) -> Path:
    """
    This function only calls Path.parent, but in a safer way.
    """
    assert isinstance(dpath, Path)
    assert is_fully_resolved(
        dpath
    ), f"dpath must be fully resolved because Path.parent has weird behavior on non-resolved paths (see https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent)"
    parent_dpath = dpath.parent
    assert isinstance(parent_dpath, Path)
    return parent_dpath


def basename_of_path(dpath: Path) -> str:
    """
    This function only calls Path.name, but in a safer way.
    """
    assert isinstance(dpath, Path)
    assert is_fully_resolved(
        dpath
    ), f'dpath must be fully resolved because Path.name has weird behavior on non-resolved paths (like giving ".." if the path ends with a "..")'
    dpath_dirname, dpath_basename = os.path.split(dpath)
    # this means the path ended with a '/' so all os.path.split() does is get rid of the slash
    if dpath_basename == "":
        return os.path.basename(dpath_dirname)
    else:
        return dpath_basename


# TODO(phw2): refactor to use Path
def is_child_path(child_path: os.PathLike[str], parent_dpath: os.PathLike[str]) -> bool:
    """
    Checks whether child_path refers to a file/dir/link that is a child of the dir referred to by parent_dpath
    If the two paths are equal, this function returns FALSE
    """
    assert os.path.isdir(parent_dpath)
    if os.path.samefile(child_path, parent_dpath):
        return False
    else:
        return os.path.samefile(
            os.path.commonpath([parent_dpath, child_path]), parent_dpath
        )


def open_and_save(dbgym_cfg: DBGymConfig, open_fpath: Path, mode: str="r") -> IO[Any]:
    """
    Open a file and "save" it to [workspace]/task_runs/run_*/.
    It takes in a str | Path to match the interface of open().
    This file does not work if open_fpath is a symlink, to make its interface identical to that of open().
        Make sure to resolve all symlinks with conv_inputpath_to_realabspath().
    To avoid confusion, I'm enforcing this function to only work with absolute paths.
    See the comment of save_file() for what "saving" means
    If you are generating a "result" for the run, _do not_ use this. Just use the normal open().
        This shouldn't be too hard to remember because this function crashes if open_fpath doesn't exist,
        and when you write results you're usually opening open_fpaths which do not exist.

    **Notable Behavior**
     - If you open the same "config" file twice in the same run, it'll only be saved the first time (even if the file has changed in between).
        - "Dependency" files should be immutable so there's no problem here.
     - If you open two "config" files of the same name but different paths, only the first open will be saved.
        - Opening two "dependency" files of the same name but different paths will lead to two different "base dirs" being symlinked.
    """
    # validate open_fpath
    assert isinstance(open_fpath, Path)
    assert is_fully_resolved(
        open_fpath
    ), f"open_and_save(): open_fpath ({open_fpath}) should be a fully resolved path"
    assert not os.path.islink(
        open_fpath
    ), f"open_fpath ({open_fpath}) should not be a symlink"
    assert os.path.exists(open_fpath), f"open_fpath ({open_fpath}) does not exist"
    # open_and_save *must* be called on files because it doesn't make sense to open a directory. note that this doesn't mean we'll always save
    #   a file though. we sometimes save a directory (see save_file() for details)
    assert os.path.isfile(open_fpath), f"open_fpath ({open_fpath}) is not a file"

    # save
    save_file(dbgym_cfg, open_fpath)

    # open
    return open(open_fpath, mode=mode)


def extract_from_task_run_fordpath(
    dbgym_cfg: DBGymConfig, task_run_fordpath: Path
) -> Tuple[Path, str, Path, str]:
    """
    The task_runs/ folder is organized like task_runs/run_*/[codebase]/[org]/any/path/you/want.
    This function extracts the [codebase] and [org] components
    """
    assert isinstance(task_run_fordpath, Path)
    assert not task_run_fordpath.is_symlink()
    parent_dpath = task_run_fordpath.parent
    # TODO(phw2): make this a common function
    assert not parent_dpath.samefile(
        dbgym_cfg.dbgym_runs_path
    ), f"task_run_fordpath ({task_run_fordpath}) should be inside a run_*/ dir instead of directly in dbgym_cfg.dbgym_runs_path ({dbgym_cfg.dbgym_runs_path})"
    assert not parent_dpath_of_path(parent_dpath).samefile(
        dbgym_cfg.dbgym_runs_path
    ), f"task_run_fordpath ({task_run_fordpath}) should be inside a run_*/[codebase]/ dir instead of directly in run_*/ ({dbgym_cfg.dbgym_runs_path})"
    assert not parent_dpath_of_path(parent_dpath_of_path(parent_dpath)).samefile(
        dbgym_cfg.dbgym_runs_path
    ), f"task_run_fordpath ({task_run_fordpath}) should be inside a run_*/[codebase]/[organization]/ dir instead of directly in run_*/ ({dbgym_cfg.dbgym_runs_path})"
    # org_dpath is the run_*/[codebase]/[organization]/ dir that task_run_fordpath is in
    org_dpath = parent_dpath
    while not parent_dpath_of_path(
        parent_dpath_of_path(parent_dpath_of_path(org_dpath))
    ).samefile(dbgym_cfg.dbgym_runs_path):
        org_dpath = parent_dpath_of_path(org_dpath)
    org_dname = basename_of_path(org_dpath)
    codebase_dpath = parent_dpath_of_path(org_dpath)
    codebase_dname = basename_of_path(codebase_dpath)

    return codebase_dpath, codebase_dname, org_dpath, org_dname


# TODO(phw2): really look at the clean PR to see what it changed
# TODO(phw2): after merging agent-train, refactor some code in agent-train to use save_file() instead of open_and_save()
def save_file(dbgym_cfg: DBGymConfig, fpath: Path) -> None:
    """
    If an external function takes in a file/directory as input, you will not be able to call open_and_save().
        In these situations, just call save_file().
    Like open_and_save(), this function only works with real absolute paths.
    "Saving" can mean either copying the file or creating a symlink to it
    We copy the file if it is a "config", meaning it just exists without having been generated
    We create a symlink if it is a "dependency", meaning a task.py command was run to generate it
        In these cases we create a symlink so we have full provenance for how the dependency was created
    """
    # validate fpath
    assert isinstance(fpath, Path)
    assert not os.path.islink(fpath), f"fpath ({fpath}) should not be a symlink"
    assert os.path.exists(fpath), f"fpath ({fpath}) does not exist"
    assert os.path.isfile(fpath), f"fpath ({fpath}) is not a file"
    assert not is_child_path(
        fpath, dbgym_cfg.dbgym_this_run_path
    ), f"fpath ({fpath}) was generated in this task run ({dbgym_cfg.dbgym_this_run_path}). You do not need to save it"

    # save _something_ to dbgym_this_run_path
    # save a symlink if the opened file was generated by a run. this is for two reasons:
    #   1. files or dirs generated by a run are supposed to be immutable so saving a symlink is safe
    #   2. files or dirs generated by a run may be very large (up to 100s of GBs) so we don't want to copy them
    if is_child_path(fpath, dbgym_cfg.dbgym_runs_path):
        # get paths we'll need later.
        _, codebase_dname, org_dpath, org_dname = extract_from_task_run_fordpath(
            dbgym_cfg, fpath
        )
        this_run_save_dpath = dbgym_cfg.dbgym_this_run_path / codebase_dname / org_dname
        os.makedirs(this_run_save_dpath, exist_ok=True)

        # if the fpath file is directly in org_dpath, we symlink the file directly
        parent_dpath = parent_dpath_of_path(fpath)
        if parent_dpath.samefile(org_dpath):
            fname = basename_of_path(fpath)
            symlink_fpath = this_run_save_dpath / (fname + ".link")
            try_create_symlink(fpath, symlink_fpath)
        # else, we know the fpath file is _not_ directly inside org_dpath dir
        # we go as far back as we can while still staying in org_dpath and symlink that "base" dir
        # this is because lots of runs create dirs within org_dpath and it's just a waste of space to symlink every individual file
        else:
            # set base_dpath such that its parent is org_dpath
            base_dpath = parent_dpath
            while not parent_dpath_of_path(base_dpath).samefile(org_dpath):
                base_dpath = parent_dpath_of_path(base_dpath)

            # create symlink
            open_base_dname = basename_of_path(base_dpath)
            symlink_dpath = this_run_save_dpath / (open_base_dname + ".link")
            try_create_symlink(base_dpath, symlink_dpath)
    # if it wasn't generated by a run
    else:
        # since we don't know where the file is at all, the location is "unknown" and the org is "all"
        this_run_save_dpath = dbgym_cfg.dbgym_this_run_path / "unknown" / "all"
        os.makedirs(this_run_save_dpath, exist_ok=True)
        fname = basename_of_path(fpath)
        # in this case, we want to copy instead of symlinking since it might disappear in the future
        copy_fpath = this_run_save_dpath / fname
        shutil.copy(fpath, copy_fpath)


# TODO(phw2): refactor our manual symlinking in postgres/cli.py to use link_result() instead
def link_result(
    dbgym_cfg: DBGymConfig, result_fordpath: Path, custom_result_name: Optional[str] = None
) -> Path:
    """
    result_fordpath must be a "result", meaning it was generated inside dbgym_cfg.dbgym_this_run_path.
    Further, result_fordpath must have been generated by this invocation to task.py. This also means that
        result_fordpath itself can be a file or a dir but not a symlink.
    Given a file or directory in task_runs/run_*/[codebase]/[org], this will create a symlink inside
        symlinks/[codebase]/[org]/.
    Will override the old symlink if there is one, so that symlinks/ always contains the latest generated
        version of a file.
    This function will return the path to the symlink that was created.
    """
    assert isinstance(result_fordpath, Path)
    assert is_fully_resolved(
        result_fordpath
    ), f"result_fordpath ({result_fordpath}) should be a fully resolved path"
    result_fordpath = conv_inputpath_to_realabspath(dbgym_cfg, result_fordpath)
    assert is_child_path(result_fordpath, dbgym_cfg.dbgym_this_run_path)
    assert not os.path.islink(result_fordpath)

    if type(custom_result_name) is str:
        result_name = custom_result_name
    else:
        if os.path.isfile(result_fordpath):
            result_name = basename_of_path(result_fordpath) + ".link"
        elif os.path.isdir(result_fordpath):
            result_name = basename_of_path(result_fordpath) + ".link"
        else:
            raise AssertionError("result_fordpath must be either a file or dir")

    # Figure out the parent directory path of the symlink
    codebase_dpath, codebase_dname, _, org_dname = extract_from_task_run_fordpath(
        dbgym_cfg, result_fordpath
    )
    # We're only supposed to save files generated by us, which means they should be in cur_task_runs_path()
    assert codebase_dpath.samefile(
        dbgym_cfg.cur_task_runs_path()
    ), f"link_result should only be called on files generated by this invocation to task.py"
    symlink_parent_dpath = dbgym_cfg.dbgym_symlinks_path / codebase_dname / org_dname
    symlink_parent_dpath.mkdir(parents=True, exist_ok=True)

    # Remove the old symlink ("old" meaning created in an earlier run) if there is one
    # Note that in a multi-threaded setting, this might remove one created by a process in the same run,
    #   meaning it's not "old" by our definition of "old". However, we'll always end up with a symlink
    #   file of the current run regardless of the order of threads.
    assert result_name.endswith(".link") and not result_name.endswith(
        ".link.link"
    ), f'result_name ({result_name}) should end with ".link"'
    symlink_path = symlink_parent_dpath / result_name
    try_remove_file(symlink_path)
    try_create_symlink(result_fordpath, symlink_path)

    return symlink_path


def try_create_symlink(src_path: Path, dst_path: Path) -> None:
    """
    Our functions that create symlinks might be called by multiple processes at once
    during HPO. Thus, this is a thread-safe way to create a symlink.
    """
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


def restart_ray(redis_port: int) -> None:
    """
    Stop and start Ray.
    This is good to do between each stage to avoid bugs from carrying over across stages
    """
    subprocess_run("ray stop -f")
    ncpu = os.cpu_count()
    # --disable-usage-stats avoids a Y/N prompt
    subprocess_run(
        f"OMP_NUM_THREADS={ncpu} ray start --head --port={redis_port} --num-cpus={ncpu} --disable-usage-stats"
    )


def make_redis_started(port: int) -> None:
    """
    Start Redis if it's not already started.
    Note that Ray uses Redis but does *not* use this function. It starts Redis on its own.
    One current use for this function to start/stop Redis for Boot.
    """
    try:
        r = redis.Redis(port=port)
        r.ping()
        # This means Redis is running, so we do nothing
        do_start_redis = False
    except (redis.ConnectionError, redis.TimeoutError):
        # This means Redis is not running, so we start it
        do_start_redis = True

    # I'm starting Redis outside of except so that errors in r.ping get propagated correctly
    if do_start_redis:
        subprocess_run(f"redis-server --port {port} --daemonize yes")
        # When you start Redis in daemon mode, it won't let you know if it's started, so we ping again to check
        r = redis.Redis(port=port)
        r.ping()


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
        print(f"An error occurred: {e}")
        return False
