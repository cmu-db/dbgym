import os
import subprocess
import shutil

from task import DBGymConfig

TUNE_RELPATH = "tune"
PROTOX_RELPATH = os.path.join(TUNE_RELPATH, "protox")
PROTOX_EMBEDDING_RELPATH = os.path.join(PROTOX_RELPATH, "embedding")
DEFAULT_HPO_SPACE_RELPATH = os.path.join(PROTOX_EMBEDDING_RELPATH, "default_hpo_space.json")
WORKSPACE_PLACEHOLDER = "[workspace]"
DATA_PATH_PLACEHOLDER = os.path.join(WORKSPACE_PLACEHOLDER, "data")
BENCHMARK_PLACEHOLDER = "[benchmark]"

# this one is named "*_path" because it could be either a relpath or abspath depending on the data_path arg
default_dataset_path = (lambda data_path, benchmark : os.path.join(data_path, f"{benchmark}_embedding_traindata.parquet"))
# this one is named "*_relpath" because it's always a relative path
# note that the argument corresponding to this is named "*_path" because the argument in principle could take in any kind of path
default_benchmark_config_relpath = (lambda benchmark : os.path.join(PROTOX_RELPATH, f"default_{benchmark}_config.yaml"))

def conv_inputpath_to_abspath(cfg: DBGymConfig, inputpath: os.PathLike) -> str:
    '''
    Convert any user inputted path to an absolute path
    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~
    Relative paths are relative to the base repo dir
    It *does not* check whether the path exists, since the user might be wanting to create a new file/dir
    Raises RuntimeError for errors
    '''
    # expanduser() is always "safe" to call
    # the reason we don't call os.path.abspath() is because the path should be relative to cfg.dbgym_repo_path,
    #   which is not necessary where cwd() points at the time of calling this function
    inputpath = os.path.expanduser(inputpath)
    if os.path.isabs(inputpath):
        return os.path.normpath(inputpath)
    else:
        return os.path.normpath(os.path.join(cfg.dbgym_repo_path, inputpath))

def is_base_git_dir(cwd) -> bool:
    '''
    Returns whether we are in the base directory of some git repository
    '''
    try:
        git_toplevel = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], encoding='utf-8').strip()
        return git_toplevel == cwd
    except subprocess.CalledProcessError as e:
        # this means we are not in _any_ git repo
        return False
    
def parent_dir(dpath: os.PathLike) -> os.PathLike:
    '''
    Return a path of the parent directory of a directory path
    Note that os.path.dirname() does not always return the parent directory (it only does when the path doesn't end with a '/')
    '''
    assert os.path.isdir(dpath) and os.path.isabs(dpath)
    return os.path.abspath(os.path.join(dpath, os.pardir))

def dir_basename(dpath: os.PathLike) -> str:
    '''
    Return the directory name of a directory path
    Note that os.path.basename() does not always return the directory name (it only does when the path doesn't end with a '/')
    '''
    assert os.path.isdir(dpath) and os.path.isabs(dpath)
    dpath_dirname, dpath_basename = os.path.split(dpath)
    # this means the path ended with a '/' so all os.path.split() does is get rid of the slash
    if dpath_basename == '':
        return os.path.basename(dpath_dirname)
    else:
        return dpath_basename

def open_and_save(cfg: DBGymConfig, open_fpath: os.PathLike, mode="r"):
    '''
    Open a file or symlink and "save" it to [workspace]/task_runs/run_*/
    It takes in a str | Path to match the interface of open()
    Note that open() only opens files, not symlinks, so the interface is not exactly the same. Opening symlinks is
        crucial because it means we can change symlink files in [workspace]/data/ instead of changing config files
    "Saving" can mean either copying the file or creating a symlink to it
    We copy the file if it is a "config", meaning it just exists without having been generated
    We create a symlink if it is a "dependency", meaning a task.py command was run to generate it
        In these cases we create a symlink so we have full provenance for how the dependency was created
    If you are generating a "result" for the run, _do not_ use this. Just use the normal open().
        This shouldn't be too hard to remember because this function crashes if open_fpath doesn't exist,
        and when you write results you're usually opening open_fpaths which do not exist
    
    **Notable Behavior**
     - If you open the same "config" file twice in the same run, it'll only be saved the first time (even if the file has changed in between)
        - "Dependency" files should be immutable so there's no problem here
     - If you open two "config" files of the same name but different paths, only the first open will be saved
        - Opening two "dependency" files of the same name but different paths will lead to two different "base dirs" being symlinked
    '''
    # process open_fpath and ensure that it's a file at the end
    open_fpath = conv_inputpath_to_abspath(cfg, open_fpath)
    open_fpath = os.path.realpath(open_fpath) # traverse symlinks
    assert os.path.isfile(open_fpath), f"Error: {open_fpath}"

    # save _something_ to dbgym_this_run_path
    # save a symlink if the opened file was generated by a run. this is for two reasons:
    #   1. files (or dirs) generated by a run are supposed to be immutable so saving a symlink is safe
    #   2. files (or dirs) generated by a run may be very large (up to 100s of GBs) so we don't want to copy them
    if os.path.samefile(os.path.commonpath([cfg.dbgym_runs_path, open_fpath]), cfg.dbgym_runs_path):
        # get open_dpath and open_run_dpath. open_run_dpath is the run_*/ dir that open_fpath is in
        open_dpath = os.path.dirname(open_fpath)
        assert not os.path.samefile(open_dpath, cfg.dbgym_runs_path), f'open_fpath ({open_fpath}) should be inside a run_*/ dir instead of directly in cfg.dbgym_runs_path ({cfg.dbgym_runs_path})'
        open_run_dpath = open_dpath
        while not os.path.samefile(parent_dir(open_run_dpath), cfg.dbgym_runs_path):
            open_run_dpath = parent_dir(open_run_dpath)
        
        # if the open_fpath file is directly in the run_*/ dir, we symlink the file directly
        if os.path.samefile(open_dpath, open_run_dpath):
            open_fname = os.path.basename(open_fpath)
            symlink_fpath = os.path.join(cfg.dbgym_this_run_path, open_fname)
            if not os.path.exists(symlink_fpath):
                os.symlink(open_fpath, symlink_fpath)
        # else, we know the open_fpath file is _not_ directly in the run_*/ dir
        # we go as far back as we can while still staying in the run_*/ and symlink that "base" dir
        # this is because lots of runs create dirs and it's just a waste of space to symlink every individual file
        else:
            # set open_base_dpath such that its parent is the run_*/ dir (meaning its grandparent is dbgym_runs_path)
            open_base_dpath = open_dpath
            while not os.path.samefile(parent_dir(open_base_dpath), open_run_dpath):
                open_base_dpath = parent_dir(open_base_dpath)

            # create symlink
            open_base_dname = dir_basename(open_base_dpath)
            symlink_dpath = os.path.join(cfg.dbgym_this_run_path, open_base_dname)
            if not os.path.exists(symlink_dpath):
                os.symlink(open_base_dpath, symlink_dpath)
    # save a copy if it wasn't generated by a run
    else:
        fname = os.path.basename(open_fpath)
        dpath = conv_inputpath_to_abspath(cfg, cfg.dbgym_this_run_path)
        copy_fpath = os.path.join(dpath, fname)
        shutil.copy(open_fpath, copy_fpath)

    # open
    return open(open_fpath, mode=mode)

def restart_ray():
    '''
    Stop and start Ray.
    This is good to do between each stage to avoid bugs from carrying over across stages
    '''
    os.system('ray stop -f')
    ncpu = os.cpu_count()
    # --disable-usage-stats avoids a Y/N prompt
    os.system(f'OMP_NUM_THREADS={ncpu} ray start --head --num-cpus={ncpu} --disable-usage-stats')
