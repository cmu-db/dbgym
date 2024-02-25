import os
import subprocess
import shutil
from pathlib import Path

TUNE_RELPATH = "tune"
PROTOX_RELPATH = f"{TUNE_RELPATH}/protox"
PROTOX_EMBEDDING_RELPATH = f"{PROTOX_RELPATH}/embedding"

def conv_inputpath_to_abspath(ctx, inputpath: str) -> str:
    '''
    Convert any user inputted path to an absolute path
    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~
    Relative paths are relative to the base repo dir
    It *does not* check whether the path exists, since the user might be wanting to create a new file/dir
    Raises RuntimeError for errors
    '''
    # checks
    # regardless of whether the user wants an absolute, relative, or home path, I will do all checks
    # this helps errors surface more quickly
    assert type(inputpath) is str
    if len(inputpath) == 0:
        raise RuntimeError(f'inputpath ({inputpath}) is empty')

    # logic
    if inputpath[0] == '~':
        return os.path.normpath(os.path.expanduser(inputpath))
    elif inputpath[0] == '/':
        return os.path.normpath(inputpath)
    else:
        return os.path.normpath(os.path.join(ctx.obj.dbgym_repo_path, inputpath))

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
    
def open_and_save(ctx, open_fpath: str, mode="r", subfolder=None):
    '''
    Open a file and "save" it to [workspace]/task_runs/run_*/
    It takes in a string for fpath instead of a pathlib.Path in order to match the interface of open()
    If the file is a symlink, we traverse it until we get to a real file
    "Saving" can mean either copying the file or creating a symlink to it
    We copy the file if it is a "config", meaning it just exists without having been generated
    We create a symlink if it is a "dependency", meaning a task.py command was run to generate it
        In these cases we create a symlink so we have full provenance for how the dependency was created
    If you are generating a "result", _do not_ use this. Just use the normal open().
    '''
    # TODO(phw2): traverse symlinks
    # TODO(phw2): check config vs dependency
    # TODO(phw2): add option for saving to subfolder. use that in Workload
    assert type(open_fpath) is str

    # get open_fpath
    open_fpath = conv_inputpath_to_abspath(ctx, open_fpath)
    open_fpath = os.path.realpath(open_fpath) # traverse symlinks

    # get copy_fpath
    fname = os.path.basename(open_fpath)
    # convert to str() because dbgym_this_run_path is a Path
    dpath = conv_inputpath_to_abspath(ctx, str(ctx.obj.dbgym_this_run_path))
    if subfolder != None:
        dpath = os.path.join(dpath, subfolder)
        # we know for a fact that dbgym_this_run_path exists. however, if subfolder != None, dpath may not exist so we should mkdir
        # parents=True because subfolder could have a "/" in it
        # exist_ok=True because we could have called open_and_save() earlier with the same subfolder argument
        Path(dpath).mkdir(parents=True, exist_ok=True)
    copy_fpath = os.path.join(dpath, fname)

    # copy
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
