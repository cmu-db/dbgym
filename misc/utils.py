import os
import subprocess
import shutil

TUNE_RELPATH = "tune"
PROTOX_RELPATH = f"{TUNE_RELPATH}/protox"
PROTOX_EMBEDDING_RELPATH = f"{PROTOX_RELPATH}/embedding"

def conv_inputpath_to_abspath(inputpath: str) -> str:
    '''
    Convert any user inputted path to an absolute path
    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~
    We assume that the user only ever runs main.py so they will always be in the base repo dir (dbgym/). Thus, all
        relative paths are relative to that
    It *does not* check whether the path exists, since the user might be wanting to create a new file/dir
    Raises RuntimeError for errors
    '''
    # checks
    # regardless of whether the user wants an absolute, relative, or home path, I will do all checks
    # this helps errors surface more quickly
    assert type(inputpath) is str
    if len(inputpath) == 0:
        raise RuntimeError(f'inputpath ({inputpath}) is empty')
    cwd = os.getcwd()
    if not is_in_base_git_dir(cwd):
        raise RuntimeError(f'cwd ({cwd}) is not the base directory of a git repo. Please run main.py from the base dbgym/ directory')

    # logic
    if inputpath[0] == '~':
        return os.path.normpath(os.path.expanduser(inputpath))
    elif inputpath[0] == '/':
        return os.path.normpath(inputpath)
    else:
        return os.path.normpath(os.path.join(cwd, inputpath))

def is_in_base_git_dir(cwd) -> bool:
    '''
    Returns whether we are in the base directory of some git repository
    '''
    try:
        git_toplevel = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], encoding='utf-8').strip()
        return git_toplevel == cwd
    except subprocess.CalledProcessError as e:
        # this means we are not in _any_ git repo
        return False
    
def open_and_save(ctx, fpath, mode="r"):
    '''
    Open a file and "save" it to [workspace]/task_runs/run_*/
    If the file is a symlink, we traverse it until we get to a real file
    "Saving" can mean either copying the file or creating a symlink to it
    We copy the file if it is a "config", meaning it just exists without having been generated
    We create a symlink if it is a "dependency", meaning a task.py command was run to generate it
        In these cases we create a symlink so we have full provenance for how the dependency was created
    '''
    # TODO: traverse symlinks
    # TODO: check config vs dependency
    fname = os.path.basename(fpath)
    shutil.copy(fpath, os.path.join(ctx.obj.dbgym_this_run_path, fname))
    return open(fpath, mode=mode)
