import os
import subprocess

def conv_inputpath_to_abspath(inputpath: str) -> str:
    '''
    Convert any user inputted path to an absolute path
    Whenever a path is required, the user is allowed to enter relative paths, absolute paths, or paths starting with ~
    We assume that the user only ever runs main.py so they will always be in the base repo dir (dbgym/). Thus, all
        relative paths are relative to that
    Raises FileNotFoundError if the path is malformed or doesn't exist
    Raises RuntimeError for other errors
    '''
    assert type(inputpath) is str
    if len(inputpath) == 0:
        raise FileNotFoundError(f'inputpath ({inputpath}) is empty')
    cwd = os.getcwd()
    if not is_in_base_git_dir(cwd):
        raise RuntimeError(f'cwd ({cwd}) is not the base directory of a git repo. Please run main.py from the base dbgym/ directory')

def is_in_base_git_dir(cwd) -> bool:
    '''
    Returns whether we are in the base directory of some git repository
    '''
    try:
        git_top_level = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], encoding='utf-8').strip()
        return git_top_level == cwd
    except subprocess.CalledProcessError as e:
        # this means we are not in _any_ git repo
        return False
