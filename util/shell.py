import logging
import os
from pathlib import Path
import subprocess
from typing import Optional

shell_util_logger = logging.getLogger("shell_util")
shell_util_logger.setLevel(logging.INFO)


def subprocess_run(c: str, cwd: Optional[Path]=None, check_returncode: bool=True, verbose: bool=True) -> subprocess.Popen[str]:
    cwd_msg = f"(cwd: {cwd if cwd is not None else os.getcwd()})"

    if verbose:
        shell_util_logger.info(f"Running {cwd_msg}: {c}")

    with subprocess.Popen(
        c,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=cwd,
        text=True,
        bufsize=0,
    ) as proc:
        while True:
            loop = proc.poll() is None
            assert proc.stdout is not None
            for line in proc.stdout:
                if verbose:
                    print(line, end="", flush=True)
            if not loop:
                break
        if check_returncode and proc.returncode != 0:
            raise RuntimeError(f"Non-zero returncode {proc.returncode} for: {c}")

    return proc
