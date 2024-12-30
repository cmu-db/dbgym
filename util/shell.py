import logging
import os
import subprocess
from pathlib import Path
from typing import Optional


def subprocess_run(
    c: str,
    cwd: Optional[Path] = None,
    check_returncode: bool = True,
    verbose: bool = True,
) -> subprocess.Popen[str]:
    cwd_msg = f"(cwd: {cwd if cwd is not None else os.getcwd()})"

    if verbose:
        logging.info(f"Running {cwd_msg}: {c}")

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
                    logging.info(line)
            if not loop:
                break
        if check_returncode and proc.returncode != 0:
            raise RuntimeError(f"Non-zero returncode {proc.returncode} for: {c}")

    return proc
