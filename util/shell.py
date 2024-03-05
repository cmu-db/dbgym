import logging
import os
import subprocess

shell_util_logger = logging.getLogger("shell_util")
shell_util_logger.setLevel(logging.INFO)


def subprocess_run(c, cwd=None, check_returncode=True, dry_run=False, verbose=True):
    cwd_msg = f"(cwd: {cwd if cwd is not None else os.getcwd()})"

    if dry_run:
        shell_util_logger.info(f"Dry run {cwd_msg}: {c}")
        return

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
            for line in proc.stdout:
                if verbose:
                    print(line, end="", flush=True)
            if not loop:
                break
        if check_returncode and proc.returncode != 0:
            raise RuntimeError(f"Non-zero returncode {proc.returncode} for: {c}")

    return proc
