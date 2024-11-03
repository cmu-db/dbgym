import logging
import click

from util.log import DBGYM_LOGGER_NAME
from util.shell import subprocess_run
from util.workspace import DBGymConfig, link_result


@click.group(name="job")
@click.pass_obj
def job_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("job")


@job_group.command(name="data")
@click.pass_obj
# The reason generate data is separate from create dbdata is because generate-data is generic
#   to all DBMSs while create dbdata is specific to a single DBMS.
def job_data(dbgym_cfg: DBGymConfig) -> None:
    _clone(dbgym_cfg)


def _clone(dbgym_cfg: DBGymConfig) -> None:
    expected_symlink_dpath = (
        dbgym_cfg.cur_symlinks_build_path(mkdir=True) / "job-kit-gym.link"
    )
    if expected_symlink_dpath.exists():
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Skipping clone: {expected_symlink_dpath}"
        )
        return

    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloning: {expected_symlink_dpath}")
    real_build_path = dbgym_cfg.cur_task_runs_build_path()
    subprocess_run(
        f"./job_setup.sh {real_build_path}", cwd=dbgym_cfg.cur_source_path()
    )
    symlink_dpath = link_result(dbgym_cfg, real_build_path / "job-kit-gym")
    assert expected_symlink_dpath.samefile(symlink_dpath)
    logging.getLogger(DBGYM_LOGGER_NAME).info(f"Cloned: {expected_symlink_dpath}")