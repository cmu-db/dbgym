import json
from pathlib import Path
from typing import NewType, TypedDict

from util.workspace import DBGymConfig

# PostgresConn doesn't use these types because PostgresConn is used internally by tuning agents.
# These types are only given as the outputs of tuning agents.
IndexesDelta = NewType("IndexesDelta", list[str])
SysKnobsDelta = NewType("SysKnobsDelta", dict[str, str])
QueryKnobsDelta = NewType("QueryKnobsDelta", dict[str, list[str]])


class DBMSConfigDelta(TypedDict):
    """
    This class represents a DBMS config delta. A "DBMS config" is the indexes, system knobs,
    and query knobs set by the tuning agent. A "delta" is the change from the prior config.

    `indexes` contains a list of SQL statements for creating indexes. Note that since it's a
    config delta, it might contain "DROP ..." statements.

    `sysknobs` contains a mapping from knob names to their values.

    `qknobs` contains a mapping from query IDs to a list of knobs. Each list contains knobs
    to prepend to the start of the query. The knobs are a list[str] instead of a dict[str, str]
    because knobs can be settings ("SET (enable_sort on)") or flags ("IndexOnlyScan(it)").
    """

    indexes: IndexesDelta
    sysknobs: SysKnobsDelta
    qknobs: QueryKnobsDelta


def get_step_delta_fpath(dbms_cfg_deltas_dpath: Path, step_num: int) -> Path:
    return dbms_cfg_deltas_dpath / f"step{step_num}_delta.json"


class TuningAgent:
    def __init__(self, dbgym_cfg: DBGymConfig) -> None:
        self.dbgym_cfg = dbgym_cfg
        self.dbms_cfg_deltas_dpath = self.dbgym_cfg.cur_task_runs_artifacts_path(
            "dbms_cfg_deltas", mkdir=True
        )
        self.next_step_num = 0

    def step(self) -> None:
        """
        This wraps _step() and saves the cfg to a file so that it can be replayed.
        """
        curr_step_num = self.next_step_num
        self.next_step_num += 1
        dbms_cfg_delta = self._step()
        with get_step_delta_fpath(self.dbms_cfg_deltas_dpath, curr_step_num).open(
            "w"
        ) as f:
            json.dump(dbms_cfg_delta, f)

    # Subclasses should override this function.
    def _step(self) -> DBMSConfigDelta:
        """
        This should be overridden by subclasses.

        This should return the delta in the config caused by this step.
        """
        raise NotImplementedError


class TuningAgentStepReader:
    def __init__(self, dbms_cfg_deltas_dpath: Path) -> None:
        self.dbms_cfg_deltas_dpath = dbms_cfg_deltas_dpath
        num_steps = 0
        while get_step_delta_fpath(self.dbms_cfg_deltas_dpath, num_steps).exists():
            num_steps += 1
        self.num_steps = num_steps

    def get_step_delta(self, step_num: int) -> DBMSConfigDelta:
        assert step_num >= 0 and step_num < self.num_steps
        with get_step_delta_fpath(self.dbms_cfg_deltas_dpath, step_num).open("r") as f:
            data = json.load(f)
            return DBMSConfigDelta(
                indexes=data["indexes"],
                sysknobs=data["sysknobs"],
                qknobs=data["qknobs"],
            )

    def get_all_deltas(self) -> list[DBMSConfigDelta]:
        return [self.get_step_delta(step_num) for step_num in range(self.num_steps)]
