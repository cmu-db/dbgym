from dataclasses import asdict, dataclass
import json
from pathlib import Path
from util.workspace import DBGymConfig


@dataclass
class DBMSConfig:
    """
    This class can either represent a config or a config delta.

    `indexes` contains a list of SQL statements for creating indexes. If you're using the class
    as a config delta, it also might contain "DROP ..." statements.

    `sysknobs` contains a mapping from knob names to their values.

    `qknobs` contains a mapping from query IDs to a list of knobs. Each list contains knobs
    to prepend to the start of the query. The knobs are a list[str] instead of a dict[str, str]
    because knobs can be settings ("SET (enable_sort on)") or flags ("IndexOnlyScan(it)").
    """
    indexes: list[str]
    sysknobs: dict[str, str]
    qknobs: dict[str, list[str]]


class TuningAgent:
    def __init__(self, dbgym_cfg: DBGymConfig) -> None:
        self.dbgym_cfg = dbgym_cfg
        self.dbms_cfg_deltas_dpath = self.dbgym_cfg.cur_task_runs_artifacts_path("dbms_cfg_deltas", mkdir=True)
        self.next_step_num = 0

    def step(self) -> None:
        """
        This wraps _step() and saves the cfg to a file so that it can be replayed.
        """
        curr_step_num = self.next_step_num
        self.next_step_num += 1
        dbms_cfg_delta = self._step()
        with self.get_step_delta_fpath(curr_step_num).open("w") as f:
            json.dump(asdict(dbms_cfg_delta), f)

    def get_step_delta_fpath(self, step_num: int) -> Path:
        return self.dbms_cfg_deltas_dpath / f"step{step_num}_delta.json"

    # Subclasses should override this function.
    def _step(self) -> DBMSConfig:
        """
        This should be overridden by subclasses.

        This should return the delta in the config caused by this step.
        """
        raise NotImplementedError
    
    def get_step_delta(self, step_num: int) -> DBMSConfig:
        assert step_num >= 0 and step_num < self.next_step_num
        delta = None
        with self.get_step_delta_fpath(step_num).open("r") as f:
            delta = DBMSConfig(**json.load(f))
        assert delta is not None
        return delta
