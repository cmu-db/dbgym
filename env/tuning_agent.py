from util.workspace import DBGymConfig


class DBMSConfig:
    def __init__(
        self,
        indexes: list[str],
        sysknobs: dict[str, str],
        query_knobs: dict[str, list[str]],
    ) -> None:
        self.indexes = indexes
        self.sysknobs = sysknobs
        self.query_knobs = query_knobs


class TuningAgent:
    def __init__(self, dbgym_cfg: DBGymConfig) -> None:
        self.dbgym_cfg = dbgym_cfg
        self.next_step_num = 0

    def step(self) -> None:
        dbms_cfg = self._step()
        # TODO: write the config out to a file
        self.next_step_num += 1

    # Subclasses should override this function.
    def _step(self) -> DBMSConfig:
        raise NotImplementedError
