from util.workspace import DBGymConfig


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
    def __init__(
        self,
        indexes: list[str],
        sysknobs: dict[str, str],
        qknobs: dict[str, list[str]],
    ) -> None:
        self.indexes = indexes
        self.sysknobs = sysknobs
        self.qknobs = qknobs


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
    
    def get_past_config(self, step_num: int) -> DBMSConfig:
        assert step_num >= 0 and step_num < self.next_step_num
