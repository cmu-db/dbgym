from pathlib import Path

from util.workspace import DBGymConfig, is_fully_resolved, open_and_save


class Workload:
    def __init__(self, dbgym_cfg: DBGymConfig, workload_dpath: Path) -> None:
        self.dbgym_cfg = dbgym_cfg
        self.workload_dpath = workload_dpath
        assert is_fully_resolved(self.workload_dpath)

        self.queries: dict[str, str] = {}
        order_fpath = self.workload_dpath / "order.txt"
        assert order_fpath.exists()
        with open_and_save(self.dbgym_cfg, order_fpath) as f:
            for line in f:
                qid, qpath = line.strip().split(",")
                qpath = Path(qpath)
                assert is_fully_resolved(qpath)
                with open_and_save(self.dbgym_cfg, qpath) as qf:
                    self.queries[qid] = qf.read()

    def get_query(self, qid: str) -> str:
        return self.queries[qid]
