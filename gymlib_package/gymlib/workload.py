from pathlib import Path

from gymlib.workspace import DBGymWorkspace, is_fully_resolved


class Workload:
    def __init__(self, dbgym_workspace: DBGymWorkspace, workload_path: Path) -> None:
        self.dbgym_workspace = dbgym_workspace
        self.workload_path = workload_path
        assert is_fully_resolved(self.workload_path)

        self.queries: dict[str, str] = {}
        order_path = self.workload_path / "order.txt"
        self.query_order: list[str] = []

        assert order_path.exists()

        with self.dbgym_workspace.open_and_save(order_path) as f:
            for line in f:
                qid, qpath = line.strip().split(",")
                qpath = Path(qpath)
                assert is_fully_resolved(qpath)

                with self.dbgym_workspace.open_and_save(qpath) as qf:
                    self.queries[qid] = qf.read()
                self.query_order.append(qid)

    def get_query(self, qid: str) -> str:
        return self.queries[qid]

    def get_query_order(self) -> list[str]:
        return self.query_order

    def get_queries_in_order(self) -> list[str]:
        return [self.queries[qid] for qid in self.query_order]
