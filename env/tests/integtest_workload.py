import unittest

from benchmark.tpch.constants import DEFAULT_TPCH_SEED, NUM_TPCH_QUERIES
from env.tests.gymlib_integtest_util import GymlibIntegtestManager
from env.workload import Workload
from util.workspace import fully_resolve_path, get_default_workload_path


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        GymlibIntegtestManager.set_up_workspace()

    def test_workload(self) -> None:
        workload_dpath = GymlibIntegtestManager.get_default_metadata().workload_path

        workload = Workload(
            GymlibIntegtestManager.get_dbgym_workspace(), workload_dpath
        )

        # Check the order of query IDs.
        self.assertEqual(
            workload.get_query_order(),
            [f"S{DEFAULT_TPCH_SEED}-Q{i}" for i in range(1, NUM_TPCH_QUERIES + 1)],
        )

        # Sanity check all queries.
        for query in workload.get_queries_in_order():
            self.assertTrue("select" in query.lower())


if __name__ == "__main__":
    unittest.main()
