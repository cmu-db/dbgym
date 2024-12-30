import unittest

from gymlib.tests.gymlib_integtest_util import GymlibIntegtestManager
from gymlib.workload import Workload
from gymlib.workspace import DBGymWorkspace

from benchmark.tpch.constants import DEFAULT_TPCH_SEED, NUM_TPCH_QUERIES


class WorkloadTests(unittest.TestCase):
    workspace: DBGymWorkspace

    @staticmethod
    def setUpClass() -> None:
        GymlibIntegtestManager.set_up_workspace()
        # Reset _num_times_created_this_run since previous tests may have created a workspace.
        DBGymWorkspace._num_times_created_this_run = 0
        WorkloadTests.workspace = DBGymWorkspace(
            GymlibIntegtestManager.get_workspace_path()
        )

    def test_workload(self) -> None:
        workload_path = GymlibIntegtestManager.get_default_metadata().workload_path
        workload = Workload(WorkloadTests.workspace, workload_path)

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
