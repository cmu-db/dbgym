import unittest

from benchmark.tpch.constants import DEFAULT_TPCH_SEED, NUM_TPCH_QUERIES
from env.integtest_util import (
    INTEGTEST_BENCHMARK,
    INTEGTEST_SCALE_FACTOR,
    GymlibIntegtestManager,
)
from env.workload import Workload
from util.workspace import (
    fully_resolve_path,
    get_default_workload_name_suffix,
    get_default_workload_path,
    get_workload_name,
)


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        GymlibIntegtestManager.set_up_workspace()

    def test_workload(self) -> None:
        workload_dpath = fully_resolve_path(
            GymlibIntegtestManager.get_dbgym_workspace(),
            get_default_workload_path(
                GymlibIntegtestManager.get_workspace_path(),
                INTEGTEST_BENCHMARK,
                get_workload_name(
                    INTEGTEST_SCALE_FACTOR,
                    get_default_workload_name_suffix(INTEGTEST_BENCHMARK),
                ),
            ),
        )

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
