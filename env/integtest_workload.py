import unittest

from benchmark.tpch.constants import DEFAULT_TPCH_SEED
from env.integtest_util import (
    INTEGTEST_BENCHMARK,
    INTEGTEST_SCALE_FACTOR,
    IntegtestWorkspace,
)
from env.workload import Workload
from util.workspace import (
    default_workload_path,
    fully_resolve_path,
    get_default_workload_name_suffix,
    get_workload_name,
)


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    def test_workload(self) -> None:
        workload_dpath = fully_resolve_path(
            IntegtestWorkspace.get_dbgym_cfg(),
            default_workload_path(
                IntegtestWorkspace.get_workspace_path(),
                INTEGTEST_BENCHMARK,
                get_workload_name(
                    INTEGTEST_SCALE_FACTOR,
                    get_default_workload_name_suffix(INTEGTEST_BENCHMARK),
                ),
            ),
        )

        # We just want to make sure no exceptions are thrown.
        workload = Workload(IntegtestWorkspace.get_dbgym_cfg(), workload_dpath)
        workload.get_query(f"S{DEFAULT_TPCH_SEED}-Q1")


if __name__ == "__main__":
    unittest.main()
