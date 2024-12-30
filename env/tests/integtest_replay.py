import unittest

from benchmark.tpch.constants import DEFAULT_TPCH_SEED
from env.replay import replay
from env.tests.gymlib_integtest_util import GymlibIntegtestManager
from env.tuning_artifacts import (
    DBMSConfigDelta,
    IndexesDelta,
    QueryKnobsDelta,
    SysKnobsDelta,
    TuningArtifactsWriter,
)
from util.workspace import DBGymWorkspace


class ReplayTests(unittest.TestCase):
    workspace: DBGymWorkspace

    @staticmethod
    def setUpClass() -> None:
        GymlibIntegtestManager.set_up_workspace()
        # Reset _num_times_created_this_run since previous tests may have created a workspace.
        DBGymWorkspace._num_times_created_this_run = 0
        ReplayTests.workspace = DBGymWorkspace(
            GymlibIntegtestManager.get_workspace_path()
        )

    def test_replay(self) -> None:
        writer = TuningArtifactsWriter(
            ReplayTests.workspace,
            GymlibIntegtestManager.get_default_metadata(),
        )
        writer.write_step(
            DBMSConfigDelta(
                indexes=IndexesDelta(
                    ["CREATE INDEX idx_orders_custkey ON orders(o_custkey)"]
                ),
                sysknobs=SysKnobsDelta(
                    {"shared_buffers": "2GB"},
                ),
                qknobs=QueryKnobsDelta(
                    {
                        f"S{DEFAULT_TPCH_SEED}-Q1": [
                            "set enable_hashagg = off",
                            "set enable_sort = on",
                        ],
                    }
                ),
            )
        )
        replay_data = replay(
            ReplayTests.workspace,
            writer.tuning_artifacts_path,
        )

        # We do some very simple sanity checks here due to the inherent randomness of executing a workload.
        # We check that there is one data point for the initial config and one for the config change.
        self.assertEqual(len(replay_data), 2)
        # We check that the second step is faster.
        self.assertLess(replay_data[1][0], replay_data[0][0])
        # We check that no queries timed out in either step.
        self.assertEqual(replay_data[0][1], 0)
        self.assertEqual(replay_data[1][1], 0)


if __name__ == "__main__":
    unittest.main()
