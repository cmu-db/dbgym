import unittest

from benchmark.tpch.constants import DEFAULT_TPCH_SEED
from env.integtest_util import IntegtestWorkspace, MockTuningAgent
from env.replay import replay
from env.tuning_agent_artifacts import (
    DBMSConfigDelta,
    IndexesDelta,
    QueryKnobsDelta,
    SysKnobsDelta,
)


class ReplayTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    def test_replay(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())
        agent.delta_to_return = DBMSConfigDelta(
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
        agent.step()
        replay_data = replay(
            IntegtestWorkspace.get_dbgym_cfg(), agent.tuning_agent_artifacts_dpath
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
