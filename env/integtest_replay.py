import unittest

from env.integtest_util import IntegtestWorkspace, MockTuningAgent
from env.replay import replay


class ReplayTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    def test_replay(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())
        replay(IntegtestWorkspace.get_dbgym_cfg(), agent.tuning_agent_artifacts_dpath)


if __name__ == "__main__":
    unittest.main()
