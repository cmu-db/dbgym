import subprocess
import unittest

from env.integtest_util import (
    ENV_INTEGTESTS_DBGYM_CONFIG_FPATH,
    get_integtest_workspace_path,
)
from env.tuning_agent import TuningAgent
from util.workspace import DBGymConfig


class MockTuningAgent(TuningAgent):
    pass


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        # If you're running the test locally, this check makes runs past the first one much faster.
        if not get_integtest_workspace_path().exists():
            subprocess.run(["./env/set_up_env_integtests.sh"], check=True)

        PostgresConnTests.dbgym_cfg = DBGymConfig(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH)

    def test_test(self) -> None:
        agent = MockTuningAgent(PostgresConnTests.dbgym_cfg)
        agent.step()


if __name__ == "__main__":
    unittest.main()
