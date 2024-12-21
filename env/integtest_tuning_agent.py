import subprocess
import unittest

from env.integtest_util import (
    ENV_INTEGTESTS_DBGYM_CONFIG_FPATH,
    get_integtest_workspace_path,
)
from env.tuning_agent import DBMSConfig, TuningAgent
from util.workspace import DBGymConfig


class MockTuningAgent(TuningAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_to_return = None

    def _step(self) -> DBMSConfig:
        assert self.config_to_return is not None
        ret = self.config_to_return
        # Setting this ensures you must set self.config_to_return every time.
        self.config_to_return = None
        return ret


class PostgresConnTests(unittest.TestCase):
    dbgym_cfg: DBGymConfig

    @staticmethod
    def setUpClass() -> None:
        # If you're running the test locally, this check makes runs past the first one much faster.
        if not get_integtest_workspace_path().exists():
            subprocess.run(["./env/set_up_env_integtests.sh"], check=True)

        PostgresConnTests.dbgym_cfg = DBGymConfig(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH)

    def test_test(self) -> None:
        agent = MockTuningAgent(PostgresConnTests.dbgym_cfg)
        config_a = DBMSConfig(["a"], {"a": "a"}, {"a": ["a"]})
        agent.config_to_return = config_a
        agent.step()
        self.assertEqual(agent.get_step_delta(0), config_a)


if __name__ == "__main__":
    unittest.main()
