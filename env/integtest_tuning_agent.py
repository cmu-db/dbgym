import unittest
from pathlib import Path
from typing import Any, Optional

from env.integtest_util import IntegtestWorkspace
from env.tuning_agent import (
    DBMSConfigDelta,
    IndexesDelta,
    QueryKnobsDelta,
    SysKnobsDelta,
    TuningAgent,
    TuningAgentMetadata,
    TuningAgentStepReader,
)
from util.workspace import fully_resolve_path


class MockTuningAgent(TuningAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config_to_return: Optional[DBMSConfigDelta] = None

    @staticmethod
    def get_mock_fully_resolved_path() -> Path:
        return fully_resolve_path(
            IntegtestWorkspace.get_dbgym_cfg(), IntegtestWorkspace.get_workspace_path()
        )

    def _get_metadata(self) -> TuningAgentMetadata:
        # We just need these to be some fully resolved path, so I just picked the workspace path.
        return TuningAgentMetadata(
            workload_path=MockTuningAgent.get_mock_fully_resolved_path(),
            pristine_dbdata_snapshot_path=MockTuningAgent.get_mock_fully_resolved_path(),
            dbdata_parent_path=MockTuningAgent.get_mock_fully_resolved_path(),
            pgbin_path=MockTuningAgent.get_mock_fully_resolved_path(),
        )

    def _step(self) -> DBMSConfigDelta:
        assert self.config_to_return is not None
        ret = self.config_to_return
        # Setting this ensures you must set self.config_to_return every time.
        self.config_to_return = None
        return ret


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    @staticmethod
    def make_config(letter: str) -> DBMSConfigDelta:
        return DBMSConfigDelta(
            indexes=IndexesDelta([letter]),
            sysknobs=SysKnobsDelta({letter: letter}),
            qknobs=QueryKnobsDelta({letter: [letter]}),
        )

    def test_get_step_delta(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        reader = TuningAgentStepReader(agent.tuning_agent_artifacts_dpath)

        self.assertEqual(reader.get_step_delta(1), PostgresConnTests.make_config("b"))
        self.assertEqual(reader.get_step_delta(0), PostgresConnTests.make_config("a"))
        self.assertEqual(reader.get_step_delta(1), PostgresConnTests.make_config("b"))
        self.assertEqual(reader.get_step_delta(2), PostgresConnTests.make_config("c"))

    def test_get_all_deltas(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        reader = TuningAgentStepReader(agent.tuning_agent_artifacts_dpath)

        self.assertEqual(
            reader.get_all_deltas(),
            [
                PostgresConnTests.make_config("a"),
                PostgresConnTests.make_config("b"),
                PostgresConnTests.make_config("c"),
            ],
        )

    def test_get_metadata(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())
        reader = TuningAgentStepReader(agent.tuning_agent_artifacts_dpath)
        metadata = reader.get_metadata()
        self.assertEqual(
            metadata.workload_path, MockTuningAgent.get_mock_fully_resolved_path()
        )
        self.assertEqual(
            metadata.pristine_dbdata_snapshot_path,
            MockTuningAgent.get_mock_fully_resolved_path(),
        )
        self.assertEqual(
            metadata.dbdata_parent_path, MockTuningAgent.get_mock_fully_resolved_path()
        )
        self.assertEqual(
            metadata.pgbin_path, MockTuningAgent.get_mock_fully_resolved_path()
        )


if __name__ == "__main__":
    unittest.main()
