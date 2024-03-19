from typing import Any

import psycopg
from gymnasium.spaces import Box

from envs.lsc.lsc import LSC
from envs.spaces.utils import check_subspace
from envs.spaces.holon_space import HolonSpace
from envs.spaces.state.metric import MetricStateSpace
from envs.spaces.state.structure import StructureStateSpace


class LSCStructureStateSpace(StructureStateSpace):
    def __init__(
        self,
        lsc: LSC,
        action_space: HolonSpace,
        seed: int,
    ) -> None:
        spaces = {"lsc": Box(low=-1, high=1.0)}
        super().__init__(action_space, spaces, seed)
        self.lsc = lsc

    def construct_offline(
        self, connection: psycopg.Connection[Any], data: Any, prev_state_container: Any
    ) -> dict[str, Any]:
        state = super().construct_offline(connection, data, prev_state_container)
        state["lsc"] = self.lsc.current_scale()
        assert check_subspace(self, state)
        return state


class LSCMetricStateSpace(MetricStateSpace):
    def __init__(self, lsc: LSC, tables: list[str], seed: int):
        spaces = {"lsc": Box(low=-1, high=1.0)}
        super().__init__(spaces, tables, seed)
        self.lsc = lsc

    def construct_offline(
        self, connection: psycopg.Connection[Any], data: Any, state_container: Any
    ) -> dict[str, Any]:
        state = super().construct_offline(connection, data, state_container)
        state["lsc"] = self.lsc.current_scale()
        assert check_subspace(self, state)
        return state

    def construct_online(self, connection: psycopg.Connection[Any]) -> dict[str, Any]:
        state = super().construct_online(connection)
        state["lsc"] = self.lsc.current_scale()
        return state

    def merge_deltas(self, merge_data: list[dict[str, Any]]) -> dict[str, Any]:
        state = super().merge_deltas(merge_data)
        state["lsc"] = self.lsc.current_scale()
        assert check_subspace(self, state)
        return state
