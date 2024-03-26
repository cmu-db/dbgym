from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from gymnasium import spaces
from psycopg import Connection

from misc.utils import DBGymConfig


class StateSpace(ABC, spaces.Space[Any]):
    @abstractmethod
    def require_metrics(self) -> bool:
        pass

    @abstractmethod
    def check_benchbase(self, dbgym_cfg: DBGymConfig, results: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def construct_offline(
        self, connection: Connection[Any], data: Any, state_container: Any
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def construct_online(self, connection: Connection[Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def state_delta(
        self, initial: dict[str, Any], final: dict[str, Any]
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def merge_deltas(self, merge_data: list[dict[str, Any]]) -> dict[str, Any]:
        pass
