from typing import Any, Optional, Tuple

import psycopg

from tune.protox.env.logger import Logger
from tune.protox.env.space.latent_space.latent_knob_space import LatentKnobSpace
from tune.protox.env.space.primitive_space import QuerySpace
from tune.protox.env.types import (
    QueryMap,
    QuerySpaceAction,
    QuerySpaceContainer,
    QuerySpaceKnobAction,
)


class LatentQuerySpace(LatentKnobSpace, QuerySpace):
    def __init__(
        self, logger: Optional[Logger] = None, *args: Any, **kwargs: Any
    ) -> None:
        # Only manually initialize against QuerySpace.
        QuerySpace.__init__(self, *args, **kwargs)
        self.logger = logger
        self.name = "query"

    def uses_embed(self) -> bool:
        return False

    def generate_state_container(
        self,
        prev_state: Optional[QuerySpaceContainer],
        action: Optional[QuerySpaceAction],
        connection: psycopg.Connection[Any],
        queries: QueryMap,
    ) -> QuerySpaceContainer:
        sc = super().generate_state_container(prev_state, action, connection, queries)
        if action is not None:
            for k, v in action.items():
                assert k in sc
                sc[k] = v
        return sc

    def generate_action_plan(
        self, action: QuerySpaceAction, sc: QuerySpaceContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        return [], []

    def generate_delta_action_plan(
        self, action: QuerySpaceAction, sc: QuerySpaceContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        return [], []

    def extract_query(self, action: QuerySpaceAction) -> QuerySpaceKnobAction:
        ret_knobs = QuerySpaceKnobAction({})
        for k, v in action.items():
            assert k in self.knobs
            ret_knobs[self.knobs[k]] = v
        return ret_knobs

    def replace_query(self, query: QuerySpaceKnobAction) -> QuerySpaceAction:
        return QuerySpaceAction({k.name(): v for k, v in query.items()})
