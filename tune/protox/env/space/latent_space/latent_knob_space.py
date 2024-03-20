import copy
from pprint import pformat
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from psycopg import Connection

from tune.protox.env.logger import Logger, time_record
from tune.protox.env.space.primitive import KnobClass, SettingType, is_knob_enum
from tune.protox.env.space.primitive.knob import resolve_enum_value
from tune.protox.env.space.primitive.latent_knob import (
    LatentCategoricalKnob,
    LatentKnob,
)
from tune.protox.env.space.primitive_space import KnobSpace
from tune.protox.env.space.utils import check_subspace, fetch_server_knobs
from tune.protox.env.types import (
    DEFAULT_NEIGHBOR_PARAMETERS,
    KnobSpaceAction,
    KnobSpaceContainer,
    NeighborParameters,
    ProtoAction,
    QueryMap,
)


class LatentKnobSpace(KnobSpace):
    def __init__(
        self, logger: Optional[Logger] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.final_dim = gym.spaces.utils.flatdim(self)
        self.categorical_start = self.final_dim
        self.logger = logger
        self.cat_dims: list[int] = []
        self.name = "knobs"

    def latent_dim(self) -> int:
        return gym.spaces.utils.flatdim(self)

    def critic_dim(self) -> int:
        return self.latent_dim()

    def transform_noise(
        self, subproto: ProtoAction, noise: Optional[torch.Tensor] = None
    ) -> ProtoAction:
        cont_dim = self.categorical_start
        cat_dim = self.final_dim - cont_dim

        cont, *cats = subproto.split([cont_dim] + self.cat_dims, dim=-1)  # type: ignore
        cont = torch.tanh(cont)
        if noise is not None:
            cont_noise, _ = noise.split([cont_dim, cat_dim], dim=-1)  # type: ignore
            # TODO(wz2): We only apply the noise to the continuous dimensions.
            # In theory, for categorical, we would noise the logits and use something like boltzmann.
            cont = torch.clamp(cont + cont_noise, -1.0, 1.0)

        if len(cats) > 0:
            cats = torch.concat([cat.softmax(dim=-1) for cat in cats], dim=-1)
            output = torch.concat([cont, cats], dim=-1)
        else:
            output = cont
        return ProtoAction(output)

    def from_latent(self, a: ProtoAction) -> ProtoAction:
        return a

    def nearest_env_action(self, output: ProtoAction) -> KnobSpaceAction:
        cont_env_act = gym.spaces.utils.unflatten(self, output.numpy())
        env_act = KnobSpaceAction({})
        for key, knob in self.knobs.items():
            assert isinstance(knob, LatentKnob) or isinstance(
                knob, LatentCategoricalKnob
            )
            env_act[key] = knob.from_latent(cont_env_act[key])
            assert knob.contains(env_act[key]), print(key, env_act[key], knob)

        assert self.contains(env_act)
        return env_act

    @time_record("to_latent")
    def to_latent(self, env_act: list[KnobSpaceAction]) -> ProtoAction:
        assert isinstance(env_act, list)

        embeds = []
        for act in env_act:
            assert check_subspace(self, act)

            kv_dict: dict[str, Any] = {}
            for k, v in act.items():
                knob = self.knobs[k]
                assert isinstance(knob, LatentKnob) or isinstance(
                    knob, LatentCategoricalKnob
                )
                kv_dict[k] = knob.to_latent(v)

            embeds.append(gym.spaces.utils.flatten(self, kv_dict))
        return ProtoAction(torch.as_tensor(np.array(embeds)).float())

    def sample_latent(self, mask: Optional[Any] = None) -> ProtoAction:
        cont_dim = self.categorical_start
        cat_dim = self.final_dim - cont_dim

        # Sample according to strategy within the latent dimension.
        cont_action = (torch.rand(size=(cont_dim,)) * 2 - 1).view(1, -1)
        cat_action = torch.rand(size=(cat_dim,)).view(1, -1)
        return ProtoAction(torch.concat([cont_action, cat_action], dim=1).float())

    def pad_center_latent(
        self, subproto: ProtoAction, lscs: torch.Tensor
    ) -> ProtoAction:
        return subproto

    @time_record("neighborhood")
    def neighborhood(
        self,
        raw_action: ProtoAction,
        neighbor_parameters: NeighborParameters = DEFAULT_NEIGHBOR_PARAMETERS,
    ) -> list[KnobSpaceAction]:
        num_neighbors = neighbor_parameters["knob_num_nearest"]
        span = neighbor_parameters["knob_span"]
        env_action = self.nearest_env_action(raw_action)
        cat_start = self.categorical_start

        valid_env_actions = [env_action]
        for _ in range(num_neighbors):
            adjust_mask = self.np_random.integers(-span, span + 1, (self.final_dim,))
            if np.sum(adjust_mask) == 0:
                continue

            new_action = KnobSpaceAction(copy.deepcopy(env_action))
            cont_it = 0
            for knobname, knob in self.spaces.items():
                # Iterate through every knob and adjust based on the sampled mask.
                if isinstance(knob, LatentCategoricalKnob):
                    new_value = knob.sample_weights(
                        weights=raw_action[
                            cat_start : cat_start + knob.num_elems
                        ].tolist()
                    )
                    new_action[knobname] = new_value
                    cat_start += knob.num_elems
                else:
                    assert isinstance(knob, LatentKnob)
                    if adjust_mask[cont_it] != 0:
                        new_value = knob.shift_offset(
                            new_action[knobname], adjust_mask[cont_it]
                        )
                        if new_value is not None:
                            # The adjustment has produced a new quantized value.
                            new_action[knobname] = new_value
                    cont_it += 1

            # Check that we aren't adding superfluous actions.
            # assert self.contains(new_action)
            valid_env_actions.append(new_action)

        queued_actions = set()
        real_actions = []
        for new_action in valid_env_actions:
            sig = pformat(new_action)
            if sig not in queued_actions:
                real_actions.append(new_action)
                queued_actions.add(sig)

        return real_actions

    def generate_state_container(
        self,
        prev_state_container: Optional[KnobSpaceContainer],
        action: Optional[KnobSpaceAction],
        connection: Connection[Any],
        queries: QueryMap,
    ) -> KnobSpaceContainer:
        return fetch_server_knobs(connection, self.tables, self.knobs, queries=queries)

    def generate_action_plan(
        self, action: KnobSpaceAction, sc: KnobSpaceContainer, **kwargs: Any
    ) -> Tuple[list[str], list[str]]:
        config_changes = []
        sql_commands = []
        require_cleanup = False

        for act, val in action.items():
            assert act in self.knobs, print(self.knobs, act)
            assert self.knobs[act].knob_class != KnobClass.QUERY
            if self.knobs[act].knob_class == KnobClass.TABLE:
                if act not in sc or sc[act] != val:
                    # Need to perform a VACUUM ANALYZE.
                    require_cleanup = True

                    tbl = self.knobs[act].table_name
                    knob = self.knobs[act].knob_name
                    sql_commands.append(f"ALTER TABLE {tbl} SET ({knob} = {val})")
                    # Rewrite immediately.
                    sql_commands.append(f"VACUUM FULL {tbl}")

            elif self.knobs[act].knob_type == SettingType.BOOLEAN:
                # Boolean knob.
                assert self.knobs[act].knob_class == KnobClass.KNOB
                flag = "on" if val == 1 else "off"
                config_changes.append(f"{act} = {flag}")

            elif is_knob_enum(self.knobs[act].knob_type):
                out_val = resolve_enum_value(self.knobs[act], val, all_knobs=action)
                config_changes.append(f"{act} = {out_val}")

            else:
                # Integer or float knob.
                assert self.knobs[act].knob_class == KnobClass.KNOB
                kt = self.knobs[act].knob_type
                param = (
                    "{act} = {val:.2f}"
                    if kt == SettingType.FLOAT
                    else "{act} = {val:.0f}"
                )
                assert (
                    kt == SettingType.FLOAT
                    or kt == SettingType.INTEGER
                    or kt == SettingType.BYTES
                    or kt == SettingType.INTEGER_TIME
                )
                config_changes.append(param.format(act=act, val=val))

        if require_cleanup:
            for tbl in self.tables:
                sql_commands.append(f"VACUUM ANALYZE {tbl}")
            sql_commands.append(f"CHECKPOINT")
        return config_changes, sql_commands

    def generate_delta_action_plan(
        self, action: KnobSpaceAction, sc: KnobSpaceContainer, **kwargs: Any
    ) -> Tuple[list[str], list[str]]:
        return self.generate_action_plan(action, sc, **kwargs)
