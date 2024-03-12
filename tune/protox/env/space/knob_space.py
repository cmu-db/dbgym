import torch
from pprint import pformat
import xml.etree.ElementTree as ET

import numpy as np
import gymnasium as gym
from tune.protox.env.space.knob import KnobClass, SettingType, is_knob_enum, resolve_enum_value, Knob, CategoricalKnob, _create_knob
from gymnasium import spaces
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatdim
from tune.protox.env.space.utils import check_subspace, fetch_server_knobs, overwrite_benchbase_hintset


class KnobSpace(spaces.Dict):
    tables: list[str] = None
    knobs: dict[str, Knob] = None
    state_container = None

    # Where categorical starts.
    categorical_start = None
    cat_dims = None
    # Total dimension.
    final_dim = None

    def save_state(self):
        return {
            "state_container": self.state_container,
        }

    def load_state(self, d):
        self.state_container = d["state_container"]

    def get_state(self, _):
        return self.state_container

    def _process_network_output(self, subproto):
        cont_dim = self.categorical_start

        cont, *cats = subproto.split([cont_dim] + self.cat_dims, dim=-1)

        cont = torch.tanh(cont)
        # Softmax the categorical outputs.
        cats = [cat.softmax(dim=-1) for cat in cats]
        output = torch.concat([cont] + cats, dim=-1)
        return output

    def _perturb_noise(self, proto, noise):
        cont_dim = self.categorical_start
        cat_dim = self.final_dim - cont_dim

        cont, cats = proto.split([cont_dim, cat_dim], dim=-1)
        cont_noise, _ = noise.split([cont_dim, cat_dim], dim=-1)
        cont = torch.clamp(cont + cont_noise, -1., 1.)
        output = torch.concat([cont, cats], dim=-1)
        return output

    def get_query_level_knobs(self, action):
        # assert self.contains(action)
        ret = {}
        for key, knob in self.knobs.items():
            if knob.knob_class == KnobClass.QUERY:
                ret[key] = (knob, action[key])
        return ret

    def get_latent_dim(self):
        return 0

    def get_critic_dim(self):
        return flatdim(self)

    @property
    def latent(self):
        return False

    def _decode(self, act):
        return act

    def __init__(self,
            tables,
            knobs,
            table_level_knobs,
            per_query_knobs,
            per_query_knobs_gen,
            quantize,
            quantize_factor,
            seed,
            per_query_parallel={},
            per_query_scans={},
            query_names=[]):
        self.knobs = {}
        self.tables = tables
        spaces = []
        for k, md in knobs.items():
            knob = _create_knob(None, None, k, md, quantize, quantize_factor, seed)
            self.knobs[knob.name()] = knob
            spaces.append((knob.name(), knob))

        for t, kv in table_level_knobs.items():
            for k, md in kv.items():
                knob = _create_knob(t, None, k, md, quantize, quantize_factor, seed)
                self.knobs[knob.name()] = knob
                spaces.append((knob.name(), knob))

        for q, kv in per_query_knobs.items():
            for k, md in kv.items():
                knob = _create_knob(None, q, k, md, quantize, quantize_factor, seed)
                self.knobs[knob.name()] = knob
                spaces.append((knob.name(), knob))

        for qname in query_names:
            for q, kv in per_query_knobs_gen.items():
                knob = _create_knob(None, qname, q, kv, quantize, quantize_factor, seed)
                self.knobs[knob.name()] = knob
                spaces.append((knob.name(), knob))

        for q, kv in per_query_scans.items():
            for _, aliases in kv.items():
                for v in aliases:
                    md = {
                        "type": "scanmethod_enum",
                        "min": 0,
                        "max": 1,
                        "quantize": 0,
                        "log_scale": 0,
                        "unit": 0,
                    }

                    knob = _create_knob(
                        table_name=None,
                        query_name=q,
                        knob_name=v + "_scanmethod",
                        metadata=md,
                        do_quantize=False,
                        default_quantize_factor=quantize_factor,
                        seed=seed)
                    self.knobs[knob.name()] = knob
                    spaces.append((knob.name(), knob))

        cat_spaces = []
        self.cat_dims = []
        for q, kv in per_query_parallel.items():
            values = []
            for _, aliases in kv.items():
                values.extend(aliases)

            if len(values) < 2:
                continue

            md = {
                "type": "query_table_enum",
                "values": values,
                "default": 0,
            }
            knob = CategoricalKnob(
                table_name=None,
                query_name=q,
                knob_name=q + "_parallel_rel",
                metadata=md,
                seed=seed)
            self.knobs[knob.name()] = knob

            cat_spaces.append((knob.name(), knob))
            self.cat_dims.append(knob.num_elems)

        # Figure out where the categorical inputs begin.
        self.categorical_start = gym.spaces.utils.flatdim(Dict(spaces))
        spaces.extend(cat_spaces)
        super().__init__(spaces, seed=seed)
        self.final_dim = gym.spaces.utils.flatdim(self)

    def _env_to_embedding(self, env_act):
        if not isinstance(env_act, list):
            env_act = [env_act]

        embeds = []
        for act in env_act:
            assert check_subspace(self, act)

            flattened = gym.spaces.utils.flatten(self, {
                k: self.knobs[k].project_env_to_embedding(v)
                for k, v in act.items()
            })

            embeds.append(flattened)
        return embeds

    def _nearest_env_action(self, act):
        assert act.shape[0] == self.final_dim

        env_act = {}
        cont_env_act = gym.spaces.utils.unflatten(self, act)
        for key, knob in self.knobs.items():
            assert isinstance(knob, Knob) or isinstance(knob, CategoricalKnob)
            env_act[key] = knob.project_embedding_to_env(cont_env_act[key])
            assert knob.contains(env_act[key]), print(key, env_act[key], knob)
        # assert self.contains(env_act)
        return env_act

    def _random_embed_action(self, num_action):
        cont_dim = self.categorical_start
        cat_dim = self.final_dim - cont_dim

        # Sample according to strategy within the latent dimension.
        cont_action = np.random.uniform(low=-1., high=1., size=(num_action, cont_dim))
        cat_action = np.random.uniform(low=0., high=1., size=(num_action, cat_dim))
        action = np.concatenate([cont_action, cat_action], axis=1)
        return action

    def reset(self, **kwargs):
        # Reset the information from postgres.
        assert "connection" in kwargs
        connection = kwargs["connection"]
        self.state_container = fetch_server_knobs(connection, self.tables, self.knobs, workload=kwargs["workload"])
        if "config" in kwargs and kwargs["config"] is not None:
            for key, knob in self.knobs.items():
                if knob.knob_class == KnobClass.QUERY:
                    self.state_container[knob.name()] = kwargs["config"][0][knob.name()]

    def advance(self, action, **kwargs):
        # Advance the state container internal state representation.
        # assert self.contains(action)
        assert "connection" in kwargs
        connection = kwargs["connection"]
        workload = kwargs["workload"]
        # Fetch main.
        self.state_container = fetch_server_knobs(connection, self.tables, self.knobs, workload=workload)
        # Set the per-query accordingly.
        for key, knob in self.knobs.items():
            if knob.knob_class == KnobClass.QUERY:
                self.state_container[knob.name()] = action[knob.name()]

    def search_embedding_neighborhood(self, act, neighbor_parameters):
        num_neighbors = neighbor_parameters["knob_num_nearest"]
        span = neighbor_parameters["knob_span"]
        act = act.numpy()

        # Dimensions.
        cont_dim = self.categorical_start
        cat_dim = self.final_dim - cont_dim

        # Get the baseline valid action.
        env_action = self._nearest_env_action(act)
        cat_start = self.categorical_start

        valid_env_actions = [env_action]
        for _ in range(num_neighbors):
            adjust_mask = self.np_random.integers(-span, span + 1, (cont_dim,))
            if np.sum(adjust_mask) == 0:
                continue

            new_action = env_action.copy()
            cont_it = 0
            for knobname, knob in self.spaces.items():
                # We assume that this order is the way in which the action is laid out.
                if isinstance(knob, CategoricalKnob):
                    new_value = knob._sample_weights(act[cat_start:cat_start + knob.num_elems])
                    new_action[knobname] = new_value
                    cat_start += knob.num_elems

                else:
                    # Iterate through every knob and adjust based on the sampled mask.
                    if adjust_mask[cont_it] != 0:
                        new_value = knob.adjust_quantize_bin(new_action[knobname], adjust_mask[cont_it])
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

    def generate_plan(self, action, **kwargs):
        assert check_subspace(self, action)
        force = kwargs.get("force", False)
        if "original_benchbase_config_path" in kwargs:
            original_benchbase_config_path = kwargs["original_benchbase_config_path"]
            benchbase_config_path = kwargs["benchbase_config_path"]

            conf_etree = ET.parse(original_benchbase_config_path)
            root = conf_etree.getroot()
        else:
            original_benchbase_config_path = None
            benchbase_config_path = None

            conf_etree = None
            root = None

        config_changes = []
        sql_commands = []
        require_cleanup = False

        for act, val in action.items():
            assert act in self.knobs, print(self.knobs, act)
            if self.knobs[act].knob_class == KnobClass.TABLE:
                if (act not in self.state_container or self.state_container[act] != val) or force:
                    # Need to perform a VACUUM ANALYZE.
                    require_cleanup = True

                    tbl = self.knobs[act].table_name
                    knob = self.knobs[act].knob_name
                    sql_commands.append(f"ALTER TABLE {tbl} SET ({knob} = {val})")
                    # Rewrite immediately.
                    sql_commands.append(f"VACUUM FULL {tbl}")

            elif self.knobs[act].knob_class == KnobClass.QUERY:
                if root is not None:
                    # Handle the per-query adjustment to the benchbase config path.
                    knob = self.knobs[act]
                    target = knob.resolve_per_query_knob(val, all_knobs=action)
                    overwrite_benchbase_hintset(root, knob.query_name, target)

            elif self.knobs[act].knob_type == SettingType.BOOLEAN:
                # Boolean knob.
                assert self.knobs[act].knob_class == KnobClass.KNOB
                flag = "on" if val == 1 else "off"
                config_changes.append(f"{act} = {flag}")

            elif is_knob_enum(self.knobs[act]):
                out_val = resolve_enum_value(self.knobs[act], val, all_knobs=action)
                config_changes.append(f"{act} = {out_val}")

            else:
                # Integer or float knob.
                assert self.knobs[act].knob_class == KnobClass.KNOB
                kt = self.knobs[act].knob_type
                param = "{act} = {val:.2f}" if kt == SettingType.FLOAT else "{act} = {val:d}"
                assert kt == SettingType.FLOAT or kt == SettingType.INTEGER or kt == SettingType.BYTES or kt == SettingType.INTEGER_TIME
                config_changes.append(param.format(act=act, val=val))

        if require_cleanup:
            for tbl in self.tables:
                sql_commands.append(f"VACUUM ANALYZE {tbl}")
            sql_commands.append(f"CHECKPOINT")

        # Write to the actual destination config path.
        if conf_etree is not None:
            conf_etree.write(benchbase_config_path)
        return config_changes, sql_commands

    def generate_action_plan(self, action, **kwargs):
        return self.generate_plan(action, **kwargs)

    def generate_delta_action_plan(self, action, **kwargs):
        return self.generate_plan(action, **kwargs)
