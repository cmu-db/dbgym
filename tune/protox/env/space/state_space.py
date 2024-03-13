import torch
import json
import numpy as np
from pathlib import Path
from gymnasium import spaces
from gymnasium.spaces import Box
from psycopg.rows import dict_row

from tune.protox.env.space.utils import fetch_server_knobs, check_subspace, METRICS_SPECIFICATION


# A metrics-based state returns the physical metrics (i.e., consequences) of running
# a particular workload in a given configuration. This serves to represent the
# assumption that we should be indifferent/invariant to <workload, configuration>
# pairs that yield the *same* physical metrics.
#
# In the RL state-action-reward-next_state sense:
# The benchmark is executed in the baseline configuration to determine the physical metrics
# as a consequence of the baseline configuration. That is the "previous state".
#
# You then pick an action that produces a new configuration. That configuration is then applied
# to the database. This is the action.
#
# You then run the benchmark again. This yields some "target" metric and also physical database
# metrics. "target" metric is used to determine the reward from the transition. The physical
# database metrics form the "next_state".
#
# In this way, the physical database metrics serves as proxy for the actual configuration at
# a given moment in time. This is arguably a little bit twisted?? Since we are using some
# metrics that are also indirectly a proxy for the actual runtime/tps. But we are banking
# on the metrics containing the relevant data to allow better action selection...
class MetricStateSpace(spaces.Dict):
    tables: list[str] = None
    internal_spaces: dict[str, spaces.Space] = {}

    @staticmethod
    def construct_key(key, metric, per_tbl, tbl):
        if per_tbl:
            return f"{key}_{metric}_{tbl}"
        return f"{key}_{metric}"

    def metrics(self):
        return True

    def __init__(self, tables: list[str], postgres_db, seed):
        self.tables = tables
        self.postgres_db = postgres_db
        self.internal_spaces = {}
        for key, spec in METRICS_SPECIFICATION.items():
            for key_metric in spec["valid_keys"]:
                if spec["per_table"]:
                    for tbl in tables:
                        tbl_metric = MetricStateSpace.construct_key(key, key_metric, True, tbl)
                        assert tbl_metric not in self.internal_spaces
                        self.internal_spaces[tbl_metric] = Box(low=-np.inf, high=np.inf)
                else:
                    metric = MetricStateSpace.construct_key(key, key_metric, False, None)
                    assert metric not in self.internal_spaces
                    self.internal_spaces[metric] = Box(low=-np.inf, high=np.inf)
        self.internal_spaces["lsc"] = Box(low=-1, high=1.)
        super().__init__(self.internal_spaces, seed)

    def check_benchbase(self, **kwargs):
        assert "results" in kwargs and kwargs["results"] is not None
        assert Path(kwargs["results"]).exists()
        metric_files = [f for f in Path(kwargs["results"]).rglob("*metrics.json")]
        if len(metric_files) != 2:
            return False

        initial = metric_files[0] if "initial" in str(metric_files[0]) else metric_files[1]
        final = metric_files[1] if initial == metric_files[0] else metric_files[0]

        try:
            with open(initial) as f:
                initial_metrics = json.load(f)

            with open(final) as f:
                final_metrics = json.load(f)
        except Exception as e:
            return False

        for key, spec in METRICS_SPECIFICATION.items():
            assert key in initial_metrics
            if key not in initial_metrics or key not in final_metrics:
                # Missing key.
                return False

            initial_data = initial_metrics[key]
            final_data = final_metrics[key]
            if spec["filter_db"]:
                initial_data = [d for d in initial_data if d["datname"] == self.postgres_db]
                final_data = [d for d in final_data if d["datname"] == self.postgres_db]
            elif spec["per_table"]:
                initial_data = sorted([d for d in initial_data if d["relname"] in self.tables], key=lambda x: x["relname"])
                final_data = sorted([d for d in final_data if d["relname"] in self.tables], key=lambda x: x["relname"])

            if len(initial_data) == 0 or len(final_data) == 0:
                return False

            for pre, post in zip(initial_data, final_data):
                for metric in spec["valid_keys"]:
                    if metric not in pre or metric not in post:
                        return False
        return True

    def construct_offline(self, **kwargs):
        # TODO: we probably need a way to fetch the current memory utilization
        # from the system so we can account for states of different memory util.

        assert "results" in kwargs and kwargs["results"] is not None
        assert Path(kwargs["results"]).exists()

        # This function computes the metrics state that is used to represent
        # consequence of executing in the current environment.
        metric_files = [f for f in Path(kwargs["results"]).rglob("*metrics.json")]
        if len(metric_files) == 1:
            with open(metric_files[0], "r") as f:
                metrics = json.load(f)
                assert "flattened" in metrics
                metrics.pop("flattened")

            def npify(d):
                data = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        data[k] = npify(v)
                    else:
                        data[k] = np.array([v], dtype=np.float32)
                return data
            return npify(metrics)

        assert len(metric_files) == 2

        initial = metric_files[0] if "initial" in str(metric_files[0]) else metric_files[1]
        final = metric_files[1] if initial == metric_files[0] else metric_files[0]

        with open(initial) as f:
            initial_metrics = json.load(f)

        with open(final) as f:
            final_metrics = json.load(f)

        return self.construct_metric_delta(initial_metrics, final_metrics)

    def construct_metric_delta(self, initial_metrics, final_metrics):
        metrics = {"lsc": np.array([-1], dtype=np.float32)}
        for key, spec in METRICS_SPECIFICATION.items():
            assert key in initial_metrics
            initial_data = initial_metrics[key]
            final_data = final_metrics[key]
            if spec["filter_db"]:
                initial_data = [d for d in initial_data if d["datname"] == self.postgres_db]
                final_data = [d for d in final_data if d["datname"] == self.postgres_db]
            elif spec["per_table"]:
                initial_data = sorted([d for d in initial_data if d["relname"] in self.tables], key=lambda x: x["relname"])
                final_data = sorted([d for d in final_data if d["relname"] in self.tables], key=lambda x: x["relname"])

            for pre, post in zip(initial_data, final_data):
                for metric in spec["valid_keys"]:
                    if pre[metric] is None or post[metric] is None:
                        diff = 0
                    else:
                        diff = float(post[metric]) - float(pre[metric])

                    metric_key = MetricStateSpace.construct_key(key, metric, spec["per_table"], pre["relname"] if spec["per_table"] else None)
                    metrics[metric_key] = np.array([diff], dtype=np.float32)

        assert check_subspace(self, metrics)
        return metrics

    def construct_online(self, **kwargs):
        assert "connection" in kwargs and kwargs["connection"] is not None
        connection = kwargs["connection"]

        metric_data = {"lsc": np.array([-1], dtype=np.float32)}
        with connection.cursor(row_factory=dict_row) as cursor:
            for key in METRICS_SPECIFICATION.keys():
                records = cursor.execute(f"SELECT * FROM {key}")
                metric_data[key] = [r for r in records]

        return metric_data

    def merge_data(self, data):
        comb_data = {}
        for datum in data:
            for key, value in datum.items():
                if key not in comb_data:
                    comb_data[key] = value
                elif key != "lsc":
                    # Assume that each datum is "localized".
                    comb_data[key] += value
        assert check_subspace(self, comb_data)
        return comb_data


class StructureStateSpace(spaces.Dict):
    action_space = None
    internal_spaces = {}
    normalize = False
    div = True

    def __init__(self, action_space, normalize, seed, div=True):
        self.action_space = action_space
        self.normalize = normalize
        self.div = div

        if normalize:
            self.internal_spaces["knobs"] = Box(low=-np.inf, high=np.inf, shape=[action_space.get_knob_space().final_dim])
        else:
            self.internal_spaces["knobs"] = self.action_space.get_knob_space()

        self.internal_spaces["index"] = Box(low=-np.inf, high=np.inf, shape=[action_space.get_index_space().get_critic_dim()])
        self.internal_spaces["lsc"] = Box(low=-1, high=1.)
        super().__init__(self.internal_spaces, seed)

    def metrics(self):
        return False

    def check_benchbase(self, **kwargs):
        # We don't use benchbase metrics anyways.
        return True

    def construct_offline(self, **kwargs):
        connection = kwargs["connection"]
        action = kwargs["action"]

        knob_state = fetch_server_knobs(connection, tables=self.action_space.get_knob_space().tables, knobs=self.action_space.get_knob_space().knobs)
        # Assimilate the query level knobs back.
        ql_knobs = self.action_space.get_query_level_knobs(action) if action is not None else {}
        knob_state.update({k: v for k, (_, v) in ql_knobs.items()})

        if self.normalize:
            # Normalize.
            knob_state = np.array(self.action_space.get_knob_space()._env_to_embedding(knob_state), dtype=np.float32)[0]
        assert check_subspace(self.internal_spaces["knobs"], knob_state)

        # Handle indexes.
        current_bias = 0 if self.action_space.get_index_space().lsc is None else self.action_space.get_index_space().lsc.current_bias()
        indexes = self.action_space.get_index_space().get_state_with_bias(None)
        if action is not None:
            indexes.append((action[1], current_bias))

        if len(indexes) > 0:
            vae = self.action_space.get_index_space().vae
            with torch.no_grad():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                env_acts = np.array([v[0] for v in indexes]).reshape(len(indexes), -1)
                biases = np.array([v[1] for v in indexes]).reshape(len(indexes), 1)

                aux_index_type = None
                aux_include = None
                if self.action_space.get_index_space().index_space_aux_type_dim > 0:
                    # Boink the index type.
                    index_val = torch.tensor(env_acts[:, 0]).view(env_acts.shape[0], -1)
                    index_type = torch.zeros(index_val.shape[0], 2, dtype=torch.int64)
                    aux_index_type = index_type.scatter_(1, index_val, 1).type(torch.float32)
                    env_acts = env_acts[:, 1:]

                if self.action_space.get_index_space().index_space_aux_include > 0:
                    aux_include = torch.tensor(env_acts[:, -self.action_space.get_index_space().index_space_aux_include:]).float()
                    env_acts = env_acts[:, :-self.action_space.get_index_space().index_space_aux_include]

                nets = vae.get_collate()(env_acts).to(device=device).float()
                latents, error = vae.latents(nets)
                latents = latents.cpu()
                assert not error

                latents = latents + biases
                if aux_index_type is not None:
                    latents = torch.concat([aux_index_type, latents], dim=1)
                if aux_include is not None:
                    latents = torch.concat([aux_include, latents], dim=1)

                if self.div:
                    index_state = (latents.sum(dim=0) / len(indexes)).numpy().flatten()
                else:
                    index_state = (latents.sum(dim=0)).numpy().flatten()
                index_state = index_state.astype(np.float32)
        else:
            index_state = np.zeros(self.action_space.get_index_space().get_critic_dim(), dtype=np.float32)

        state = {
            "knobs": knob_state,
            "index": index_state,
            "lsc": current_bias,
        }
        return state
