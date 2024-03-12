import json
import yaml
import shutil
from gymnasium.spaces import Space
from pathlib import Path
import logging
import torch

from tune.protox.env.space.index_space import IndexSpace
from tune.protox.env.space.state_space import MetricStateSpace, StructureStateSpace
from tune.protox.env.space.knob_space import KnobSpace
from tune.protox.env.space.action_space import ActionSpace
from tune.protox.env.workload import Workload
from tune.protox.env.lsc import LSC
from tune.protox.utils.logger import Logger
from misc.utils import open_and_save


def gen_scale_output(mean_output_act, output_scale):
    def f(x):
        if mean_output_act == "sigmoid":
            return torch.nn.Sigmoid()(x) * output_scale
        else:
            assert False, print(mean_output_act)
    return f


class Spec(object):
    # Path specifications.
    original_benchbase_config_path: str = None

    # Connection string to postgres.
    connection: str = None

    # Action and observation spaces.
    action_space: ActionSpace = None
    observation_space: Space = None

    workload: Workload = None

    def __build_observation_space(self, seed):
        if self.metric_state == "metric":
            return MetricStateSpace(self.tables, seed)
        elif self.metric_state == "structure":
            return StructureStateSpace(self.action_space, False, seed)
        elif self.metric_state == "structure_normalize":
            return StructureStateSpace(self.action_space, True, seed)
        elif self.metric_state == "structure_normalize_nodiv":
            return StructureStateSpace(self.action_space, True, seed, div=False)
        else:
            assert False

    def __init__(self, dbgym_cfg, agent_type, seed, embedding_path, protox_config_path, benchbase_config_path, benchmark_config_path, horizon, workload_timeout, logger=None):
        with open_and_save(dbgym_cfg, protox_config_path, "r") as f:
            print(f'protox_config_path={protox_config_path}')
            protox_config = yaml.safe_load(f)["protox"]
        for k, v in protox_config.items():
            setattr(self, k, v)

        # Acquire the latent index dimension.
        use_vae = self.index_vae_metadata["index_vae"]

        index_latent_dim = 0
        index_output_scale = 1.
        index_output_func = torch.nn.Tanh()
        vae_config = {}
        if use_vae:
            self.index_vae_metadata["embedding_pth_path"] = embedding_path / "embedding.pth"
            self.index_vae_metadata["embedding_cfg_path"] = embedding_path / "config"
            with open_and_save(dbgym_cfg, self.index_vae_metadata["embedding_cfg_path"], "r") as f:
                vae_config = config = json.load(f)
                index_latent_dim = config["latent_dim"]
                index_output_func = gen_scale_output(config.get("mean_output_act", "sigmoid"), config["output_scale"])
                index_output_scale = config["output_scale"]

        self.postgres_data_folder = self.postgres_data
        self.postgres_data = str(Path(self.postgres_path) / self.postgres_data)
        self.postgres_port = int(self.postgres_port)
        self.connection = "host={host} port={port} dbname={dbname} user={user} password={password}".format(
            host=self.postgres_host,
            port=self.postgres_port,
            dbname=self.postgres_db,
            user=self.postgres_user,
            password=self.postgres_password,
        )
        logging.debug("%s", self.connection)

        self.benchbase_config_path = benchbase_config_path
        self.original_benchbase_config_path = self.benchbase_config_path
        new_benchbase_config_stem = self.benchbase_config_path.stem()
        new_benchbase_config_stem += f"_{self.postgres_port}"
        self.benchbase_config_path = self.benchbase_config_path.with_stem(new_benchbase_config_stem)
        shutil.copy(self.original_benchbase_config_path, self.benchbase_config_path)

        with open_and_save(dbgym_cfg, benchmark_config_path, "r") as f:
            benchmark_config = yaml.safe_load(f)["protox"]
        for k, v in benchmark_config.items():
            setattr(self, k, v)

        # Construct logger.
        if logger is None:
            self.logger = Logger(self)
        else:
            self.logger = logger

        self.workload = Workload(
            tables=self.tables,
            attributes=self.attributes,
            query_spec=self.query_spec,
            pid=None,
            workload_eval_mode=self.workload_eval_mode,
            workload_eval_inverse=self.workload_eval_inverse if hasattr(self, "workload_eval_inverse") else False,
            workload_timeout=workload_timeout,
            workload_timeout_penalty=self.workload_timeout_penalty if hasattr(self, "workload_timeout_penalty") else 1.,
            logger=self.logger)

        # Get column usage.
        modified_attrs = self.workload.process_column_usage()
        tbl_include_subsets = self.workload.tbl_include_subsets

        per_query_scans = {}
        if hasattr(self, "per_query_scan_method") and self.per_query_scan_method:
            per_query_scans = self.workload.query_aliases

        per_query_parallel = {}
        if hasattr(self, "per_query_select_parallel") and self.per_query_select_parallel:
            per_query_parallel = self.workload.query_aliases

        # Build the knob space so we can actually get the knobs.
        if (not hasattr(self, "knob_space")) or self.knob_space:
            ks = KnobSpace(
                self.tables,
                self.system_knobs,
                self.table_level_knobs,
                self.per_query_knobs,
                self.per_query_knob_gen,
                self.enable_quantize,
                self.default_quantization_factor,
                seed,
                per_query_parallel=per_query_parallel,
                per_query_scans=per_query_scans,
                query_names=self.workload.order)
        else:
            ks = None

        lsc = None
        lsc_embed = False
        if hasattr(self, "lsc_parameters") and self.lsc_parameters["lsc_enabled"]:
            lsc = LSC(self, horizon, vae_config)
            lsc_embed = self.lsc_parameters["lsc_embed"]

        if not hasattr(self, "workload_eval_reset"):
            self.workload_eval_reset = False

        if (not hasattr(self, "index_space")) or self.index_space:
            idxs = IndexSpace(
                agent_type=agent_type,
                tables=self.tables,
                max_num_columns=self.max_num_columns,
                index_repr=self.index_repr,
                seed=seed,
                latent_dim=index_latent_dim,
                index_output_scale=index_output_scale,
                index_output_func=index_output_func,
                index_vae_config=(vae_config, self.index_vae_metadata.get("embedding_pth_path", None)),
                attributes_overwrite=modified_attrs,
                tbl_include_subsets=tbl_include_subsets,
                lsc=lsc,
                scale_noise_perturb=self.scale_noise_perturb,
                index_space_aux_type=getattr(self, "index_space_aux_type", False),
                index_space_aux_include=getattr(self, "index_space_aux_include", False))
            self.max_num_columns = idxs.max_num_columns
        else:
            idxs = None
            self.max_num_columns = 0
        # Update the maximum number of columns?

        self.action_space = ActionSpace(ks, idxs, seed, lsc_embed)
        self.observation_space = self.__build_observation_space(seed)

    def save_state(self):
        # Save what can't be reconstructed correctly.
        return {
            "original_benchbase_config_path": self.original_benchbase_config_path,
            "connection": self.connection,
            "workload": self.workload.save_state(),
            "action_space": self.action_space.save_state(),
            #"observation_space": self.observation_space.save_state(),
        }

    def load_state(self, data):
        self.workload.load_state(data["workload"])
        self.action_space.load_state(data["action_space"])
