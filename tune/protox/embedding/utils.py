import gc
import logging
import os

import pandas as pd
import torch
from hyperopt import hp
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from misc.utils import open_and_save
from tune.protox.embedding.loss import COST_COLUMNS
from tune.protox.env.space.index_policy import IndexRepr
from tune.protox.env.space.index_space import IndexSpace
from tune.protox.env.workload import Workload


def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively.
    There should be no duplicated keys across all nested
    subdictionaries, or some instances will be lost without warning

    Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """
    res = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            res = {**res, k: v, **f_unpack_dict(v)}
        else:
            res[k] = v
    return res


def parse_hyperopt_config(config):
    assert isinstance(config, dict)

    def parse_key(key_dict):
        if key_dict["type"] == "constant":
            return key_dict["value"]
        elif key_dict["type"] == "uniform":
            return hp.uniform(key_dict["choice_name"], key_dict["min"], key_dict["max"])
        elif key_dict["type"] == "choice":
            return hp.choice(key_dict["choice_name"], key_dict["choices"])
        elif key_dict["type"] == "subspaces":
            subspaces = [parse_hyperopt_config(c) for c in key_dict["subspaces"]]
            return hp.choice(key_dict["choice_name"], subspaces)
        else:
            print("Unknown hyperopt config definition", key_dict)
            assert False

    parsed_config = {}
    for key, key_dict in config.items():
        parsed_config[key] = parse_key(key_dict)
    return parsed_config


def fetch_index_parameters(cfg, benchmark, data):
    tables = data["protox"]["tables"]
    attributes = data["protox"]["attributes"]
    query_spec = data["protox"]["query_spec"]

    # TODO(phw2): figure out how to pass query_directory. should it in the .yaml or should it be a CLI args?
    if "query_directory" not in query_spec:
        assert "query_order" not in query_spec
        query_spec["query_directory"] = os.path.join(
            cfg.dbgym_data_path, f"{benchmark}_queries"
        )
        query_spec["query_order"] = os.path.join(
            query_spec["query_directory"], f"order.txt"
        )

    workload = Workload(cfg, tables, attributes, query_spec, pid=None)
    att_usage = workload.process_column_usage()

    space = IndexSpace(
        "wolp",
        tables,
        max_num_columns=0,
        index_repr=IndexRepr.ONE_HOT.name,
        seed=0,
        latent_dim=0,
        attributes_overwrite=att_usage,
    )
    space._build_mapping(att_usage)
    max_cat_features = max(
        len(tables), space.max_num_columns + 1
    )  # +1 for the one hot encoding.
    max_attrs = space.max_num_columns + 1  # +1 to account for the table index.
    return max_attrs, max_cat_features, att_usage, space.class_mapping


def load_input_data(cfg, input_path, train_size, max_attrs, require_cost, seed):
    # Load the input data.
    columns = []
    columns += ["tbl_index", "idx_class"]
    columns += [f"col{c}" for c in range(max_attrs - 1)]
    if require_cost:
        columns += COST_COLUMNS

    with open_and_save(cfg, input_path, mode="rb") as input_file:
        df = pd.read_parquet(input_file, columns=columns)
    num_classes = df.idx_class.max() + 1

    # Get the y's and the x's.
    targets = (COST_COLUMNS + ["idx_class"]) if require_cost else ["idx_class"]
    y = df[targets].values
    df.drop(columns=COST_COLUMNS + ["idx_class"], inplace=True, errors="ignore")
    x = df.values
    del df
    gc.collect()
    gc.collect()

    if train_size == 1:
        train_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        del x
        gc.collect()
        gc.collect()
        return train_dataset, y, y[:, -1], None, num_classes

    # Perform the train test split.
    train_x, val_x, train_y, val_y = train_test_split(
        x,
        y,
        test_size=1 - train_size,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y[:, -1],
    )
    del x
    del y
    gc.collect()
    gc.collect()

    # Form the tensor datasets.
    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    del val_x
    del val_y
    del train_x
    gc.collect()
    gc.collect()
    logging.info("Train Dataset Size: %s", len(train_dataset))
    return train_dataset, train_y, train_y[:, -1], val_dataset, num_classes
