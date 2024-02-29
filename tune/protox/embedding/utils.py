from hyperopt import hp


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
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, k:v, **f_unpack_dict(v)}
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
