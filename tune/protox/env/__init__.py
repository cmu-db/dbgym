from enum import unique, Enum
from gymnasium.envs.registration import register
import logging

register(
    id="Postgres-v0",
    entry_point="envs.pg_env:PostgresEnv",
)

@unique
class SettingType(Enum):
    INVALID = -1
    BOOLEAN = 0
    INTEGER = 1
    BYTES = 2
    INTEGER_TIME = 3
    FLOAT = 4

    BINARY_ENUM = 5
    SCANMETHOD_ENUM = 6
    SCANMETHOD_ENUM_CATEGORICAL = 7
    MAGIC_HINTSET_ENUM_CATEGORICAL = 8
    QUERY_TABLE_ENUM = 9

@unique
class KnobClass(Enum):
    INVALID = -1
    KNOB = 0
    TABLE = 1
    QUERY = 2


def is_knob_enum(knob):
    return knob.knob_type in [
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM,
        SettingType.SCANMETHOD_ENUM_CATEGORICAL,
        SettingType.MAGIC_HINTSET_ENUM_CATEGORICAL,
        SettingType.QUERY_TABLE_ENUM,
    ]

def is_binary_enum(knob):
    return knob.knob_type in [
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM,
    ]

def resolve_enum_value(knob, value, all_knobs={}):
    assert is_knob_enum(knob)
    if knob.knob_type == SettingType.BINARY_ENUM:
        return "on" if value == 1 else "off"

    if knob.knob_type == SettingType.QUERY_TABLE_ENUM:
        integral_value = int(value)
        if integral_value == 0:
            return ""

        assert "max_worker_processes" in all_knobs
        max_workers = all_knobs["max_worker_processes"]

        selected_table = knob.values[integral_value - 1]
        # FIXME: pg_hint_plan lets specifying any and then pg will tweak it down.
        return f"Parallel({selected_table} {max_workers})"

    if knob.knob_type in [SettingType.SCANMETHOD_ENUM, SettingType.SCANMETHOD_ENUM_CATEGORICAL]:
        assert "_scanmethod" in knob.knob_name
        tbl = knob.knob_name.split("_scanmethod")[0]
        if value == 1:
            return f"IndexOnlyScan({tbl})"
        return f"SeqScan({tbl})"

    if knob.knob_type in [SettingType.MAGIC_HINTSET_ENUM_CATEGORICAL]:
        # This is curated from BAO.
        assert knob.knob_class == KnobClass.QUERY

        enable_hashjoin = "off"
        enable_mergejoin = "off"
        enable_nestloop = "off"
        enable_indexscan = "off"
        enable_seqscan = "off"
        enable_indexonlyscan = "off"

        if value == 0:
            enable_hashjoin = "on"
            enable_indexscan = "on"
            enable_mergejoin = "on"
            enable_nestloop = "on"
            enable_seqscan = "on"
            enable_indexonlyscan = "on"
        elif value == 1:
            enable_hashjoin = "on"
            enable_indexonlyscan = "on"
            enable_indexscan = "on"
            enable_mergejoin = "on"
            enable_seqscan = "on"
        elif value == 2:
            enable_hashjoin = "on"
            enable_indexonlyscan = "on"
            enable_nestloop = "on"
            enable_seqscan = "on"
        elif value == 3:
            enable_hashjoin = "on"
            enable_indexonlyscan = "on"
            enable_seqscan = "on"
        elif value == 4:
            enable_hashjoin = "on"
            enable_indexonlyscan = "on"
            enable_indexscan = "on"
            enable_nestloop = "on"
            enable_seqscan = "on"
        else:
            assert False, print(knob, value)

        set_args = " ".join([
            f"Set (enable_hashjoin {enable_hashjoin})",
            f"Set (enable_mergejoin {enable_mergejoin})",
            f"Set (enable_nestloop {enable_nestloop})",
            f"Set (enable_indexscan {enable_indexscan})",
            f"Set (enable_seqscan {enable_seqscan})",
            f"Set (enable_indexonlyscan {enable_indexonlyscan})",
        ])
        return set_args

    assert False


def regress_ams(qid_knobs, access_method, explain):
    new_qid_knobs = []
    for (knob, v) in qid_knobs:
        if knob.knob_type in [SettingType.SCANMETHOD_ENUM, SettingType.SCANMETHOD_ENUM_CATEGORICAL]:
            assert "_scanmethod" in knob.knob_name
            alias = knob.knob_name.split("_scanmethod")[0]
            if alias in access_method:
                value = 1 if "Index" in access_method[alias] else 0
                new_qid_knobs.append((knob, value))
            else:
                # Log out the missing alias for debugging reference.
                logging.debug(f"Found missing {alias} in the parsed {access_method}.")
                logging.debug(f"{explain}")
                new_qid_knobs.append((knob, 0.))
        else:
            new_qid_knobs.append((knob, v))
    return new_qid_knobs


def regress_qid_knobs(qid_knobs, real_knobs, access_method, explain):
    global_qid_knobs = []
    for (knob, _) in qid_knobs:
        assert knob.knob_type != SettingType.MAGIC_HINTSET_ENUM_CATEGORICAL
        if knob.knob_name in real_knobs:
            global_qid_knobs.append((knob, real_knobs[knob.knob_name]))
        elif knob.knob_type in [SettingType.SCANMETHOD_ENUM, SettingType.SCANMETHOD_ENUM_CATEGORICAL]:
            assert "_scanmethod" in knob.knob_name
            alias = knob.knob_name.split("_scanmethod")[0]
            if alias in access_method:
                value = 1 if "Index" in access_method[alias] else 0
                global_qid_knobs.append((knob, value))
            else:
                # Log out the missing alias for debugging reference.
                logging.debug(f"Found missing {alias} in the parsed {access_method}.")
                logging.debug(f"{explain}")
                global_qid_knobs.append((knob, 0.))
        elif knob.knob_type == SettingType.BOOLEAN:
            global_qid_knobs.append((knob, 1.))
        elif knob.knob_name == "random_page_cost":
            global_qid_knobs.append((knob, knob.project_scraped_setting(4.)))
        elif knob.knob_name == "seq_page_cost":
            global_qid_knobs.append((knob, knob.project_scraped_setting(1.)))
        elif knob.knob_name == "hash_mem_multiplier":
            global_qid_knobs.append((knob, knob.project_scraped_setting(2.)))
        elif knob.is_categorical():
            global_qid_knobs.append((knob, knob.default_value))
    assert len(global_qid_knobs) == len(qid_knobs)
    return global_qid_knobs
