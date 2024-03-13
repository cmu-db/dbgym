import logging
import re
import xml.etree.ElementTree as ET
from distutils import util

from psycopg.rows import dict_row

from tune.protox.env import KnobClass, SettingType
from tune.protox.env.space.knob import CategoricalKnob, Knob, full_knob_name


def check_subspace(space, action):
    if not space.contains(action):
        for i, subspace in enumerate(space.spaces):
            if isinstance(subspace, str):
                print(f"subspace={subspace}")
                print(f"subspace in space.spaces={subspace in space.spaces}, subspace in action={subspace in action}")
                print(f"space.spaces.keys()={space.spaces.keys()}, action.keys()={action.keys()}")
                if not space.spaces[subspace].contains(action[subspace]):
                    logging.error("Subspace %s rejects %s", subspace, action[subspace])
                    return False
            elif not subspace.contains(action[i]):
                logging.error("Subspace %s rejects %s", subspace, action[i])
                return False
    return True


# Defines the relevant metrics that we care about from benchbase.
# <filter_db>: whether to filter with the benchbase database.
# <per_table>: whether to process the set of valid_keys per table.
METRICS_SPECIFICATION = {
    "pg_stat_database": {
        "filter_db": True,
        "per_table": False,
        "valid_keys": [
            "temp_files",
            "tup_returned",
            "xact_commit",
            "xact_rollback",
            "conflicts",
            "blks_hit",
            "blks_read",
            "temp_bytes",
            "deadlocks",
            "tup_inserted",
            "tup_fetched",
            "tup_updated",
            "tup_deleted",
        ],
    },
    "pg_stat_bgwriter": {
        "filter_db": False,
        "per_table": False,
        "valid_keys": [
            "checkpoint_write_time",
            "buffers_backend_fsync",
            "buffers_clean",
            "buffers_checkpoint",
            "checkpoints_req",
            "checkpoints_timed",
            "buffers_alloc",
            "buffers_backend",
            "maxwritten_clean",
        ],
    },
    "pg_stat_database_conflicts": {
        "filter_db": True,
        "per_table": False,
        "valid_keys": [
            "confl_deadlock",
            "confl_lock",
            "confl_bufferpin",
            "confl_snapshot",
        ],
    },
    "pg_stat_user_tables": {
        "filter_db": False,
        "per_table": True,
        "valid_keys": [
            "n_tup_ins",
            "n_tup_upd",
            "n_tup_del",
            "n_ins_since_vacuum",
            "n_mod_since_analyze",
            "n_tup_hot_upd",
            "idx_tup_fetch",
            "seq_tup_read",
            "autoanalyze_count",
            "autovacuum_count",
            "n_live_tup",
            "n_dead_tup",
            "seq_scan",
            "idx_scan",
        ],
    },
    "pg_statio_user_tables": {
        "filter_db": False,
        "per_table": True,
        "valid_keys": [
            "heap_blks_hit",
            "heap_blks_read",
            "idx_blks_hit",
            "idx_blks_read",
            "tidx_blks_hit",
            "tidx_blks_read",
            "toast_blks_hit",
            "toast_blks_read",
        ],
    },
}


# Convert a string time unit to microseconds.
def _time_unit_to_us(str):
    if str == "d":
        return 1e6 * 60 * 60 * 24
    elif str == "h":
        return 1e6 * 60 * 60
    elif str == "min":
        return 1e6 * 60
    elif str == "s":
        return 1e6
    elif str == "ms":
        return 1e3
    elif str == "us":
        return 1.0
    else:
        return 1.0


# Parse a pg_setting field value.
def _parse_field(type, value):
    if type == SettingType.BOOLEAN:
        return util.strtobool(value)
    elif type == SettingType.BINARY_ENUM:
        if "off" in value.lower():
            return False
        return True
    elif type == SettingType.INTEGER:
        return int(value)
    elif type == SettingType.BYTES:
        if value in ["-1", "0"]:
            # Hardcoded default/disabled values for this field.
            return int(value)
        bytes_regex = re.compile(r"(\d+)\s*([kmgtp]?b)", re.IGNORECASE)
        order = ("b", "kb", "mb", "gb", "tb", "pb")
        field_bytes = None
        for number, unit in bytes_regex.findall(value):
            field_bytes = int(number) * (1024 ** order.index(unit.lower()))
        assert (
            field_bytes is not None
        ), f"Failed to parse bytes from value string {value}"
        return field_bytes
    elif type == SettingType.INTEGER_TIME:
        if value == "-1":
            # Hardcoded default/disabled values for this field.
            return int(value)
        bytes_regex = re.compile(r"(\d+)\s*((?:d|h|min|s|ms|us)?)", re.IGNORECASE)
        field_us = None
        for number, unit in bytes_regex.findall(value):
            field_us = int(number) * _time_unit_to_us(unit)
        assert field_us is not None, f"Failed to parse time from value string {value}"
        return int(field_us)
    elif type == SettingType.FLOAT:
        return float(value)
    else:
        return None


def _project_pg_setting(knob: Knob, setting: str):
    # logging.debug(f"Projecting {setting} into knob {knob.knob_name}")
    value = _parse_field(knob.knob_type, setting)
    value = value if knob.knob_unit == 0 else value / knob.knob_unit
    return knob.project_scraped_setting(value)


def fetch_server_knobs(connection, tables, knobs, workload=None):
    knob_targets = {}
    with connection.cursor(row_factory=dict_row) as cursor:
        records = cursor.execute("SHOW ALL")
        for record in records:
            setting_name = record["name"]
            if setting_name in knobs:
                setting_str = record["setting"]
                value = _project_pg_setting(knobs[setting_name], setting_str)
                knob_targets[setting_name] = value

        for tbl in tables:
            pgc_record = [
                r
                for r in cursor.execute(
                    f"SELECT * FROM pg_class where relname = '{tbl}'", prepare=False
                )
            ][0]
            if pgc_record["reloptions"] is not None:
                for record in pgc_record["reloptions"]:
                    for key, value in re.findall(r"(\w+)=(\w*)", record):
                        tbl_key = full_knob_name(table=tbl, knob_name=key)
                        if tbl_key in knobs:
                            value = _project_pg_setting(knobs[tbl_key], value)
                            knob_targets[tbl_key] = value
            else:
                for knobname, knob in knobs.items():
                    if knob.knob_class == KnobClass.TABLE:
                        if knob.knob_name == "fillfactor":
                            tbl_key = full_knob_name(
                                table=tbl, knob_name=knob.knob_name
                            )
                            knob_targets[tbl_key] = _project_pg_setting(knob, 100.0)

    q_ams = None
    for knobname, knob in knobs.items():
        if knob.knob_class == KnobClass.QUERY:
            # Set the default to inherit from the base knob setting.
            if knob.knob_name in knob_targets:
                knob_targets[knobname] = knob_targets[knob.knob_name]
            elif isinstance(knob, CategoricalKnob):
                knob_targets[knobname] = knob.default_value
            elif knob.knob_name.endswith("_scanmethod"):
                assert knob.knob_name.endswith("_scanmethod")
                assert knob.query_name is not None
                installed = False
                if q_ams is None:
                    q_ams = {}
                    if workload is not None:
                        # Get all access methods.
                        q_ams = workload.parse_all_access_methods(connection)

                if knob.query_name in q_ams:
                    alias = knob.knob_name.split("_scanmethod")[0]
                    if alias in q_ams[knob.query_name]:
                        val = 1 if "Index" in q_ams[knob.query_name][alias] else 0
                        knob_targets[knobname] = val
                        installed = True

                if not installed:
                    knob_targets[knobname] = 0.0
            elif knob.knob_type == SettingType.BOOLEAN:
                knob_targets[knobname] = 1.0
            elif knob.knob_name == "random_page_cost":
                value = _project_pg_setting(knob, 4.0)
                knob_targets[knobname] = value
            elif knob.knob_name == "seq_page_cost":
                value = _project_pg_setting(knob, 1.0)
                knob_targets[knobname] = value
            elif knob.knob_name == "hash_mem_multiplier":
                value = _project_pg_setting(knob, 2.0)
                knob_targets[knobname] = value
    return knob_targets


def fetch_server_indexes(connection, tables):
    rel_metadata = {t: [] for t in tables}
    existing_indexes = {}
    with connection.cursor(row_factory=dict_row) as cursor:
        records = cursor.execute(
            """
            SELECT c.relname, a.attname
            FROM pg_attribute a, pg_class c
            WHERE a.attrelid = c.oid AND a.attnum > 0
            ORDER BY c.relname, a.attnum"""
        )
        for record in records:
            relname = record["relname"]
            attname = record["attname"]
            if relname in rel_metadata:
                rel_metadata[relname].append(attname)

        records = cursor.execute(
            """
            SELECT
                t.relname as table_name,
                i.relname as index_name,
                am.amname as index_type,
                a.attname as column_name,
                array_position(ix.indkey, a.attnum) pos,
                (array_position(ix.indkey, a.attnum) >= ix.indnkeyatts) as is_include
            FROM pg_class t, pg_class i, pg_index ix, pg_attribute a, pg_am am
            WHERE t.oid = ix.indrelid
            and am.oid = i.relam
            and i.oid = ix.indexrelid
            and a.attrelid = t.oid
            and a.attnum = ANY(ix.indkey)
            and t.relkind = 'r'
            and ix.indisunique = false
            order by t.relname, i.relname, pos;
        """
        )

        for record in records:
            relname = record["table_name"]
            idxname = record["index_name"]
            colname = record["column_name"]
            index_type = record["index_type"]
            is_include = record["is_include"]
            if relname in rel_metadata:
                if relname not in existing_indexes:
                    existing_indexes[relname] = {}

                if idxname not in existing_indexes[relname]:
                    existing_indexes[relname][idxname] = {
                        "index_type": index_type,
                        "columns": [],
                        "include": [],
                    }

                if is_include:
                    existing_indexes[relname][idxname]["include"].append(colname)
                else:
                    existing_indexes[relname][idxname]["columns"].append(colname)
    return rel_metadata, existing_indexes


def overwrite_benchbase_hintset(root, query_name, set_str):
    ttypes = root.find("transactiontypes")
    for ttype in ttypes:
        name = ttype.find("name")
        if name is None:
            continue

        if name.text == query_name:
            hint_set = ttype.find("hintset")
            if hint_set is None:
                hint_set = ET.Element("hintset")
                hint_set.text = set_str
                ttype.append(hint_set)
            else:
                # Append the hint.
                hint_set.text = hint_set.text + " " + set_str

            break
