import json
import logging
import re
from pathlib import Path
from typing import Any

import click
import tensorflow
from google.protobuf.json_format import MessageToJson
from tensorflow.core.util.event_pb2 import Event

from util.log import DBGYM_OUTPUT_LOGGER_NAME


@click.group(name="analyze")
def analyze_group() -> None:
    pass


@click.command(name="tfevents")
@click.argument("tfevents-path", type=Path)
def analyze_tfevents(tfevents_path: Path) -> None:
    minimal_json = tboard_to_minimal_json(tfevents_path)
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"seconds spent resetting: {get_total_instr_time_event(minimal_json, r'.*PostgresEnv_reset$')}"
    )
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"seconds spent reconfiguring: {get_total_instr_time_event(minimal_json, r'.*PostgresEnv_shift_state$')}"
    )
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"seconds spent evaluating workload: {get_total_instr_time_event(minimal_json, r'.*Workload_execute$')}"
    )
    logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME).info(
        f"seconds spent training agent: {get_total_instr_time_event(minimal_json, r'.*(WolpPolicy_train_actor|WolpPolicy_train_critic)$')}"
    )


# The "minimal json" unwraps each summary so that we're left only with the parts that differ between summaries
def tboard_to_minimal_json(tfevent_fpath: Path) -> list[dict[str, Any]]:
    minimal_json = []

    raw_dataset = tensorflow.data.TFRecordDataset(tfevent_fpath)

    for raw_record in raw_dataset:
        event = Event()
        event.ParseFromString(raw_record.numpy())

        # Convert event to JSON
        json_summary = json.loads(MessageToJson(event.summary))

        # We get a {} at the very start
        if json_summary == {}:
            continue

        assert "value" in json_summary
        json_summary = json_summary["value"]
        assert len(json_summary) == 1
        json_summary = json_summary[0]

        minimal_json.append(json_summary)

    return minimal_json


# An "instr_time_event" is an event with a "tag" that looks like "instr_time/*"
def get_total_instr_time_event(
    minimal_json: list[dict[str, Any]], event_regex: str
) -> float:
    event_pattern = re.compile(event_regex)
    total_time = 0

    for json_summary in minimal_json:
        if event_pattern.fullmatch(json_summary["tag"]) is not None:
            total_time += json_summary["simpleValue"]

    return total_time


analyze_group.add_command(analyze_tfevents)
