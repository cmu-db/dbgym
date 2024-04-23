"""
Replaying a tuning run gives you the authoritative runtimes of that tuning run.
The original tuning run has per-query timeouts, so the runtimes may be inaccurate. The
    replayed tuning run does not have per-query timeouts.
Additionally, the original tuning run may have been accelerated by Boot, whereas the
    replayed tuning run is not.
"""
import datetime
import json
import logging
import pickle
import click
import numpy as np
import pandas as pd
import tqdm
from pathlib import Path
from dateutil.parser import parse

from misc.utils import DEFAULT_BOOT_CONFIG_FPATH, DEFAULT_WORKLOAD_TIMEOUT, DBGymConfig, TuningMode, conv_inputpath_to_realabspath, open_and_save, save_file, workload_name_fn, default_tuning_steps_dpath
# sys.path.append("/home/phw2/dbgym") # TODO(phw2): figure out if this is required

from tune.protox.agent.build_trial import build_trial
from tune.protox.env.pg_env import PostgresEnv
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.space.utils import fetch_server_indexes, fetch_server_knobs
from tune.protox.env.types import HolonAction, IndexSpaceRawSample


REPLAY_DATA_FNAME = "replay_data.csv"


class ReplayArgs:
    def __init__(
        self, workload_timeout_during_replay: bool, simulated: bool, cutoff: float, blocklist: list
    ):
        self.workload_timeout_during_replay = workload_timeout_during_replay
        self.simulated = simulated
        self.cutoff = cutoff
        self.blocklist = blocklist


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.option("--seed-start", type=int, default=15721, help="A workload consists of queries from multiple seeds. This is the starting seed (inclusive).")
@click.option("--seed-end", type=int, default=15721, help="A workload consists of queries from multiple seeds. This is the ending seed (inclusive).")
@click.option(
    "--query-subset",
    type=click.Choice(["all", "even", "odd"]),
    default="all",
)
@click.option(
    "--scale-factor",
    default=1.0,
    help="The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--boot-enabled-during-tune",
    is_flag=True,
    help="Whether Boot was enabled during tuning.",
)
@click.option(
    "--tuning-steps-dpath",
    default=None,
    type=Path,
    help="The path to the `tuning_steps` directory to be replayed."
)
@click.option(
    "--workload-timeout-during-replay",
    default=None,
    type=int,
    # You can make it use the workload timeout used during tuning if you want.
    # I just made it use the workload timeout from HPO because I don't currently persist the tuning HPO params.
    help="The timeout (in seconds) of a workload when replaying. By default, it will be equal to the workload timeout used during HPO."
)
@click.option(
    "--simulated",
    is_flag=True,
    help="Set to true to use the runtimes from the original tuning run instead of replaying the workload."
)
@click.option(
    "--cutoff",
    default=None,
    type=float,
    help="Only evaluate configs up to cutoff hours. None means \"evaluate all configs\"."
)
@click.option(
    "--blocklist",
    default=[],
    type=list,
    help="Ignore running queries in the blocklist."
)
def replay(dbgym_cfg: DBGymConfig, benchmark_name: str, seed_start: int, seed_end: int, query_subset: str, scale_factor: float, boot_enabled_during_tune: bool, tuning_steps_dpath: Path, workload_timeout_during_replay: bool, simulated: bool, cutoff: float, blocklist: list) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    workload_name = workload_name_fn(scale_factor, seed_start, seed_end, query_subset)

    if tuning_steps_dpath == None:
        tuning_steps_dpath = default_tuning_steps_dpath(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, boot_enabled_during_tune)

    # Convert all input paths to absolute paths
    tuning_steps_dpath = conv_inputpath_to_realabspath(dbgym_cfg, tuning_steps_dpath)

    # Group args together to reduce the # of parameters we pass into functions
    replay_args = ReplayArgs(workload_timeout_during_replay, simulated, cutoff, blocklist)

    # Replay
    replay_tuning_run(dbgym_cfg, tuning_steps_dpath, replay_args)


def replay_tuning_run(dbgym_cfg: DBGymConfig, tuning_steps_dpath: Path, replay_args: ReplayArgs):
    """
    Replay a single tuning run (as in one tuning_steps/ folder).
    """
    def _is_tuning_step_line(line: str) -> bool:
        return "mv" in line and "tuning_steps" in line and "baseline" not in line

    hpo_params_fpath = tuning_steps_dpath / "params.json"
    with open_and_save(dbgym_cfg, hpo_params_fpath, "r") as f:
        hpo_params = json.load(f)

    # Set defaults that depend on hpo_params
    if replay_args.workload_timeout_during_replay == None:
        replay_args.workload_timeout_during_replay = hpo_params["workload_timeout"][str(TuningMode.HPO)]

    # Set the hpo_params that are allowed to differ between HPO, tuning, and replay.
    hpo_params["enable_boot"][str(TuningMode.REPLAY)] = False
    hpo_params["boot_config_fpath"][str(TuningMode.REPLAY)] = None
    hpo_params["workload_timeout"][str(TuningMode.REPLAY)] = replay_args.workload_timeout_during_replay

    # Go through output.log and find the tuning_steps/[time]/ folders
    # This finds all the [time] folders in tuning_steps/ (except "baseline" since we ignore that in `_is_tuning_step_line()`),
    #   so you could just do `ls tuning_steps/` if you wanted to.
    folders = []
    start_found = False
    output_log_fpath = tuning_steps_dpath / "output.log"
    with open_and_save(dbgym_cfg, output_log_fpath, "r") as f:
        for line in f:
            if not start_found:
                if "Baseline Metric" in line:
                    start_time = parse(line.split("INFO:")[-1].split(" Baseline Metric")[0].split("[")[0])
                    start_found = True
            else:
                if _is_tuning_step_line(line):
                    repo = eval(line.split("Running ")[-1])[-1]
                    last_folder = repo.split("/")[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0].split("[")[0])
                    if replay_args.cutoff == None or (time_since_start - start_time).total_seconds() < replay_args.cutoff * 3600:
                        folders.append(last_folder)
    
    # Set tune_duration to be high so that it doesn't cut the replay off early
    hpo_params["tune_duration"][str(TuningMode.REPLAY)] = replay_args.workload_timeout_during_replay * len(folders)

    # Build PostgresEnv.
    _, _, agent_env, _, _ = build_trial(dbgym_cfg, TuningMode.REPLAY, hpo_params["seed"], hpo_params)
    pg_env: PostgresEnv = agent_env.unwrapped
    action_space: HolonSpace = pg_env.action_space

    # Reset things.
    if not replay_args.simulated:
        pg_env.pg_conn.restore_pristine_snapshot()

    num_lines = 0
    with open_and_save(dbgym_cfg, output_log_fpath, "r") as f:
        for line in f:
            if "Baseline Metric" in line:
                num_lines += 1
            elif _is_tuning_step_line(line):
                num_lines += 1

    # A convenience wrapper around execute_workload() which fills in the arguments properly
    def _execute_workload_wrapper(actions_info: list["HolonAction"]) -> list[float]:
        logging.info(f"\n\nfetch_server_knobs(): {fetch_server_knobs(pg_env.pg_conn.conn(), action_space.get_knob_space().tables, action_space.get_knob_space().knobs, pg_env.workload.queries)}\n\n")
        logging.info(f"\n\nfetch_server_indexes(): {fetch_server_indexes(pg_env.pg_conn.conn(), action_space.get_knob_space().tables)}\n\n")
        assert replay_args.workload_timeout_during_replay == hpo_params["workload_timeout"][str(TuningMode.REPLAY)] == pg_env.workload.workload_timeout, "All these different sources of workload_timeout during replay should show the same value"
        all_holon_action_variations = actions_info["all_holon_action_variations"]
        replayed_runtime = pg_env.workload.execute_workload(
            pg_conn=pg_env.pg_conn,
            actions=[holon_action for (_, holon_action) in all_holon_action_variations],
            variation_names=[variation_name for (variation_name, _) in all_holon_action_variations],
            observation_space=None,
            action_space=action_space,
            reset_metrics=None,
            query_timeout=None,
            workload_qdir=None,
            blocklist=replay_args.blocklist,
            first=False,
        )
        assert type(replayed_runtime) is float, "Workload.execute_workload() can return either a float or a tuple. During replay, we must ensure that it returns a float."
        return replayed_runtime

    run_data = []
    progess_bar = tqdm.tqdm(total=num_lines)
    with open_and_save(dbgym_cfg, output_log_fpath, "r") as f:
        current_step = 0
        start_found = False
        start_time = None
        maximal_repo = None
        existing_index_acts = []

        for line in f:
            # Keep going until we've found the start.
            if not start_found:
                if "Baseline Metric" in line:
                    start_found = True
                    start_time = parse(line.split("INFO:")[-1].split(" Baseline Metric")[0].split("[")[0])
                    progess_bar.update(1)
                continue

            elif _is_tuning_step_line(line):
                if _is_tuning_step_line(line):
                    repo = eval(line.split("Running ")[-1])[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0].split("[")[0])
                elif "Found new maximal state with" in line:
                    repo = eval(maximal_repo.split("Running ")[-1])[-1]
                    time_since_start = parse(maximal_repo.split("DEBUG:")[-1].split(" Running")[0].split("[")[0])
                    maximal_repo = None

                # Get the original runtime as well as whether any individual queries and/or the full workload timed out.
                run_raw_csv_fpath = tuning_steps_dpath / repo / "run.raw.csv"
                save_file(dbgym_cfg, run_raw_csv_fpath)
                run_raw_csv = pd.read_csv(run_raw_csv_fpath)
                assert len(run_raw_csv.columns) == 7
                # `did_any_query_time_out_in_original` will be true when *all variations* of at least one query of the original workload did not execute
                #   to completion, regardless of how it happened. Even if this was because there was only 1s before the workload timed out and thus the
                #   query was "unfairly" given a 1s "statement_timeout", we will still set `did_any_query_time_out_in_original` to true because that query
                #   didn't not execute to completion.
                # When setting `did_any_query_time_out_in_original`, we can't just check whether the latency in run.raw.csv == `query_timeout` because
                #   this doesn't handle the edge case where the "statement_timeout" setting in Postgres is set to be < `query_timeout`. This edge case
                #   would happen when the amount of time remaining before we hit `workload_timeout` is less then `query_timeout` and thus Proto-X sets
                #   "statement_timeout" to be < `query_timeout` in order to not exceed the `workload_timeout`.
                did_any_query_time_out_in_original = any(run_raw_csv["Timed Out"])
                # When setting `did_workload_time_out_in_original`, we can't just check whether the sum of latencies in run.raw.csv == `workload_timeout`
                #   because Proto-X decreases `workload_timeout` over the course of the tuning run. Specifically, at the end of a tuning step, Proto-X
                #   sets `workload_timeout` to be equal to the runtime of the workload that just ran.
                # We separate the penalty rows from the non-penalty rows to process them separately.
                run_raw_csv_penalty_rows = run_raw_csv[run_raw_csv["Transaction Name"] == "P"]
                run_raw_csv_non_penalty_rows = run_raw_csv[run_raw_csv["Transaction Name"] != "P"]
                # Penalties are added when the workload times out so this is a reliable indicator of whether the workload timed out.
                did_workload_time_out_in_original = len(run_raw_csv_penalty_rows) > 0
                # Penalties are meant to affect the reward of the tuning agent but they are unrelated to the actual runtime, so we ignore them when
                #   computing the original runtime.
                original_runtime = run_raw_csv_non_penalty_rows["Latency (microseconds)"].sum() / 1e6
                assert original_runtime > 0

                # Extract the necessary values from action.pkl
                with open_and_save(dbgym_cfg, tuning_steps_dpath / repo / "action.pkl", "rb") as f:
                    actions_info = pickle.load(f)
                    all_holon_action_variations = actions_info["all_holon_action_variations"]
                    # Extract the KnobSpaceAction and IndexAction from all_holon_action_variations.
                    # These two should be identical across all HolonActions, which we will assert.
                    _, first_holon_action = all_holon_action_variations[0]
                    knob_space_action = first_holon_action[0]
                    index_space_raw_sample = first_holon_action[1]
                    index_action = action_space.get_index_space().to_action(index_space_raw_sample)
                    assert all([knob_space_action == holon_action[0] for (_, holon_action) in all_holon_action_variations])
                    assert all([index_action == action_space.get_index_space().to_action(holon_action[1]) for (_, holon_action) in all_holon_action_variations])

                # Get the indexes from this action and the prior state
                index_acts = set()
                index_acts.add(index_action)
                assert len(index_acts) > 0
                with open_and_save(dbgym_cfg, tuning_steps_dpath / repo / "prior_state.pkl", "rb") as f:
                    prior_states = pickle.load(f)
                    all_sc = set(prior_states[1])
                    for index_act in index_acts:
                        all_sc.add(index_act)

                    all_sc = {a for a in all_sc if not "USING btree ()" in a.sql(True)}
                    index_acts = all_sc
                # Get the CREATE INDEX or DROP INDEX statements to turn the state into the one we should be in at this tuning step
                index_modification_sqls = []
                for index_act in index_acts:
                    if index_act not in existing_index_acts:
                        index_modification_sqls.append(index_act.sql(True))
                for existing_index_act in existing_index_acts:
                    if existing_index_act not in index_acts:
                        index_modification_sqls.append(existing_index_act.sql(False))

                # Modify Postgres to have the right indexes and system-wide knobs. `index_modification_sqls` holds the indexes
                #   while `cc` holds the system-wide knobs.
                if not replay_args.simulated:
                    cc, _ = action_space.get_knob_space().generate_action_plan(knob_space_action, prior_states[0])
                    # Like in tuning, we don't dump the page cache when calling shift_state() to see how the workload
                    #   performs in a warm cache scenario.
                    pg_env.shift_state(cc, index_modification_sqls, dump_page_cache=False)
                existing_index_acts = index_acts

                # Execute the workload to get the runtime.
                if not replay_args.simulated:
                    replayed_runtime = _execute_workload_wrapper(actions_info)
                    logging.info(f"Original Runtime: {original_runtime} (timed out? {did_any_query_time_out_in_original}). Replayed Runtime: {replayed_runtime}")
                else:
                    replayed_runtime = original_runtime

                # Add this tuning step's data to `run_data``.
                run_data.append({
                    "step": current_step,
                    "original_runtime": original_runtime,
                    "did_any_query_time_out_in_original": did_any_query_time_out_in_original,
                    "did_workload_time_out_in_original": did_workload_time_out_in_original,
                    "time_since_start": (time_since_start - start_time).total_seconds(),
                    "replayed_runtime": replayed_runtime,
                })
                current_step += 1

                run_folder = repo.split("/")[-1]
                if run_folder in folders and run_folder == folders[-1]:
                    break
            progess_bar.update(1)

    # Output.
    run_data_df = pd.DataFrame(run_data)
    pd.set_option('display.max_columns', 10)
    print(f"Finished replaying with run_data_df=\n{run_data_df}\n. Data stored in {dbgym_cfg.cur_task_runs_path()}.")
    run_data_df.to_csv(dbgym_cfg.cur_task_runs_data_path("run_data.csv"), index=False)
    pg_env.close()