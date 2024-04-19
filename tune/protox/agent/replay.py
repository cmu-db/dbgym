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
import pandas as pd
import tqdm
from pathlib import Path
from dateutil.parser import parse

from misc.utils import DEFAULT_BOOT_CONFIG_FPATH, DEFAULT_WORKLOAD_TIMEOUT, DBGymConfig, TuningMode, conv_inputpath_to_realabspath, open_and_save, save_file, workload_name_fn, default_tuning_steps_dpath
# sys.path.append("/home/phw2/dbgym") # TODO(phw2): figure out if this is required

from tune.protox.agent.build_trial import build_trial
from tune.protox.env.pg_env import PostgresEnv
from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.space.utils import fetch_server_indexes, fetch_server_knobs
from tune.protox.env.types import HolonAction


REPLAY_DATA_FNAME = "replay_data.csv"


class ReplayArgs:
    def __init__(
        self, workload_timeout_during_replay: bool, num_samples: int, threshold: float, threshold_limit: float, maximal: bool, simulated: bool, maximal_only: bool, cutoff: float, blocklist: list
    ):
        self.workload_timeout_during_replay = workload_timeout_during_replay
        self.num_samples = num_samples
        self.threshold = threshold
        self.threshold_limit = threshold_limit
        self.maximal = maximal
        self.simulated = simulated
        self.maximal_only = maximal_only
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
    "--num-samples",
    default=1,
    type=int,
    help="The number of times to run the workload for each DBMS config being evaluated."
)
@click.option(
    "--threshold",
    default=0,
    type=float,
    help="The minimum delta between the runtimes of consecutive DBMS configs to warrant a config being evaluated."
)
@click.option(
    "--threshold-limit",
    default=None,
    type=float,
    help="Only use threshold within threshold-limit hours from the start. None means \"always use threshold\"."
)
@click.option(
    "--maximal",
    is_flag=True,
    help="If set to true, only evaluate configs that are strictly \"better\"."
)
@click.option(
    "--simulated",
    is_flag=True,
    help="Set to true to use the runtimes from the original tuning run instead of replaying the workload."
)
@click.option(
    "--maximal-only",
    is_flag=True,
    help="If set to true, only evaluate the best config"
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
def replay(dbgym_cfg: DBGymConfig, benchmark_name: str, seed_start: int, seed_end: int, query_subset: str, scale_factor: float, boot_enabled_during_tune: bool, tuning_steps_dpath: Path, workload_timeout_during_replay: bool, num_samples: int, threshold: float, threshold_limit: float, maximal: bool, simulated: bool, maximal_only: bool, cutoff: float, blocklist: list) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    workload_name = workload_name_fn(scale_factor, seed_start, seed_end, query_subset)

    if tuning_steps_dpath == None:
        tuning_steps_dpath = default_tuning_steps_dpath(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, boot_enabled_during_tune)

    # Convert all input paths to absolute paths
    tuning_steps_dpath = conv_inputpath_to_realabspath(dbgym_cfg, tuning_steps_dpath)

    # Group args together to reduce the # of parameters we pass into functions
    replay_args = ReplayArgs(workload_timeout_during_replay, num_samples, threshold, threshold_limit, maximal, simulated, maximal_only, cutoff, blocklist)

    # Replay
    replay_tuning_run(dbgym_cfg, tuning_steps_dpath, replay_args)


def replay_tuning_run(dbgym_cfg: DBGymConfig, tuning_steps_dpath: Path, replay_args: ReplayArgs):
    """
    Replay a single tuning run (as in one tuning_steps/ folder).
    """
    def _is_tuning_step_line(line: str) -> bool:
        return "mv" in line and "tuning_steps" in line and "baseline" not in line

    maximal = replay_args.maximal
    maximal_only = replay_args.maximal_only
    threshold = replay_args.threshold

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

    # Go through output.log and find the tuning_steps/[time]/ folders, as well as the time of the last folder
    # This finds all the [time] folders in tuning_steps/ (except "baseline" since we ignore that in `_is_tuning_step_line()`),
    #   so you could just do `ls tuning_steps/` if you wanted to.
    folders = []
    start_found = False
    last_evaluation = None
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
                    last_evaluation = time_since_start
                    if replay_args.cutoff == None or (time_since_start - start_time).total_seconds() < replay_args.cutoff * 3600:
                        folders.append(last_folder)
    
    # Set tune_duration to be high so that it doesn't cut the replay off early
    hpo_params["tune_duration"][str(TuningMode.REPLAY)] = replay_args.workload_timeout_during_replay * len(folders)

    # Only apply threshold if time is less than.
    threshold_limit = last_evaluation - datetime.timedelta(seconds=int(replay_args.threshold_limit * 3600)) if replay_args.threshold_limit != None else None

    # Build PostgresEnv.
    _, _, agent_env, _, _ = build_trial(dbgym_cfg, TuningMode.REPLAY, hpo_params["seed"], hpo_params)
    pg_env: PostgresEnv = agent_env.unwrapped

    # Reset things.
    if not replay_args.simulated:
        pg_env.pg_conn.restore_pristine_snapshot()

    # Get the minimum reward.
    run_raw_csv_fpaths = [tuning_steps_dpath / fold / "run.raw.csv" for fold in folders]
    run_raw_csvs = [pd.read_csv(run_raw_csv_fpath) for run_raw_csv_fpath in run_raw_csv_fpaths]
    original_runtime_infos = [(run_raw_csv["Latency (microseconds)"].sum() / 1e6, (run_raw_csv["Latency (microseconds)"].max() / 1e6) == hpo_params["query_timeout"]) for run_raw_csv in run_raw_csvs]
    original_runtime_infos = sorted(original_runtime_infos, key=lambda x: x[0])
    min_original_runtime = min([r[0] for r in original_runtime_infos])

    if maximal:
        target = [r[1] for r in original_runtime_infos if r[0] == min_original_runtime]
        assert len(target) >= 1
        if target[0]:
            # Don't use maximal if the min maximal is timed out.
            # Don't threshold either.
            threshold = 0
            maximal = False
            # Reject maximal only.
            maximal_only = False
            logging.warn("Maximal disabled.")
        else:
            logging.info(f"Maximal found: {min_original_runtime}")

    num_lines = 0
    with open_and_save(dbgym_cfg, output_log_fpath, "r") as f:
        for line in f:
            if "Baseline Metric" in line:
                num_lines += 1
            elif _is_tuning_step_line(line):
                num_lines += 1

    def _run_sample(action_info: "HolonAction") -> list[float]:
        samples = []
        for _ in range(replay_args.num_samples):
            logging.info(f"\n\nfetch_server_knobs(): {fetch_server_knobs(pg_env.pg_conn.conn(), pg_env.action_space.get_knob_space().tables, pg_env.action_space.get_knob_space().knobs, pg_env.workload.queries)}\n\n")
            logging.info(f"\n\nfetch_server_indexes(): {fetch_server_indexes(pg_env.pg_conn.conn(), pg_env.action_space.get_knob_space().tables)}\n\n")
            assert replay_args.workload_timeout_during_replay == hpo_params["workload_timeout"][str(TuningMode.REPLAY)] == pg_env.workload.workload_timeout, "All these different sources of workload_timeout during replay should show the same value"
            runtime = pg_env.workload.execute_workload(
                pg_conn=pg_env.pg_conn,
                actions=[action_info],
                actions_names=["Replay"],
                observation_space=None,
                action_space=pg_env.action_space,
                reset_metrics=None,
                query_timeout=None,
                workload_qdir=None,
                disable_pg_hint=False,
                blocklist=replay_args.blocklist,
                first=False,
            )
            samples.append(runtime)
            logging.info(f"Runtime: {runtime}")

            if runtime >= replay_args.workload_timeout_during_replay:
                break

        return samples

    run_data = []
    pbar = tqdm.tqdm(total=num_lines)
    with open_and_save(dbgym_cfg, output_log_fpath, "r") as f:
        current_step = 0
        start_found = False
        start_time = None
        cur_reward_max = replay_args.workload_timeout_during_replay
        noop_index = False
        maximal_repo = None
        existing_index_acts = []
        if1_count = 0
        if2_count = 0

        for line in f:
            # Keep going until we've found the start.
            if not start_found:
                if "Baseline Metric" in line:
                    start_found = True
                    start_time = parse(line.split("INFO:")[-1].split(" Baseline Metric")[0].split("[")[0])
                    pbar.update(1)
                continue

            elif "Selected action: " in line:
                act = eval(line.split("Selected action: ")[-1])
                noop_index = "NOOP" in act[1][0]

            elif (maximal and (_is_tuning_step_line(line))):
                maximal_repo = line

            elif (maximal and "Found new maximal state with" in line) or (not maximal and _is_tuning_step_line(line)):
                if1_count += 1
                print(f"if1_count={if1_count}")

                if _is_tuning_step_line(line):
                    repo = eval(line.split("Running ")[-1])[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0].split("[")[0])
                elif "Found new maximal state with" in line:
                    repo = eval(maximal_repo.split("Running ")[-1])[-1]
                    time_since_start = parse(maximal_repo.split("DEBUG:")[-1].split(" Running")[0].split("[")[0])
                    maximal_repo = None

                # Get the evaluation reward.
                run_raw_csv_fpath = tuning_steps_dpath / repo / "run.raw.csv"
                save_file(dbgym_cfg, run_raw_csv_fpath)
                run_raw_csv = pd.read_csv(run_raw_csv_fpath)
                assert len(run_raw_csv.columns) == 6
                has_timeout = (run_raw_csv["Latency (microseconds)"].max() / 1e6) == hpo_params["query_timeout"]
                original_runtime = run_raw_csv["Latency (microseconds)"].sum() / 1e6
                assert original_runtime > 0

                if ((not replay_args.maximal_only and original_runtime < cur_reward_max) or original_runtime == min_original_runtime) and (not maximal or not has_timeout):
                    if2_count += 1
                    print(f"if2_count={if2_count}")

                    if if2_count >= 2:
                        break

                    index_acts = set()

                    with open_and_save(dbgym_cfg, tuning_steps_dpath / repo / "action.pkl", "rb") as f:
                        actions_info = pickle.load(f)
                        assert type(actions_info) is list and len(actions_info) == 1, f"there should only be one action in actions_info {actions_info}"
                        action_info = actions_info[0]
                        assert type(action_info) is tuple and len(action_info) == 3, f"action_info ({action_info}) should be a tuple with system knobs, an index, and per-query knobs"
                        index_acts.add(action_info[1])

                    assert len(index_acts) > 0
                    with open_and_save(dbgym_cfg, tuning_steps_dpath / repo / "prior_state.pkl", "rb") as f:
                        prior_states = pickle.load(f)
                        all_sc = set(prior_states[1])
                        if not noop_index:
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

                    if not replay_args.simulated:
                        # Apply index changes
                        cc, _ = pg_env.action_space.get_knob_space().generate_action_plan(action_info[0], prior_states[0])
                        # Like in tuning, we don't dump the page cache when calling shift_state() to see how the workload
                        #   performs in a warm cache scenario.
                        pg_env.shift_state(cc, index_modification_sqls)
                    existing_index_acts = index_acts

                    if not replay_args.simulated:
                        # Get samples.
                        run_samples = samples = _run_sample(action_info)
                        logging.info(f"Original Runtime: {original_runtime} (timed out? {has_timeout}). New Samples: {samples}")
                    else:
                        run_samples = samples = [original_runtime, original_runtime]

                    data = {
                        "step": current_step,
                        "original_runtime": original_runtime,
                        "time_since_start": (time_since_start - start_time).total_seconds(),
                    }
                    samples = {f"runtime{i}": s for i, s in enumerate(samples)}
                    data.update(samples)
                    run_data.append(data)

                    current_step += 1

                    if (not has_timeout) or (max(run_samples) < replay_args.workload_timeout_during_replay):
                        # Apply a tolerance..
                        # If we've timed out, only apply threshold only if we've found a strictly better config.
                        apply_threshold = threshold if threshold_limit == None or time_since_start < threshold_limit else 0
                        cur_reward_max = original_runtime - apply_threshold

                run_folder = repo.split("/")[-1]
                if run_folder in folders and run_folder == folders[-1]:
                    break
                elif maximal_only and original_runtime == min_original_runtime:
                    break
            pbar.update(1)

        if len(run_data) > 0:
            data = {
                "step": current_step,
                "original_runtime": run_data[-1]["original_runtime"],
                "time_since_start": -1,
                "runtime0": run_data[-1]["runtime0"],
            }
            run_data.append(data)

    # Output.
    run_data_df = pd.DataFrame(run_data)
    print(f"Finished replaying with run_data_df=\n{run_data_df}\n. Data stored in {dbgym_cfg.cur_task_runs_path()}.")
    run_data_df.to_csv(dbgym_cfg.cur_task_runs_data_path("run_data.csv"), index=False)
    pg_env.close()