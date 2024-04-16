import datetime
import logging
import click
import yaml
import pandas as pd
import tqdm
import argparse
from pathlib import Path
from dateutil.parser import parse

from misc.utils import DBGymConfig, conv_inputpath_to_realabspath, workload_name_fn, default_tuning_steps_dpath
# sys.path.append("/home/phw2/dbgym") # TODO(phw2): figure out if this is required

from tune.protox.env.pg_env import PostgresEnv

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


REPLAY_DATA_FNAME = "replay_data.csv"


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
    "--simulated",
    is_flag=True,
    help="Set to true to use the runtimes from the original tuning run instead of replaying the workload."
)
def replay(dbgym_cfg: DBGymConfig, benchmark_name: str, seed_start: int, seed_end: int, query_subset: str, scale_factor: float, boot_enabled_during_tune: bool, tuning_steps_dpath: Path, simulated: bool) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    workload_name = workload_name_fn(scale_factor, seed_start, seed_end, query_subset)
    if tuning_steps_dpath == None:
        tuning_steps_dpath = default_tuning_steps_dpath(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, boot_enabled_during_tune)

    # Convert all input paths to absolute paths
    tuning_steps_dpath = conv_inputpath_to_realabspath(dbgym_cfg, tuning_steps_dpath)

    # Replay
    runs = sorted(tuning_steps_dpath.rglob("config.yaml"))
    for run in tqdm.tqdm([f for f in runs], leave=False):
        print(f"Parsing {run.parent}")
        # gogo(new_args)


def gogo(args):
    maximal = args.maximal
    maximal_only = args.maximal_only
    threshold = args.threshold

    with open(f"{args.input}/config.yaml") as f:
        mythril = yaml.safe_load(f)
        mythril["mythril"]["benchbase_config_path"] = f"{args.input}/benchmark.xml"
        mythril["mythril"]["verbose"] = True
        mythril["mythril"]["postgres_path"] = args.pg_path

    if args.alternate:
        horizon = args.horizon
        per_query_timeout = args.query_timeout
    else:
        with open(f"{args.input}/stdout", "r") as f:
            config = f.readlines()[0]
            config = eval(config.split("HPO Configuration: ")[-1])
            horizon = config["horizon"]

        with open(f"{args.input}/stdout", "r") as f:
            for line in f:
                if "HPO Configuration: " in line:
                    hpo = eval(line.split("HPO Configuration: ")[-1].strip())
                    per_query_timeout = hpo["mythril_args"]["timeout"]

    folders = []
    start_found = False
    filename = "output.log" if args.alternate else "stderr"
    last_evaluation = None
    with open(f"{args.input}/{filename}", "r") as f:
        for line in f:
            if not start_found:
                if "Baseline Metric" in line:
                    start_time = parse(line.split("INFO:")[-1].split(" Baseline Metric")[0])
                    start_found = True
            else:
                if "mv" in line and "tuning_steps" in line:
                    repo = eval(line.split("Running ")[-1])[-1]
                    last_folder = repo.split("/")[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0])
                    last_evaluation = time_since_start
                    if (time_since_start - start_time).total_seconds() < args.cutoff * 3600 or args.cutoff == 0:
                        folders.append(last_folder)

    # Only apply threshold if time is less than.
    threshold_limit = last_evaluation - datetime.timedelta(seconds=int(args.threshold_limit * 3600))

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=horizon,
        config_path=f"{args.input}/config.yaml2",
        benchmark_config_path=f"{args.input}/{args.benchmark}.yaml",
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=horizon,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    if not args.simulated:
        env.restore_pristine_snapshot()
        env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
        spec.workload.reset()

    # Get the minimum reward.
    runs = [Path(args.input) / "tuning_steps" / fold / "run.raw.csv" for fold in folders]
    runs = [pd.read_csv(run) for run in runs]
    rewards = [(run["Latency (microseconds)"].sum() / 1e6, (run["Latency (microseconds)"].max() / 1e6) == per_query_timeout) for run in runs]
    rewards = sorted(rewards, key=lambda x: x[0])
    min_reward = min([r[0] for r in rewards])
    if maximal:
        target = [r[1] for r in rewards if r[0] == min_reward]
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
            logging.info(f"Maximal found: {min_reward}")

    num_lines = 0
    with open(f"{args.input}/{filename}", "r") as f:
        for line in f:
            if "Baseline Metric" in line:
                num_lines += 1
            elif "mv" in line and "tuning_steps" in line:
                num_lines += 1

    def run_sample(action, timeout):
        samples = []
        # This should reliably check that we are loading the correct knobs...
        ql_knobs = spec.action_space.get_knob_space().get_query_level_knobs(action) if action is not None else {}
        for i in range(args.samples):
            runtime = spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=timeout,
                ql_knobs=ql_knobs,
                env_spec=spec,
                blocklist=[l for l in args.blocklist.split(",") if len(l) > 0])
            samples.append(runtime)
            logging.info(f"Runtime: {runtime}")

            if runtime >= args.workload_timeout:
                break

            if args.samples == 2 and runtime >= timeout:
                break
            elif args.samples > 2 and len(samples) >= 2 and runtime >= timeout:
                break

        return samples

    run_data = []
    pbar = tqdm.tqdm(total=num_lines)
    with open(f"{args.input}/{filename}", "r") as f:
        current_step = 0

        start_found = False
        start_time = None
        timeout = args.workload_timeout
        cur_reward_max = timeout
        selected_action_knobs = None
        noop_index = False
        maximal_repo = None
        existing_indexes = []

        for line in f:
            # Keep going until we've found the start.
            if not start_found:
                if "Baseline Metric" in line:
                    start_found = True
                    start_time = parse(line.split("INFO:")[-1].split(" Baseline Metric")[0])
                    pbar.update(1)
                continue

            elif "Selected action: " in line:
                act = eval(line.split("Selected action: ")[-1])
                selected_action_knobs = env.action_space.get_knob_space().from_jsonable(act[0])[0]
                noop_index = "NOOP" in act[1][0]

            elif (maximal and ("mv" in line and "tuning_steps" in line)):
                maximal_repo = line

            elif (maximal and "Found new maximal state with" in line) or (not maximal and ("mv" in line and "tuning_steps" in line)):
                if "mv" in line and "tuning_steps" in line:
                    repo = eval(line.split("Running ")[-1])[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0])
                elif "Found new maximal state with" in line:
                    repo = eval(maximal_repo.split("Running ")[-1])[-1]
                    time_since_start = parse(maximal_repo.split("DEBUG:")[-1].split(" Running")[0])
                    maximal_repo = None

                # Get the evaluation reward.
                reward = pd.read_csv(f"{args.input}/{repo}/run.raw.csv")
                assert len(reward.columns) == 6
                has_timeout = (reward["Latency (microseconds)"].max() / 1e6) == per_query_timeout
                reward = reward["Latency (microseconds)"].sum() / 1e6
                assert reward > 0

                if ((not maximal_only and reward < cur_reward_max) or reward == min_reward) and (not maximal or not has_timeout):
                    index_sqls = []
                    knobs = {}
                    insert_knobs = False

                    with open(f"{args.input}/{repo}/act_sql.txt", "r") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) == 0:
                                insert_knobs = True
                            elif not insert_knobs:
                                index_sqls.append(line)
                            else:
                                k, v = line.split(" = ")
                                knobs[k] = float(v)

                    assert len(index_sqls) > 0
                    assert len(knobs) > 0
                    with open(f"{args.input}/{repo}/prior_state.txt", "r") as f:
                        prior_states = eval(f.read())
                        all_sc = [s.strip() for s in prior_states[1]]
                        if not noop_index:
                            all_sc.extend(index_sqls)

                        all_sc = [a for a in all_sc if not "USING btree ()" in a]
                        index_sqls = all_sc

                    execute_sqls = []
                    for index_sql in index_sqls:
                        if index_sql in existing_indexes:
                            continue
                        execute_sqls.append(index_sql)
                    for index_sql in existing_indexes:
                        if index_sql not in index_sqls:
                            indexname = index_sql.split("CREATE INDEX")[-1].split(" ON ")[0]
                            execute_sqls.append(f"DROP INDEX IF EXISTS {indexname}")

                    if not args.simulated:
                        # Reset snapshot.
                        env.action_space.reset(connection=env.connection, workload=env.workload)
                        cc, _ = env.action_space.get_knob_space().generate_plan(selected_action_knobs if selected_action_knobs else {})
                        env.shift_state(cc, execute_sqls, dump_page_cache=True)
                    existing_indexes = index_sqls

                    if not args.simulated:
                        # Get samples.
                        run_samples = samples = run_sample(knobs, timeout)
                        logging.info(f"Original Runtime: {reward} (timeout {has_timeout}). New Samples: {samples}")
                    else:
                        run_samples = samples = [reward, reward]

                    data = {
                        "step": current_step,
                        "orig_cost": reward,
                        "time_since_start": (time_since_start - start_time).total_seconds(),
                    }
                    samples = {f"runtime{i}": s for i, s in enumerate(samples)}
                    data.update(samples)
                    run_data.append(data)

                    current_step += 1

                    if (not has_timeout) or (max(run_samples) < timeout):
                        # Apply a tolerance..
                        # If we've timed out, only apply threshold only if we've found a strictly better config.
                        apply_threshold = threshold if time_since_start < threshold_limit else 0
                        cur_reward_max = reward - apply_threshold

                    if max(run_samples) < timeout:
                        timeout = max(run_samples)

                run_folder = repo.split("/")[-1]
                if run_folder in folders and run_folder == folders[-1]:
                    break
                elif maximal_only and reward == min_reward:
                    break
            pbar.update(1)

        if len(run_data) > 0:
            data = {
                "step": current_step,
                "orig_cost": run_data[-1]["orig_cost"],
                "time_since_start": -1,
                "runtime0": run_data[-1]["runtime0"],
            }
            run_data.append(data)

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()