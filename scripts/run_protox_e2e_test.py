import os
import shutil
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path

import yaml

from benchmark.constants import DEFAULT_SCALE_FACTOR
from benchmark.tpch.constants import DEFAULT_TPCH_SEED
from util.pg import get_is_postgres_running
from util.workspace import (
    default_embedder_path,
    default_hpoed_agent_params_path,
    default_pristine_dbdata_snapshot_path,
    default_replay_data_fpath,
    default_repo_path,
    default_tables_path,
    default_traindata_path,
    default_tuning_steps_dpath,
    default_workload_path,
    get_workload_name,
)

# Be careful when changing these constants. In some places, the E2E test is hardcoded to work for these specific constants.
DBMS = "postgres"
AGENT = "protox"
E2ETEST_DBGYM_CONFIG_FPATH = Path("scripts/e2e_test_dbgym_config.yaml")


def get_workspace_dpath(config_fpath: Path) -> Path:
    with open(config_fpath, "r") as file:
        config = yaml.safe_load(file)
    return Path(config.get("dbgym_workspace_path"))


def clear_workspace(workspace_dpath: Path) -> None:
    actual_workspace_dpath = Path("../dbgym_workspace")
    if workspace_dpath.exists():
        if actual_workspace_dpath.exists():
            assert not workspace_dpath.samefile(
                actual_workspace_dpath
            ), "YOU MAY BE ABOUT TO DELETE YOUR ACTUAL WORKSPACE"
        shutil.rmtree(workspace_dpath)


class Stage(Enum):
    Tables = auto()
    Workload = auto()
    DBRepo = auto()
    DBData = auto()
    EmbeddingData = auto()
    EmbeddingModel = auto()
    TuneHPO = auto()
    TuneTune = auto()
    Replay = auto()


# When debugging the E2E test, this gives you an easy way of turning off certain stages to speed up your iteration cycle.
#
# I made this slightly convoluted system is because you can't just naively comment out a big chunk of code with all the stages
# you don't want to run. Many stages define variables that are used by future stages, which can't be commented out.
#
# One useful debugging workflow is to run all stages up until a point, make a copy of that workspace, and then rerun the
# integration test as many times as you want starting from that copy.
ALL_STAGES = {stage for stage in Stage}
# This is a set and not a list because the order of stages is already pre-defined. This just defines what not to skip.
STAGES_TO_RUN = ALL_STAGES


def run_e2e_for_benchmark(benchmark_name: str, intended_dbdata_hardware: str) -> None:
    if benchmark_name == "tpch":
        scale_factor = 0.01
        query_subset = "all"
        workload_name_suffix = f"{DEFAULT_TPCH_SEED}_{DEFAULT_TPCH_SEED}_{query_subset}"
        embedding_datagen_args = "--override-sample-limits lineitem,32768"
        embedding_train_args = "--iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2"
        tune_hpo_args = "--num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01"
    elif benchmark_name == "job":
        scale_factor = DEFAULT_SCALE_FACTOR
        query_subset = "demo"
        workload_name_suffix = query_subset
        embedding_datagen_args = ""
        embedding_train_args = "--iterations-per-epoch 1 --num-points-to-sample 2 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2"
        tune_hpo_args = "--num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 2 --tune-duration-during-hpo 0.03"
    else:
        assert False

    # Clear the E2E testing workspace so we always run the test with a clean slate.
    workspace_dpath = get_workspace_dpath(E2ETEST_DBGYM_CONFIG_FPATH)
    clear_workspace(workspace_dpath)

    # Make other checks that we have a clean slate for testing.
    assert not get_is_postgres_running()

    # Run the full Proto-X training pipeline, asserting things along the way
    # Setup (workload and database)
    tables_dpath = default_tables_path(workspace_dpath, benchmark_name, scale_factor)
    if Stage.Tables in STAGES_TO_RUN:
        assert not tables_dpath.exists()
        subprocess.run(
            f"python task.py benchmark {benchmark_name} data {scale_factor}".split(),
            check=True,
        )
        assert tables_dpath.exists()

    workload_name = get_workload_name(scale_factor, workload_name_suffix)
    workload_dpath = default_workload_path(
        workspace_dpath, benchmark_name, workload_name
    )
    if Stage.Workload in STAGES_TO_RUN:
        assert not workload_dpath.exists()
        subprocess.run(
            f"python task.py benchmark {benchmark_name} workload --query-subset {query_subset} --scale-factor {scale_factor}".split(),
            check=True,
        )
        assert workload_dpath.exists()

    repo_dpath = default_repo_path(workspace_dpath)
    if Stage.DBRepo in STAGES_TO_RUN:
        assert not repo_dpath.exists()
        subprocess.run(f"python task.py dbms {DBMS} build".split(), check=True)
        assert repo_dpath.exists()

    pristine_dbdata_snapshot_fpath = default_pristine_dbdata_snapshot_path(
        workspace_dpath, benchmark_name, scale_factor
    )
    if Stage.DBData in STAGES_TO_RUN:
        assert not pristine_dbdata_snapshot_fpath.exists()
        subprocess.run(
            f"python task.py dbms {DBMS} dbdata {benchmark_name} --scale-factor {scale_factor} --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
            check=True,
        )
        assert pristine_dbdata_snapshot_fpath.exists()

    # Tuning (embedding, HPO, and actual tuning)
    traindata_dpath = default_traindata_path(
        workspace_dpath, benchmark_name, workload_name
    )
    if Stage.EmbeddingData in STAGES_TO_RUN:
        assert not traindata_dpath.exists()
        subprocess.run(
            f"python task.py tune {AGENT} embedding datagen {benchmark_name} --workload-name-suffix {workload_name_suffix} --scale-factor {scale_factor} {embedding_datagen_args} --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
            check=True,
        )
        assert traindata_dpath.exists()

    embedder_dpath = default_embedder_path(
        workspace_dpath, benchmark_name, workload_name
    )
    if Stage.EmbeddingModel in STAGES_TO_RUN:
        assert not embedder_dpath.exists()
        subprocess.run(
            f"python task.py tune {AGENT} embedding train {benchmark_name} --workload-name-suffix {workload_name_suffix} --scale-factor {scale_factor} {embedding_train_args}".split(),
            check=True,
        )
        assert embedder_dpath.exists()

    hpoed_agent_params_fpath = default_hpoed_agent_params_path(
        workspace_dpath, benchmark_name, workload_name
    )
    if Stage.TuneHPO in STAGES_TO_RUN:
        assert not hpoed_agent_params_fpath.exists()
        subprocess.run(
            f"python task.py tune {AGENT} agent hpo {benchmark_name} --workload-name-suffix {workload_name_suffix} --scale-factor {scale_factor} {tune_hpo_args} --intended-dbdata-hardware {intended_dbdata_hardware}".split(),
            check=True,
        )
        assert hpoed_agent_params_fpath.exists()

    tuning_steps_dpath = default_tuning_steps_dpath(
        workspace_dpath, benchmark_name, workload_name, False
    )
    if Stage.TuneTune in STAGES_TO_RUN:
        assert not tuning_steps_dpath.exists()
        subprocess.run(
            f"python task.py tune {AGENT} agent tune {benchmark_name} --workload-name-suffix {workload_name_suffix} --scale-factor {scale_factor}".split(),
            check=True,
        )
        assert tuning_steps_dpath.exists()

    # Post-training (replay)
    replay_data_fpath = default_replay_data_fpath(
        workspace_dpath, benchmark_name, workload_name, False
    )
    if Stage.Replay in STAGES_TO_RUN:
        assert not replay_data_fpath.exists()
        subprocess.run(
            f"python3 task.py tune {AGENT} agent replay {benchmark_name} --workload-name-suffix {workload_name_suffix} --scale-factor {scale_factor}".split(),
            check=True,
        )
        assert replay_data_fpath.exists()

    # Clear it at the end as well to avoid leaving artifacts.
    clear_workspace(workspace_dpath)


if __name__ == "__main__":
    intended_dbdata_hardware = sys.argv[1] if len(sys.argv) > 1 else "hdd"

    # Set the config file so that we use resources that don't conflict with normal usage (e.g. a different workspace, different ports, etc.).
    os.environ["DBGYM_CONFIG_PATH"] = str(E2ETEST_DBGYM_CONFIG_FPATH)

    run_e2e_for_benchmark("tpch", intended_dbdata_hardware)
    run_e2e_for_benchmark("job", intended_dbdata_hardware)
