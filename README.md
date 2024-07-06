# 🛢️ Database Gym 🏋🏻‍♂️
[\[Slides\]](http://www.cidrdb.org/cidr2023/slides/p27-lim-slides.pdf) [\[Paper\]](https://www.cidrdb.org/cidr2023/papers/p27-lim.pdf)

*An end-to-end system which trains RL agents to tune databases.*

## Overview
The Database Gym (DBGym) is a research project from the CMU Database Group (CMU-DB) in the field of automated database tuning via reinforcement learning (RL). **Tuning** a database means selecting a configuration ([1] indexes, [2] system-wide knobs, and [3] per-query knobs) that optimizes the throughput or latency of a **benchmark** (a set of queries over a collection of data).

RL involves an **agent** performing **actions** against an **environment** which then gives **rewards** and **observations** back to the agent. When applying RL to database tuning, the actions are modifications to the database's configuration, the rewards are the throughput or latency (possibly estimated) of the benchmark under the new configuration. The observations may be things such as the current configuration or various execution statistics from running the benchmark.

In this workflow, a key challenge is in gathering the rewards and observations. Naively, this involves executing the full benchmark after every tuning action, but this introduces significant overhead. The DBGym project researches two directions to mitigate this overhead: [1] **approximating** the rewards and observations without executing the full benchmark and [2] extrapolating the information (rewards and observations) received about a single configuration to learn about **multiple configurations**. The first direction corresponds to the *environment* while the second direction corresponds to the *agent*, which is why DBGym is an **end-to-end** system.

## How to Run
The repository is designed as a vehicle for database tuning research in which the researcher performs a variety of **tasks**. `task.py` is the entrypoint for all tasks. The tasks are grouped into categories that correspond to the top-level directories of the repository:
* `benchmark` - tasks to generate data and queries for different benchmarks (e.g. TPC-H or TPC-C).
* `dbms` - tasks to build and start DBMSs (e.g. Postgres). As DBGym is an end-to-end system, we generally use custom forks of DBMSs.
* `tune` - tasks to train an RL agent to tune a live DBMS.

Below is an example sequence of commands which generate the TPC-H benchmark, build and run a custom fork of Postgres, and train the Proto-X tuning agent. It has been tested to work from a fresh repository clone on a Linux machine.
```
## Dependencies
# Python
pip install -r dependency/requirements.sh

# Linux
cat dependency/apt_requirements.txt | xargs sudo apt-get install -y

# Install Rust
./dependency/rust.sh


## Environment Variables
# SF 0.01 is a quick way to get started
SCALE_FACTOR=0.01

# You can use "hdd" instead
INTENDED_PGDATA_HARDWARE=ssd

# You can make this a path to a dir in HDD instead
PGDATA_PARENT_DPATH=/path/to/dir/in/ssd/mount


## Benchmark
# Generate data
python3 task.py --no-startup-check benchmark tpch data $SCALE_FACTOR

# Generate queries
python3 task.py --no-startup-check benchmark tpch workload --scale-factor $SCALE_FACTOR


## DBMS
# Build Postgres
python3 task.py --no-startup-check dbms postgres build

# Load TPC-H into Postgres
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH


## Tune
# Generate training data for Proto-X's index embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH

# Train Proto-X's index embedding model
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# Search for hyperparameters to train the Proto-X agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --duration 0.1  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH --enable-boot-during-hpo

# Train the Proto-X agent
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
```

## Cite
If you use this repository in an academic paper, please cite:
```
@inproceedings{Lim2023DatabaseG,
  title={Database Gyms},
  author={Wan Shen Lim and Matthew Butrovich and William Zhang and Andrew Crotty and Lin Ma and Peijing Xu and Johannes Gehrke and Andrew Pavlo},
  booktitle={Conference on Innovative Data Systems Research},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:259186019}
}
```