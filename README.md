# 🛢️ Database Gym 🏋️
[\[Slides\]](http://www.cidrdb.org/cidr2023/slides/p27-lim-slides.pdf) [\[Paper\]](https://www.cidrdb.org/cidr2023/papers/p27-lim.pdf)

*An end-to-end research vehicle for self-driving databases.*

## Quickstart

These steps were tested on a fresh repository clone, Ubuntu 22.04.

```
# Setup dependencies.
# You may want to create a Python 3.10 virtual environment (e.g. with conda) before doing this.
./dependency/install_dependencies.sh

# Compile a custom fork of PostgreSQL, load TPC-H (SF 0.01), train the Proto-X agent, and tune.
./scripts/quickstart.sh postgres tpch 0.01 protox
```

## Overview

Autonomous DBMS research often involves more engineering than research.
As new advances in state-of-the-art technology are made, it is common to find that they have
reimplemented the database tuning pipeline from scratch: workload capture, database setup,
training data collection, model creation, model deployment, and more.
Moreover, these bespoke pipelines make it difficult to combine different techniques even when they
should be independent (e.g., using a different operator latency model in a tuning algorithm).

The database gym project is our attempt at standardizing the APIs between these disparate tasks,
allowing researchers to mix-and-match the different pipeline components.
It draws inspiration from the Farama Foundation's Gymnasium (formerly OpenAI Gym), which
accelerates the development and comparison of reinforcement learning algorithms by providing a set
of agents, environments, and a standardized API for communicating between them.
Through the database gym, we hope to save other people time and reimplementation effort by
providing an extensible open-source platform for autonomous DBMS research.

This project is under active development.
Currently, we decompose the database tuning pipeline into the following components:

1. Workload: collection, forecasting, synthesis
2. Database: database loading, instrumentation, orchestrating workload execution
3. Agent: identifying tuning actions, suggesting an action

## Repository Structure

`task.py` is the entrypoint for all tasks.
The tasks are grouped into categories that correspond to the top-level directories of the repository:

- `benchmark` - tasks to generate data and queries for different benchmarks (e.g., TPC-H, JOB)
- `dbms` - tasks to build and start DBMSs (e.g., PostgreSQL)

## Credits

The Database Gym project rose from the ashes of the [NoisePage](https://db.cs.cmu.edu/projects/noisepage/) self-driving DBMS project.

The first prototype was written by [Patrick Wang](https://github.com/wangpatrick57), integrating [Boot (VLDB 2024)](https://github.com/lmwnshn/boot) and [Proto-X (VLDB 2024)](https://github.com/17zhangw/protox) into a cohesive system.

## Citing This Repository

If you use this repository in an academic paper, please cite one or more of the following based on your usage:

### Reference Implementation (`dbgym`)
```
@inproceedings{wang25demo,
  author = {Wang, Patrick and Lim, Wan Shen and Zhang, William and Arch, Samuel and Pavlo, Andrew},
  title = {Automated Database Tuning vs. Human-Based Tuning in a Simulated Stressful Work Environment: A Demonstration of the Database Gym},
  year = {2025},
  doi = {10.1145/3722212.3725083},
  booktitle = {Companion of the 2025 International Conference on Management of Data},
  pages = {247--250},
  numpages = {4},
  series = {SIGMOD/PODS '25},
  url = {https://db.cs.cmu.edu/papers/2025/wang-sigmoddemo2025.pdf},
}
```

### General Idea (Database Gyms)
```
@inproceedings{lim23,
  author = {Lim, Wan Shen and Butrovich, Matthew and Zhang, William and Crotty, Andrew and Ma, Lin and Xu, Peijing and Gehrke, Johannes and Pavlo, Andrew},
  title = {Database Gyms},
  booktitle = {{CIDR} 2023, Conference on Innovative Data Systems Research},
  year = {2023},
  url = {https://db.cs.cmu.edu/papers/2023/p27-lim.pdf},
 }
```

### Accelerating Training Data Generation
```
@article{lim24boot,
  author = {Lim, Wan Shen and Ma, Lin and Zhang, William and Butrovich, Matthew and Arch, Samuel I and Pavlo, Andrew},
  title = {Hit the Gym: Accelerating Query Execution to Efficiently Bootstrap Behavior Models for Self-Driving Database Management Systems},
  journal = {Proc. {VLDB} Endow.},
  volume = {17},
  number = {11},
  pages = {3680--3693},
  year = {2024},
  url = {https://www.vldb.org/pvldb/vol17/p3680-lim.pdf},
}
```

### Simultaneously Tuning Multiple Configuration Spaces with Proto Actions
```
@article{zhang24holon,
  author = {Zhang, William and Lim, Wan Shen and Butrovich, Matthew and Pavlo, Andrew},
  title = {The Holon Approach for Simultaneously Tuning Multiple Components in a Self-Driving Database Management System with Machine Learning via Synthesized Proto-Actions},
  journal = {Proc. {VLDB} Endow.},
  volume = {17},
  number = {11},
  pages = {3373--3387},
  year = {2024},
  url = {https://www.vldb.org/pvldb/vol17/p3373-zhang.pdf},
}
```
