#!/bin/bash

set -euxo pipefail

# benchmark
python3 task.py --no-startup-check benchmark tpch generate-data 0.01
python3 task.py --no-startup-check benchmark tpch generate-workload queries_15721_15723 15721 15723

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor 0.01

# embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --override-sample-limits "lineitem,32768"

# tune
