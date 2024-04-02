#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=10

# benchmark
python3 task.py --no-startup-check benchmark tpch data $SCALE_FACTOR
python3 task.py --no-startup-check benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR

# embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768"
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor $SCALE_FACTOR --scale-factor 10 --train-max-concurrent 10

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --max-concurrent 4
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
