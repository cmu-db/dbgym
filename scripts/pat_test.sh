#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=0.01

# benchmark
python3 task.py --no-startup-check benchmark tpch data $SCALE_FACTOR
python3 task.py --no-startup-check benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR

# embedding
# python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --default-sample-limit 64 --file-limit 64 # short datagen for testing
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" # long datagen so that train doesn't crash
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --duration 0.01
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
