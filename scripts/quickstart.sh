#!/bin/bash

set -euxo pipefail

DBMS=$1
BENCHMARK=$2
SCALE_FACTOR=$3
AGENT=$4

# Benchmark
python3 task.py benchmark $BENCHMARK data $SCALE_FACTOR
python3 task.py benchmark $BENCHMARK workload --scale-factor $SCALE_FACTOR

# DBMS
python3 task.py dbms $DBMS build
python3 task.py dbms $DBMS dbdata tpch --scale-factor $SCALE_FACTOR

# Tune
python3 task.py tune $AGENT embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" # long datagen so that train doesn't crash
python3 task.py tune $AGENT embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2
python3 task.py tune $AGENT agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01
python3 task.py tune $AGENT agent tune tpch --scale-factor $SCALE_FACTOR
