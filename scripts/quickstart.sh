#!/bin/bash

set -euxo pipefail

DBMS=$1
PGDATA_PARENT_DPATH=$2
BENCHMARK=$3
SCALE_FACTOR=$4
AGENT=$5
INTENDED_PGDATA_HARDWARE=ssd

# Benchmark
python3 task.py benchmark $BENCHMARK data $SCALE_FACTOR
python3 task.py benchmark $BENCHMARK workload --scale-factor $SCALE_FACTOR

# DBMS
python3 task.py dbms $DBMS build
python3 task.py dbms $DBMS pgdata tpch --scale-factor $SCALE_FACTOR --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH

# Tune
python3 task.py tune $AGENT embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH # long datagen so that train doesn't crash
python3 task.py tune $AGENT embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2
python3 task.py tune $AGENT agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH --build-space-good-for-boot
python3 task.py tune $AGENT agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py tune $AGENT agent replay tpch --scale-factor $SCALE_FACTOR
