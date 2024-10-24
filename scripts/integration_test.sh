#!/bin/bash

set -euxo pipefail

DBMS=postgres
BENCHMARK=tpch
SCALE_FACTOR=0.01
AGENT=protox
INTENDED_DBDATA_HARDWARE="${1:-hdd}"

export DBGYM_CONFIG_PATH=scripts/integtest_dbgym_config.yaml

# Benchmark
python3 task.py benchmark $BENCHMARK data $SCALE_FACTOR
python3 task.py benchmark $BENCHMARK workload --scale-factor $SCALE_FACTOR

# DBMS
python3 task.py dbms $DBMS build
python3 task.py dbms $DBMS dbdata $BENCHMARK --scale-factor $SCALE_FACTOR --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE

# Tune
python3 task.py tune $AGENT embedding datagen $BENCHMARK --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE # long datagen so that train doesn't crash
python3 task.py tune $AGENT embedding train $BENCHMARK --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2
python3 task.py tune $AGENT agent hpo $BENCHMARK --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE
python3 task.py tune $AGENT agent tune $BENCHMARK --scale-factor $SCALE_FACTOR
python3 task.py tune $AGENT agent replay $BENCHMARK --scale-factor $SCALE_FACTOR
