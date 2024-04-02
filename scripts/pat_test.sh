#!/bin/bash

set -euxo pipefail

# benchmark
python3 task.py --no-startup-check benchmark tpch data 0.01
python3 task.py --no-startup-check benchmark tpch workload --scale-factor 0.01

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor 0.01

# embedding
# python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor 0.01 --default-sample-limit 64 --file-limit 64 # short datagen for testing
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor 0.01 --override-sample-limits "lineitem,32768" --scale-factor 0.01 # long datagen so that train doesn't crash
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor 0.01 --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor 0.01 --num-samples 2 --max-concurrent 2 --duration 0.001
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor 0.01
