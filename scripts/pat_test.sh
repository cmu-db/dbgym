#!/bin/bash

set -euxo pipefail

# benchmark
python3 task.py --no-startup-check benchmark tpch generate-data 0.01
python3 task.py --no-startup-check benchmark tpch generate-workload queries_15721_15723 15721 15723

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor 0.01

# embedding
# python3 task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --scale-factor 0.01 --default-sample-limit 64 --file-limit 64 # short datagen for testing
python3 task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --override-sample-limits "lineitem,32768" --scale-factor 0.01 # long datagen so that train doesn't crash
exit 0
python3 task.py --no-startup-check tune protox embedding train tpch queries_15721_15723 --scale-factor 0.01 --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch queries_15721_15723 --scale-factor 0.01
