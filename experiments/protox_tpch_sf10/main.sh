#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=10

# benchmark
python3 task.py --no-startup-check benchmark tpch generate-data $SCALE_FACTOR
python3 task.py --no-startup-check benchmark tpch generate-workload queries_15721_15723 15721 15723

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR

# embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --override-sample-limits "lineitem,32768"
python3 task.py --no-startup-check tune protox embedding train tpch queries_15721_15723 --scale-factor $SCALE_FACTOR --scale-factor 10 --train-max-concurrent 10

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch queries_15721_15723 --scale-factor $SCALE_FACTOR --max-concurrent 4
python3 task.py --no-startup-check tune protox agent tune tpch queries_15721_15723 --scale-factor $SCALE_FACTOR
