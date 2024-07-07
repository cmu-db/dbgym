#!/bin/bash

set -euxo pipefail

# Build Postgres
python3 task.py dbms postgres repo

# Generate TPC-H
python3 task.py benchmark tpch generate-data 1
python3 task.py benchmark tpch generate-workload queries_15721_15723 15721 15723

# Create tpch_sf1.tgz
python3 task.py dbms postgres pgdata tpch --scale-factor 1

# Run Proto-X
python3 task.py dbms postgres start
python3 task.py tune protox embedding datagen tpch queries_15721_15723 --connection-str "host=localhost port=15721 dbname=tpch_sf1 user=noisepage_user password=noisepage_pass" --override-sample-limits "lineitem,32768"
python3 task.py tune protox embedding train tpch queries_15721_15723 --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4 --num-points-to-sample 32 --max-segments 3
python3 task.py dbms postgres stop
