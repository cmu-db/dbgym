#!/bin/bash

set -euxo pipefail

# Setup DBMS.
python3 task.py --no-startup-check dbms postgres clone
python3 task.py --no-startup-check dbms postgres init-pgdata --remove-existing
python3 task.py --no-startup-check dbms postgres start
python3 task.py --no-startup-check dbms postgres init-auth
python3 task.py --no-startup-check dbms postgres run-sql-file ./config/pgtune.sql
python3 task.py --no-startup-check dbms postgres run-sql-file ./config/setup.sql
python3 task.py --no-startup-check dbms postgres stop

# Generate TPC-H.
python3 task.py --no-startup-check benchmark tpch generate-sf 1
python3 task.py --no-startup-check benchmark tpch generate-workload queries_15721_15723 15721 15723

# Load TPC-H.
python3 task.py --no-startup-check dbms postgres start
python3 task.py --no-startup-check dbms postgres init-db tpch_sf1
python3 task.py --no-startup-check benchmark tpch load-sf 1 postgres tpch_sf1
python3 task.py --no-startup-check dbms postgres stop

# Proto-X.
python3 task.py --no-startup-check dbms postgres start
python3 task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --connection-str "host=localhost port=15721 dbname=tpch_sf1 user=noisepage_user password=noisepage_pass" --override-sample-limits "lineitem,32768"
python3 task.py --no-startup-check tune protox embedding train tpch queries_15721_15723 --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4 --num-points-to-sample 32 --max-segments 3
python3 task.py --no-startup-check dbms postgres stop
