#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=0.01
INTENDED_DBDATA_HARDWARE=ssd
. ./experiments/load_per_machine_envvars.sh

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
python3 task.py tune protox agent replay tpch --scale-factor $SCALE_FACTOR
exit 0

# benchmark
python3 task.py benchmark tpch data $SCALE_FACTOR
python3 task.py benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py dbms postgres build
python3 task.py dbms postgres dbdata tpch --scale-factor $SCALE_FACTOR --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH

# embedding
# python3 task.py tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --default-sample-limit 64 --file-limit 64 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH # short datagen for testing
python3 task.py tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH # long datagen so that train doesn't crash
python3 task.py tune protox embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01  --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH --build-space-good-for-boot
python3 task.py tune protox agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py tune protox agent replay tpch --scale-factor $SCALE_FACTOR
