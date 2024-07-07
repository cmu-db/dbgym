#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=0.1
INTENDED_PGDATA_HARDWARE=ssd
. ./experiments/load_per_machine_envvars.sh
echo $PGDATA_PARENT_DPATH

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
# python3 task.py tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 4 --max-concurrent 4 --workload-timeout 100 --query-timeout 15 --tune-duration-during-hpo 0.1  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH
python3 task.py tune protox agent tune tpch --scale-factor $SCALE_FACTOR --tune-duration-during-tune 0.2
python3 task.py tune protox agent replay tpch --scale-factor $SCALE_FACTOR
exit 0

# benchmark
python3 task.py benchmark tpch data $SCALE_FACTOR
python3 task.py benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py dbms postgres build
python3 task.py dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH

exit 0

# embedding
python3 task.py tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH # long datagen so that train doesn't crash
python3 task.py tune protox embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 4 --max-concurrent 4 --workload-timeout 100 --query-timeout 15 --tune-duration-during-hpo 1  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH --build-space-good-for-boot
python3 task.py tune protox agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py tune protox agent replay tpch --scale-factor $SCALE_FACTOR
