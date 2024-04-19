#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=0.1
INTENDED_PGDATA_HARDWARE=ssd
PGDATA_PARENT_DPATH=/mnt/nvme1n1/phw2/dbgym_tmp/

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
# python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 100 --query-timeout 15 --tune-duration-during-hpo 0.1  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH
# python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py --no-startup-check tune protox agent replay tpch --scale-factor $SCALE_FACTOR --workload-timeout-during-replay 10
exit 0

# benchmark
python3 task.py --no-startup-check benchmark tpch data $SCALE_FACTOR
python3 task.py --no-startup-check benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH

exit 0

# embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH # long datagen so that train doesn't crash
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor $SCALE_FACTOR --iterations-per-epoch 1 --num-points-to-sample 1 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --num-samples 2 --max-concurrent 2 --workload-timeout 100 --query-timeout 15 --tune-duration-during-hpo 1  --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py --no-startup-check tune protox agent replay tpch --scale-factor $SCALE_FACTOR
