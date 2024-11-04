#!/bin/bash

set -euxo pipefail

. ./experiments/load_per_machine_envvars.sh

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
python3 task.py tune protox embedding datagen job --workload-name-suffix a --default-sample-limit 64 --file-limit 64 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH # short datagen for testing
python3 task.py tune protox embedding train job --workload-name-suffix a --iterations-per-epoch 1 --num-points-to-sample 2 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2
exit 0

# benchmark
python3 task.py benchmark job data
python3 task.py benchmark job workload --query-subset a

# postgres
python3 task.py dbms postgres build
python3 task.py dbms postgres dbdata job --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH

# embedding
python3 task.py tune protox embedding datagen job --workload-name-suffix a --default-sample-limit 64 --file-limit 64 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH # short datagen for testing
python3 task.py tune protox embedding train job --workload-name-suffix a --iterations-per-epoch 1 --num-points-to-sample 2 --num-batches 1 --batch-size 64 --start-epoch 15 --num-samples 4 --train-max-concurrent 4 --num-curate 2

# agent
python3 task.py tune protox agent hpo job --workload-name-suffix a --num-samples 2 --max-concurrent 2 --workload-timeout 15 --query-timeout 1 --tune-duration-during-hpo 0.01  --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH --build-space-good-for-boot
python3 task.py tune protox agent tune job --workload-name-suffix a
python3 task.py tune protox agent replay job --workload-name-suffix a
