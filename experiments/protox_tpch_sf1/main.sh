#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=1
INTENDED_DBDATA_HARDWARE=ssd
. ./experiments/load_per_machine_envvars.sh

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
python3 task.py tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --max-concurrent 4 --tune-duration-during-hpo 1 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH --build-space-good-for-boot
exit 0

# benchmark
python3 task.py benchmark tpch data $SCALE_FACTOR
python3 task.py benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py dbms postgres build
python3 task.py dbms postgres dbdata tpch --scale-factor $SCALE_FACTOR --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH

# embedding
python3 task.py tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH
python3 task.py tune protox embedding train tpch --scale-factor $SCALE_FACTOR --train-max-concurrent 10

# agent
python3 task.py tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --max-concurrent 4 --tune-duration-during-hpo 4 --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-dpath $DBDATA_PARENT_DPATH --build-space-good-for-boot
python3 task.py tune protox agent tune tpch --scale-factor $SCALE_FACTOR
