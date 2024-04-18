#!/bin/bash

set -euxo pipefail

SCALE_FACTOR=10
INTENDED_PGDATA_HARDWARE=ssd
PGDATA_PARENT_DPATH=/mnt/nvme1n1/phw2/dbgym_tmp/

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR --enable-boot-during-tune
python3 task.py --no-startup-check tune protox agent replay tpch --scale-factor $SCALE_FACTOR
python3 task.py --no-startup-check tune protox agent replay tpch --scale-factor $SCALE_FACTOR --boot-enabled-during-tune
exit 0

# benchmark
python3 task.py --no-startup-check benchmark tpch data $SCALE_FACTOR
python3 task.py --no-startup-check benchmark tpch workload --scale-factor $SCALE_FACTOR

# postgres
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor $SCALE_FACTOR --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH

# embedding
python3 task.py --no-startup-check tune protox embedding datagen tpch --scale-factor $SCALE_FACTOR --override-sample-limits "lineitem,32768" --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH
python3 task.py --no-startup-check tune protox embedding train tpch --scale-factor $SCALE_FACTOR --train-max-concurrent 10

# agent
python3 task.py --no-startup-check tune protox agent hpo tpch --scale-factor $SCALE_FACTOR --max-concurrent 4 --duration 4 --intended-pgdata-hardware $INTENDED_PGDATA_HARDWARE --pgdata-parent-dpath $PGDATA_PARENT_DPATH --enable-boot-during-hpo
python3 task.py --no-startup-check tune protox agent tune tpch --scale-factor $SCALE_FACTOR
