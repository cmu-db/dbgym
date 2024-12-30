#!/bin/bash

set -euxo pipefail

. ./scripts/_load_per_machine_envvars.sh

# space for testing. uncomment this to run individual commands from the script (copy pasting is harder because there are envvars)
exit 0

# benchmark
python3 task.py benchmark job data
python3 task.py benchmark job workload --query-subset demo

# postgres
python3 task.py dbms postgres build
python3 task.py dbms postgres dbdata job --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE --dbdata-parent-path $DBDATA_PARENT_PATH