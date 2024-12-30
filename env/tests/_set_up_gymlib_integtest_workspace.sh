#!/bin/bash

# DO NOT RUN THIS SCRIPT DIRECTLY.
# This script only runs correctly when run by GymlibIntegtestManager.set_up_workspace() as it sets the necessary envvars.
# By allowing GymlibIntegtestManager.set_up_workspace() to set the envvars, we ensure that the envvars are only defined
# in a single location (inside GymlibIntegtestManager).

# Gymlib integration tests relies on Postgres being built and workloads/dbdata being generated.
# Generating these things is not considered a part of the test which is why it's in its own shell script.
# The reason there's a shell script generating them instead of them just being in the repo is because (a)
#   the Postgres repo is very large and (b) the built binary will be different for different machines.
# This script should be run from the base dbgym/ directory.

set -euxo pipefail

# INTENDED_DBDATA_HARDWARE can be set elsewhere (e.g. by tests_ci.yaml) but we use hdd by default.
INTENDED_DBDATA_HARDWARE="${INTENDED_DBDATA_HARDWARE:-hdd}"

python3 task.py benchmark $BENCHMARK tables $SCALE_FACTOR
python3 task.py benchmark $BENCHMARK workload --scale-factor $SCALE_FACTOR

python3 task.py dbms postgres build
python3 task.py dbms postgres dbdata $BENCHMARK --scale-factor $SCALE_FACTOR --intended-dbdata-hardware $INTENDED_DBDATA_HARDWARE
