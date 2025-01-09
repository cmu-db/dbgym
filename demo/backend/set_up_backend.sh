#!/bin/bash
set -euxo pipefail

# Run this from dbgym/.
python task.py benchmark job tables 1
python task.py benchmark job workload --query-subset demo
python task.py dbms postgres build
python task.py dbms postgres dbdata job