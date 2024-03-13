#!/bin/bash

set -euxo pipefail

# # Setup DBMS.
# python3 task.py --no-startup-check dbms postgres clone
# python3 task.py --no-startup-check dbms postgres init-pgdata --remove-existing
# python3 task.py --no-startup-check dbms postgres start
# python3 task.py --no-startup-check dbms postgres init-auth
# python3 task.py --no-startup-check dbms postgres run-sql-file ./config/pgtune.sql
# python3 task.py --no-startup-check dbms postgres run-sql-file ./config/setup.sql
# python3 task.py --no-startup-check dbms postgres stop

# # Generate TPC-H.
# python3 task.py --no-startup-check benchmark tpch generate-sf 1
# python3 task.py --no-startup-check benchmark tpch generate-workload queries_15721_15723 15721 15723

# # Load TPC-H.
# python3 task.py --no-startup-check dbms postgres start
# python3 task.py --no-startup-check dbms postgres init-db tpch_sf1
# python3 task.py --no-startup-check benchmark tpch load-sf 1 postgres tpch_sf1
# python3 task.py --no-startup-check dbms postgres stop

# Make it a .tgz file
# TODO(phw2): do this programmatically instead
pgdata_path=$HOME/dbgym_workspace/symlinks/dbgym_dbms_postgres/build/repo/boot/build/postgres/bin/pgdata
# -C "$pgdata_path" changes the current directory to $pgdata_path before creating the tar archive
# this way, the .tgz file archives the files in pgdata5435 "directly" without all the preceding directories
tar -czf $HOME/tpch_sf1.tgz -C $pgdata_path .
