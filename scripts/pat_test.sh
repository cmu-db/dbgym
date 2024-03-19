#!/bin/bash

set -euxo pipefail

# generate pgdata.tgz
python3 task.py --no-startup-check dbms postgres build
python3 task.py --no-startup-check benchmark tpch generate-data 0.01
python3 task.py --no-startup-check dbms postgres pgdata tpch --scale-factor 0.01

# start postgres with pgdata.tgz
PGDATA_TGZ_SYMLINK_FPATH="$HOME/dbgym_workspace/symlinks/dbgym_dbms_postgres/data/tpch_sf0point01_pgdata.tgz"
PGDATA_REAL_DPATH="$HOME/tpch_sf0point01_pgdata"
PGBIN_SYMLINK_DPATH="$HOME/dbgym_workspace/symlinks/dbgym_dbms_postgres/build/repo/boot/build/postgres/bin"
PGBIN_REAL_DPATH=`realpath $PGBIN_SYMLINK_DPATH`
rm -rf "$PGDATA_REAL_DPATH"
mkdir "$PGDATA_REAL_DPATH"
cd "$PGDATA_REAL_DPATH"
tar -xzf "$PGDATA_TGZ_SYMLINK_FPATH"
cd -
"$PGBIN_REAL_DPATH/pg_ctl" -D "$PGDATA_REAL_DPATH" start
"$PGBIN_REAL_DPATH/psql" -d "dbgym"