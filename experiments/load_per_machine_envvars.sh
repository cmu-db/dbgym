#!/bin/bash
host=$(hostname)

if [ "$host" == "dev4" ]; then
    export PGDATA_PARENT_DPATH=/mnt/nvme1n1/phw2/dbgym_tmp/
elif [ "$host" == "dev6" ]; then
    export PGDATA_PARENT_DPATH=/mnt/nvme0n1/phw2/dbgym_tmp/
else
    echo "Did not recognize host \"$host\""
    exit 1
fi