#!/bin/bash
host=$(hostname)

if [ "$host" == "dev4" ]; then
    export DBDATA_PARENT_PATH=/mnt/nvme1n1/phw2/dbgym_tmp/
    export INTENDED_DBDATA_HARDWARE=ssd
elif [ "$host" == "dev6" ]; then
    export DBDATA_PARENT_PATH=/mnt/nvme0n1/phw2/dbgym_tmp/
    export INTENDED_DBDATA_HARDWARE=ssd
elif [ "$host" == "patnuc" ]; then
    export DBDATA_PARENT_PATH=../dbgym_workspace/tmp/
    export INTENDED_DBDATA_HARDWARE=hdd
else
    echo "Did not recognize host \"$host\""
    exit 1
fi