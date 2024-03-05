#!/bin/bash
# /mnt/nvme1n1/phw2/noisepage/pg_ctl stop -D /mnt/nvme1n1/phw2/noisepage/pgdata &> /dev/null || true
# tar xf /home/wz2/mythril/data/tpch_sf10.tgz -C /mnt/nvme1n1/phw2/noisepage/
# /mnt/nvme1n1/phw2/noisepage/pg_ctl start -D /mnt/nvme1n1/phw2/noisepage/pgdata
python task.py --no-startup-check protox embedding datagen tpch --connection-str "host=localhost port=5432 dbname=benchbase user=admin" --override-sample-limits "lineitem,32768"
python task.py --no-startup-check protox embedding train tpch --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4 --num-points-to-sample 32 --max-segments 3