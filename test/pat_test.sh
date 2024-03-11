#!/bin/bash
# /mnt/nvme1n1/phw2/noisepage/pg_ctl stop -D /mnt/nvme1n1/phw2/noisepage/pgdata &> /dev/null || true
# tar xf /home/wz2/mythril/data/tpch_sf10.tgz -C /mnt/nvme1n1/phw2/noisepage/
# /mnt/nvme1n1/phw2/noisepage/pg_ctl start -D /mnt/nvme1n1/phw2/noisepage/pgdata

# python task.py --no-startup-check tune protox embedding datagen tpch queries_15721_15723 --connection-str "host=localhost port=5432 dbname=benchbase user=admin" --override-sample-limits "lineitem,32768"
# python task.py --no-startup-check tune protox embedding train tpch queries_15721_15723 --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4 --num-points-to-sample 32 --max-segments 3

python task.py --no-startup-check tune protox agent train tpch queries_15721_15723