#!/bin/bash
python task.py --no-startup-check protox embedding datagen tpch --connection-str "host=localhost port=5432 dbname=benchbase user=admin"
# python task.py --no-startup-check protox embedding train tpch --iterations-per-epoch 1 --num-samples 4 --train-max-concurrent 4 --num-points-to-sample 32 --max-segments 3