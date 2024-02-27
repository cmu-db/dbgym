#!/bin/bash
python task.py --no-startup-check protox embedding train tpch --iterations-per-epoch 1 --num-samples 6 --train-max-concurrent 6 --num-points-to-sample 64