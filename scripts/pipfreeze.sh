#!/bin/bash
# Ignore gymlib because we install it manually inside _build_conda_env.sh (not from requirements.txt).
pip freeze | grep -v "^gymlib @" >scripts/configs/requirements.txt