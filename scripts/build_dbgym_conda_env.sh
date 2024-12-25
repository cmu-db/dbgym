#!/bin/bash
# This script builds the conda environment used by the gym itself (i.e. the orchestrator).
# This script is optional. You don't need to use conda if you don't want to (the CI doesn't use conda, for instance)

set -euo pipefail

./scripts/_build_conda_env.sh "dbgym" "scripts/configs/.python_version" "scripts/configs/requirements.txt"
