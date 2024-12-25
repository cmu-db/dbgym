#!/bin/bash
# This script builds the conda environment used by the gym itself (i.e. the orchestrator).

set -euo pipefail

./scripts/_build_conda_env.sh "dbgym" "scripts/.python_version" "scripts/requirements.txt"
