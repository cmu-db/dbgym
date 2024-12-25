#!/bin/bash
# This script creates a conda environment for a specific agent.
# - Name matches the agent name.
# - Python version from .python_version file in the agent's folder (if exists).
# - Dependencies from requirements.txt file in the agent's folder (if exists).
# - gymlib is installed.
#
# Using this script is *optional*. If you have a more complex environment setup
# for your agent, just do that manually.
#
# Run it from the dbgym root folder (e.g. `./scripts/build_agent_conda_env.sh <agent_name>`).
#
# Before running this script, the user must update the folder of the agent
# they want to create a conda environment for (e.g. by calling submodule update).
# There are other things the user must do as well but these are all checked
# automatically by this script.

set -euo pipefail

if [ -z "$1" ]; then
    echo "Usage: ./build_agent_conda_env.sh <agent_name>"
    exit 1
fi

agent_name=$1

if [ ! -d "agents/$agent_name" ]; then
    echo "Error: Agent folder '$agent_name' does not exist"
    exit 1
fi

./scripts/_build_conda_env.sh "$agent_name" "agents/$agent_name/.python_version" "agents/$agent_name/requirements.txt"