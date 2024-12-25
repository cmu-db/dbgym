#!/bin/bash
# This script creates a conda environment for a specific agent.
# - Name matches the agent name.
# - Python version from .python_version file (if exists).
# - Dependencies from requirements.txt file (if exists).
# - gymlib is installed.
#
# Using this script is *optional*. If you have a more complex environment setup
# for your agent, just do that manually.
#
# Run it from the dbgym root folder (e.g. `./agents/create_conda.sh <agent_name>`).
#
# Before running this script, the user must update the folder of the agent
# they want to create a conda environment for (e.g. by calling submodule update).
# There are other things the user must do as well but these are all checked
# automatically by this script.

set -euo pipefail

# 1. Checks.
# 1.1. Check that conda is installed.
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed"
    exit 1
fi

# 1.2. Input validation.
if [ -z "$1" ]; then
    echo "Usage: ./create_conda.sh <agent_name>"
    exit 1
fi

agent_name=$1

if [ ! -d "agents/$agent_name" ]; then
    echo "Error: Agent folder '$agent_name' does not exist"
    exit 1
fi

# 1.3. Checks relating to conda environments.
if conda info --envs | grep -q "^$agent_name "; then
    echo "Error: Conda environment '$agent_name' already exists"
    exit 1
fi

# 1.4. Check that we're not in any conda environment
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Error: Must run from outside any conda environment (try 'conda deactivate')"
    exit 1
fi

# 2. Set up the environment.
# Note: I am intentionally not using environment.yml. I am instead using
# requirements.txt and .python_version. This is for two reasons:
#   1. environment.yml sets the conda env name. However, I want to enforce
#      that the conda env name is the same as the agent name.
#   2. requirements.txt can be used by pip and only contains packages and
#      not any additional conda-specific syntax, making it more modular
#      and flexible.

# 2.1. Set python_version variable.
if [ -f "agents/$agent_name/.python_version" ]; then
    python_version=$(cat "agents/$agent_name/.python_version")
else
    echo "Warning: .python_version not found in agents/$agent_name/. Using default Python 3.10."
    python_version="3.10"
fi

# 2.2. Create conda environment with specified Python version.
echo "Creating conda environment '$agent_name' with Python $python_version..."
eval "$(conda shell.bash hook)"
conda create -y -n "$agent_name" python="$python_version"

# 2.3. Install the packages.
conda activate "$agent_name"

if [ -f "agents/$agent_name/requirements.txt" ]; then
    echo "Installing pip requirements from agents/$agent_name/requirements.txt..."
    pip install -r "agents/$agent_name/requirements.txt"
else
    echo "Warning: requirements.txt not found in agents/$agent_name/."
fi

# We always install gymlib so that the agent has access to it.
if [ -d "gymlib" ]; then
    echo "Installing gymlib in editable mode..."
    pip install -e ./gymlib
else
    echo "Error: gymlib directory not found in $(pwd). Please ensure you're running this script from the right folder."
    exit 1
fi

conda deactivate

# 2.4. Success message.
echo "Conda environment '$agent_name' created successfully."
echo "It is not currently activated. To activate it, run 'conda activate $agent_name'."
