#!/bin/bash
# This helper script creates a conda environment.
# You should not run this directly. Instead, use build_agent_conda_env.sh or build_gym_conda_env.sh.

set -euo pipefail

# 1. Checks.
# 1.1. Check that conda is installed.
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed"
    exit 1
fi

# 1.2. Input validation.
if [ "$#" -lt 3 ]; then
    echo "Usage: ./_build_conda_env.sh <env_name> <python_version_path> <requirements_path>"
    exit 1
fi

env_name=$1
python_version_path=$2
requirements_path=$3

# 1.3. Check that the environment doesn't already exist.
if conda info --envs | grep -q "^$env_name "; then
    echo "Error: Conda environment '$env_name' already exists"
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
if [ -f "$python_version_path" ]; then
    python_version=$(cat "$python_version_path")
else
    echo "Info: .python_version not found in $python_version_path. Using default Python 3.10."
    python_version="3.10"
fi

# 2.2. Create conda environment with specified Python version.
echo "Creating conda environment '$env_name' with Python $python_version..."
eval "$(conda shell.bash hook)"
conda create -y -n "$env_name" python="$python_version"

# 2.3. Install the packages.
conda activate "$env_name"

if [ -f "$requirements_path" ]; then
    echo "Installing pip requirements from $requirements_path..."
    pip install -r "$requirements_path"
else
    echo "Info: $requirements_path not found. Skipping pip install."
fi

# We always install gymlib so that the agent has access to it.
if [ -d "gymlib_package" ]; then
    echo "Installing gymlib in editable mode..."
    pip install ./gymlib_package
else
    echo "Error: gymlib_package directory not found in $(pwd). Please ensure you're running this script from the right folder."
    exit 1
fi

conda deactivate

# 2.4. Success message.
echo "Conda environment '$env_name' created successfully."
echo "It is not currently activated. To activate it, run 'conda activate $env_name'."
