#!/bin/bash
# This script creates a conda environment for a specific agent.
# Preconditions:
#   - conda is installed and the user has access to the conda command.
#   - The user has already updated the folder of the agent they want to
#     create a conda environment for (e.g. by calling submodule update).
#   - There does not already exist a conda environment with the same name
#     as the agent.

# Check if agent name is provided.
if [ -z "$1" ]; then
    echo "Usage: ./create_conda.sh <agent_name>"
    exit 1
fi

agent_name=$1

# Check if agent folder exists
if [ ! -d "agents/$agent_name" ]; then
    echo "Error: Agent folder '$agent_name' does not exist"
    exit 1
fi

# Note: I am intentionally not using environment.yml. I am instead using
# requirements.txt and .python_version. This is for two reasons:
#   1. environment.yml sets the conda env name. However, I want to enforce
#      that the conda env name is the same as the agent name.
#   2. requirements.txt can be used by pip and only contains packages and
#      not any additional conda-specific syntax, making it more modular
#      and flexible.

# Read in .python_version if it exists.
if [ -f "agents/$agent_name/.python_version" ]; then
    python_version=$(cat "agents/$agent_name/.python_version")
else
    echo "Warning: .python_version not found in agents/$agent_name/. Using default Python 3.10."
    python_version="3.10"
fi

echo "hello world!"
echo "python_version: $python_version"