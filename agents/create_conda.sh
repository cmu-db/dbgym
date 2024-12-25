#!/bin/bash
# This script creates a conda environment for a specific agent.
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

echo "hello world!"