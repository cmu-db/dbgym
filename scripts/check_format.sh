#!/bin/bash
set -euxo pipefail

# Ignore agents/ because those are all submodules.
black . --check --exclude agents
isort . --profile black -c --skip agents
