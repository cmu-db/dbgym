#!/bin/bash
set -euo pipefail

# Ignore agents/ because those are all submodules.
black . --exclude agents
python -m isort . --profile black --skip agents
