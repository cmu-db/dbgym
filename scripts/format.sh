#!/bin/bash
set -euo pipefail

# Ignore agents/ because those are all submodules.
black . --exclude agents
isort . --profile black --skip agents
