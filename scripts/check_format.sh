#!/bin/bash
set -euxo pipefail

black . --check
isort . --profile black -c
