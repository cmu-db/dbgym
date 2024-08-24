#!/bin/bash
set -euxo pipefail

black . --check
isort . -c
