#!/bin/bash
set -euxo pipefail

black .
isort . --profile black
