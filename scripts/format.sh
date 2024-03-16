#!/bin/bash
set -euxo pipefail

find . \( ! -regex '.*/\..*' \) -name '*.py' -exec black {} +
find . \( ! -regex '.*/\..*' \) -name '*.py' -exec isort --profile black {} +