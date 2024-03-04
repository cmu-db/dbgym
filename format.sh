#!/bin/bash
set -euxo pipefail

find . -name '*.py' -exec black {} +
find . -name '*.py' -exec isort --profile black {} +