#!/bin/bash
set -euxo pipefail

black **/*.py
isort --profile black **/*.py