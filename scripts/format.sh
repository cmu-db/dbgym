#!/bin/bash
set -euxo pipefail

black .
isort .
