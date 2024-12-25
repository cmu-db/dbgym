#!/bin/bash
# Ignore agents/ because those are all submodules.
mypy --config-file scripts/configs/mypy.ini . --exclude agents/