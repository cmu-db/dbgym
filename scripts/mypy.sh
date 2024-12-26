#!/bin/bash
# Ignore agents/ because those are all submodules.
# Ignore gymlib_package/build/ to avoid the error of mypy finding two gymlib packages.
mypy --config-file scripts/configs/mypy.ini . --exclude agents/ --exclude gymlib_package/build/