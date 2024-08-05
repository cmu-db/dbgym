#!/usr/bin/env bash

set -euxo pipefail

TPCH_REPO_ROOT="$1"

if [ ! -d "${TPCH_REPO_ROOT}/tpch-kit" ]; then
  mkdir -p "${TPCH_REPO_ROOT}"
  cd "${TPCH_REPO_ROOT}"
  git clone https://github.com/lmwnshn/tpch-kit.git --single-branch --branch master --depth 1
  cd ./tpch-kit/dbgen
  make MACHINE=LINUX DATABASE=POSTGRESQL
fi
