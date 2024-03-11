#!/usr/bin/env bash

set -euxo pipefail

PG_REPO_ROOT="$1"

if [ ! -d "${PG_REPO_ROOT}/boot" ]; then
  mkdir -p "${PG_REPO_ROOT}"
  cd "${PG_REPO_ROOT}"
  git clone git@github.com:lmwnshn/boot.git --single-branch --branch boot --depth 1
  cd ./boot
  ./cmudb/build/configure.sh release "${PG_REPO_ROOT}/boot/build/postgres"
  make clean
  make install-world-bin -j4
  cd ../

  git clone git@github.com:HypoPG/hypopg.git
  cd ./hypopg
  PG_CONFIG=../boot/build/postgres/bin/pg_config make install
  cd ../
fi
