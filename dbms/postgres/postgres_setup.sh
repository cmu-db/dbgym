#!/usr/bin/env bash

set -euxo pipefail

PG_REPO_ROOT="$1"

if [ ! -d "${PG_REPO_ROOT}/boot" ]; then
  # download and make postgres with boot
  mkdir -p "${PG_REPO_ROOT}"
  cd "${PG_REPO_ROOT}"
  git clone git@github.com:lmwnshn/boot.git --single-branch --branch boot --depth 1
  cd ./boot
  ./cmudb/build/configure.sh release "${PG_REPO_ROOT}/boot/build/postgres"
  make clean
  make install-world-bin -j4
  cd ../

  # download and make hypopg
  git clone git@github.com:HypoPG/hypopg.git
  cd ./hypopg
  PG_CONFIG=../boot/build/postgres/bin/pg_config make install
  cd ../

  # download and make pg_hint_plan
  # we need -L to follow links
  curl -L https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL15_1_5_1.tar.gz -o REL15_1_5_1.tar.gz
  tar -xzf REL15_1_5_1.tar.gz
  cd ./pg_hint_plan-REL15_1_5_1
  PATH="${PG_REPO_ROOT}/boot/build/postgres/bin:$PATH" make
  PATH="${PG_REPO_ROOT}/boot/build/postgres/bin:$PATH" make install
  cp ./pg_hint_plan.so ${PG_REPO_ROOT}/boot/build/postgres/lib
fi
