#!/usr/bin/env bash

set -euxo pipefail

REPO_REAL_PARENT_PATH="$1"

# Download and make postgres from the boot repository.
mkdir -p "${REPO_REAL_PARENT_PATH}"
cd "${REPO_REAL_PARENT_PATH}"
git clone https://github.com/lmwnshn/boot.git --single-branch --branch vldb_2024 --depth 1
cd ./boot
./cmudb/build/configure.sh release "${REPO_REAL_PARENT_PATH}/boot/build/postgres"
make clean
make install-world-bin -j4

# Download and make boot.
cd ./cmudb/extension/boot_rs/
cargo build --release
cbindgen . -o target/boot_rs.h --lang c
cd "${REPO_REAL_PARENT_PATH}/boot"

cd ./cmudb/extension/boot/
make clean
make install -j
cd "${REPO_REAL_PARENT_PATH}/boot"

# Download and make hypopg.
git clone https://github.com/HypoPG/hypopg.git
cd ./hypopg
PG_CONFIG="${REPO_REAL_PARENT_PATH}/boot/build/postgres/bin/pg_config" make install
cd "${REPO_REAL_PARENT_PATH}/boot"

# Download and make pg_hint_plan.
# We need -L to follow links.
curl -L https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL15_1_5_1.tar.gz -o REL15_1_5_1.tar.gz
tar -xzf REL15_1_5_1.tar.gz
rm REL15_1_5_1.tar.gz
cd ./pg_hint_plan-REL15_1_5_1
PATH="${REPO_REAL_PARENT_PATH}/boot/build/postgres/bin:$PATH" make
PATH="${REPO_REAL_PARENT_PATH}/boot/build/postgres/bin:$PATH" make install
cp ./pg_hint_plan.so ${REPO_REAL_PARENT_PATH}/boot/build/postgres/lib
