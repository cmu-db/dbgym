#!/usr/bin/env bash

set -euxo pipefail

REPO_REAL_DPATH="$1"

# download and make postgres with boot
mkdir -p "${REPO_REAL_DPATH}"
cd "${REPO_REAL_DPATH}"
git clone git@github.com:lmwnshn/boot.git --single-branch --branch boot --depth 1
cd ./boot
./cmudb/build/configure.sh release "${REPO_REAL_DPATH}/boot/build/postgres"
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
rm REL15_1_5_1.tar.gz
cd ./pg_hint_plan-REL15_1_5_1
PATH="${REPO_REAL_DPATH}/boot/build/postgres/bin:$PATH" make
PATH="${REPO_REAL_DPATH}/boot/build/postgres/bin:$PATH" make install
cp ./pg_hint_plan.so ${REPO_REAL_DPATH}/boot/build/postgres/lib
