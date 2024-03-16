#!/usr/bin/env bash

set -euxo pipefail

PGDATA_REAL_DPATH="$1"
PGBIN_REAL_DPATH="$2"

mkdir -p "${PGDATA_REAL_DPATH}"
$PGBIN_REAL_DPATH/initdb "${PGDATA_REAL_DPATH}"
