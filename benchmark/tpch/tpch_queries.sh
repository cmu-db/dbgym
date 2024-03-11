#!/usr/bin/env bash

set -euxo pipefail

cd "${TPCH_REPO_ROOT}/dbgen"
set +x
for seed in $(seq "${TPCH_QUERY_START}" "${TPCH_QUERY_STOP}"); do
  if [ ! -d "${TPCH_QUERY_ROOT}/${seed}" ]; then
    mkdir -p "${TPCH_QUERY_ROOT}/${seed}"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "${TPCH_QUERY_ROOT}/${seed}/${qnum}.sql"
    done
  fi
done
set -x
cd "${ROOT_DIR}"
