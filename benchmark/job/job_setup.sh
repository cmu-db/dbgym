#!/usr/bin/env bash

set -euxo pipefail

JOB_REPO_ROOT="$1"

if [ ! -d "${JOB_REPO_ROOT}/job-kit-gym" ]; then
  mkdir -p "${JOB_REPO_ROOT}"
  cd "${JOB_REPO_ROOT}"
  git clone https://github.com/wangpatrick57/job-kit-gym.git --single-branch --branch master --depth 1
fi
