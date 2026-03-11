#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-./checkpoints/pi0_fast_clanker/default}"
S3_URI="${S3_URI:-}"

if [[ -z "${S3_URI}" ]]; then
  echo "Set S3_URI, for example: s3://my-bucket/openpi/pi0_fast_clanker/default" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found in PATH" >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"
aws s3 sync "${S3_URI}" "${LOCAL_DIR}" --no-progress
