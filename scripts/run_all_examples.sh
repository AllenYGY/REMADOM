#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
CFG_DIR="${REPO_ROOT}/configs/examples"

if [[ ! -d "${CFG_DIR}" ]]; then
  echo "[remadom] config directory not found: ${CFG_DIR}" >&2
  exit 1
fi

declare -a MOCK_CASES=(
  "mock_paired.yaml"
  "mock_unpaired.yaml"
  "mock_bridge.yaml"
  "mock_mosaic.yaml"
  "mock_prediction.yaml"
  "mock_hierarchical.yaml"
)

echo "[remadom] running mock problem-type configs"
echo "[remadom] additional CLI overrides: ${*:-<none>}"

cd "${REPO_ROOT}"

for stem in "${MOCK_CASES[@]}"; do
  cfg="${CFG_DIR}/${stem}"
  if [[ ! -f "${cfg}" ]]; then
    echo "[remadom] warning: missing config ${cfg}, skipping" >&2
    continue
  fi
  echo ""
  echo ">>> running ${stem}"
  python -m remadom.cli.train --cfg "${cfg}" "$@"
done

echo ""
echo "[remadom] mock runs completed"
