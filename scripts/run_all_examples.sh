#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
CFG_DIR="${REPO_ROOT}/configs/examples"

SUMMARY=0
CLI_ARGS=()
PASS_THRU=0
while [[ $# -gt 0 ]]; do
  if [[ ${PASS_THRU} -eq 1 ]]; then
    CLI_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --summary)
      SUMMARY=1
      shift
      ;;
    --)
      PASS_THRU=1
      shift
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

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
if [[ ${#CLI_ARGS[@]} -gt 0 ]]; then
  echo "[remadom] additional CLI overrides: ${CLI_ARGS[*]}"
else
  echo "[remadom] additional CLI overrides: <none>"
fi

cd "${REPO_ROOT}"

declare -a SUMMARY_ROWS=()

for stem in "${MOCK_CASES[@]}"; do
  cfg="${CFG_DIR}/${stem}"
  if [[ ! -f "${cfg}" ]]; then
    echo "[remadom] warning: missing config ${cfg}, skipping" >&2
    continue
  fi
  echo ""
  echo ">>> running ${stem}"
  python -m remadom.cli.train --cfg "${cfg}" "${CLI_ARGS[@]}"
  metrics_file="${REPO_ROOT}/runs/mock/${stem%.*}/metrics.final.json"
  eval_file="${REPO_ROOT}/runs/mock/${stem%.*}/evaluation.mock.json"
  if [[ ${SUMMARY} -eq 1 && -f "${metrics_file}" ]]; then
    SUMMARY_ROWS+=("${metrics_file}")
  fi
  if [[ -f "${eval_file}" ]]; then
    echo "[remadom] evaluation metrics stored at: ${eval_file}"
  fi
done

echo ""
echo "[remadom] mock runs completed"

if [[ ${SUMMARY} -eq 1 && ${#SUMMARY_ROWS[@]} -gt 0 ]]; then
  summary_out="${REPO_ROOT}/runs/mock/summary.txt"
  python - <<'PY'
import json
import pathlib
import sys

rows = []
for path in sys.argv[1:]:
    cfg = pathlib.Path(path).parent.name
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    train = data.get("train", {})
    rows.append((cfg, train.get("loss"), train.get("recon"), train.get("kl")))

rows.sort(key=lambda r: r[0])
out = pathlib.Path(sys.argv[0])
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("config\tloss\trecon\tkl\n")
    for cfg, loss, recon, kl in rows:
        fh.write(f"{cfg}\t{loss:.4f}\t{recon:.4f}\t{kl:.4f}\n")
PY
  "${summary_out}" "${SUMMARY_ROWS[@]}"
  echo "[remadom] summary written to: ${summary_out}"
fi
