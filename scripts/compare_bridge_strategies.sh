#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
CFG="${REPO_ROOT}/configs/examples/mock_bridge.yaml"
RUN_ROOT="${REPO_ROOT}/runs/bridge_comparison"

declare -a METHODS=("mnn" "seeded" "dictionary" "linmap")
declare -a SUMMARY_ROWS=()

mkdir -p "${RUN_ROOT}"
echo "[remadom] comparing bridge strategies on ${CFG}"

for method in "${METHODS[@]}"; do
  run_dir="${RUN_ROOT}/${method}"
  echo ""
  echo ">>> bridge.method=${method}"
  python -m remadom.cli.train --cfg "${CFG}" logging.run_dir="${run_dir}" bridge.method="${method}" "$@"
  metrics_file="${run_dir}/metrics.final.json"
  if [[ -f "${metrics_file}" ]]; then
    SUMMARY_ROWS+=("${metrics_file}")
  fi
done

if [[ ${#SUMMARY_ROWS[@]} -gt 0 ]]; then
  summary_out="${RUN_ROOT}/summary.txt"
  python - <<'PY'
import json
import pathlib
import sys

rows = []
for path in sys.argv[1:]:
    run_dir = pathlib.Path(path).parent
    method = run_dir.name
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    train = data.get("train", {})
    eval_metrics = data.get("evaluation", {}).get("bridge_imputation", {})
    rows.append((method, train.get("loss"), train.get("recon"), train.get("kl"), eval_metrics.get("rna_mae"), eval_metrics.get("atac_mae")))

rows.sort(key=lambda r: r[0])
out_path = pathlib.Path(sys.argv[0])
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("method\tloss\trecon\tkl\trna_mae\tatac_mae\n")
    for row in rows:
        method, loss, recon, kl, rna_mae, atac_mae = row
        fh.write(f"{method}\t{loss:.4f}\t{recon:.4f}\t{kl:.4f}\t{rna_mae:.4f}\t{atac_mae:.4f}\n")
PY
  "${summary_out}" "${SUMMARY_ROWS[@]}"
  echo ""
  echo "[remadom] bridge comparison summary written to: ${summary_out}"
fi
