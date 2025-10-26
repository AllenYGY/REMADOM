#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate REMADOM   # make sure dependencies are installed
#   bash scripts/generate_all_mock_data.sh
#
# This will populate examples/mock/*.h5ad and configs/examples/mock_*.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${REPO_ROOT}/examples/mock"
CFG_DIR="${REPO_ROOT}/configs/examples"
GENERATOR="${REPO_ROOT}/scripts/make_mock_multimodal.py"

# Ensure the repository is on PYTHONPATH so `remadom` can be imported
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"
mkdir -p "${CFG_DIR}"

declare -A PROBLEMS=(
  [paired]="mock_paired_cite.h5ad"
  [unpaired]="mock_unpaired_rna_atac.h5ad"
  [bridge]="mock_bridge_rna_atac.h5ad"
  [mosaic]="mock_mosaic_multiome.h5ad"
  [prediction]="mock_prediction_rna_to_atac.h5ad"
  [hierarchical]="mock_hierarchical_multistudy.h5ad"
)

for problem in "${!PROBLEMS[@]}"; do
  out_file="${OUT_DIR}/${PROBLEMS[$problem]}"
  cfg_file="${CFG_DIR}/mock_${problem}.yaml"
  echo "[remadom] generating ${problem} dataset -> ${out_file}"
  python "${GENERATOR}" \
    --problem "${problem}" \
    --out "${out_file}" \
    --config-out "${cfg_file}" \
    --seed 0 \
    --cells 600 \
    --genes 1000 \
    --peaks 5000 \
    --proteins 30
done

echo "[remadom] mock datasets ready in ${OUT_DIR}"
