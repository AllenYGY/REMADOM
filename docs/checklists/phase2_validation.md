# Phase 2 Validation Checklist

Use this checklist to verify Phase 2 features (alignment heads, bridge providers, diagnostics, schedules) are working end-to-end.

## Before running
- `conda activate REMADOM`
- Re-generate mock datasets/configs to ensure truth fields are present:
  ```bash
  bash scripts/generate_all_mock_data.sh
  ```

## Core smoke tests
- Unit tests (bridge providers, schedules, alignment metrics):
  ```bash
  python -m pytest tests/unit/test_bridge_providers.py tests/unit/test_schedules.py tests/unit/test_alignment_metrics.py
  ```
- Integration smoke (Phase 1 training):
  ```bash
  python -m pytest tests/integration/test_phase1_training.py
  ```

## Bridge comparison
- Run all bridge strategies on the bridge mock:
  ```bash
  ./scripts/compare_bridge_strategies.sh -- optim.epochs=50
  ```
- Check artefacts per method under `runs/bridge_comparison/<method>/`:
  - `metrics.final.json` (includes evaluation.bridge_imputation)
  - `bridge_metrics.json`, `bridge_edges.png`
  - `evaluation_plots.png`, `evaluation_samples.npz`
- Inspect summary table:
  ```bash
  cat runs/bridge_comparison/summary.txt
  ```

## Example configs
- Full mock suite:
  ```bash
  ./scripts/run_all_examples.sh -- optim.epochs=20
  ```
- For each `runs/mock/<case>/`, verify:
  - `metrics.final.json` exists with train metrics and optional evaluation block.
  - `evaluation.mock.json` and `evaluation_plots.png` (if evaluation.enabled).
  - Loss curves (`loss_curve.png`) rendered.

## Diagnostics
- Alignment metrics: when `logging.collect_metrics=true`, check `alignment_metrics.json` (bridge case) and `metrics.final.json` alignment block.
- Bridge edges: ensure non-zero edges for MNN/Seeded/LinearMap/Dictionary runs; review `bridge_metrics.json`.

## Schedules
- Validate schedule behaviour via unit tests (`tests/unit/test_schedules.py`) and by inspecting `config.resolved.yaml` with beta/weight schedules present.

## Troubleshooting
- No evaluation outputs: ensure configs have `evaluation.enabled: true` and `evaluation.tasks` set; re-run `generate_all_mock_data.sh` after code updates.
- Singular matrix in bridge providers: recent fallback to lstsq should avoid crashes; if persists, reduce `bridge_size` or add `lam`.
- Missing alignment metrics: set `logging.collect_metrics: true` in the YAML to record centroid/silhouette proxies.
