# Phase 3 Validation Checklist

Use this to verify Phase 3 features (graph/temporal regularisers, adapters, SCIB/batch metrics, visualisations, resume/schedules).

## Prereqs
- `conda activate REMADOM`
- Rebuild mock data/configs: `bash scripts/generate_all_mock_data.sh`
- Optional deps for full metrics/plots: `pip install umap-learn scikit-learn scib anndata`

## Smoke tests
- Unit tests:
  ```bash
  python -m pytest tests/unit/test_alignment_metrics.py tests/unit/test_bridge_providers.py tests/unit/test_schedules.py
  ```
- Integration (Phase 1/2 baseline):
  ```bash
  python -m pytest tests/integration/test_phase1_training.py
  ```

## Graph/Temporal heads
- Run a config exercising graph/temporal (after implementation), e.g. `configs/examples/temporal_hybrid.yaml`:
  ```bash
  python -m remadom.cli.train --cfg configs/examples/temporal_hybrid.yaml --plot-umap
  ```
- Check artefacts:
  - `metrics.final.json` includes graph/temporal losses.
  - UMAP/TSNE plots saved if flags passed.

## SCIB/batch metrics
- Run any example with `logging.collect_metrics: true` and SCIB deps installed:
  ```bash
  python -m remadom.cli.train --cfg configs/examples/mock_bridge.yaml --plot-umap --scib
  ```
- Verify outputs:
  - `scib_metrics.json` exists with ilisi/kBET (if deps available).
  - `metrics.final.json` contains `scib` block.

## Adapters / reference mapping
- Once adapters are wired to CLI, run mapping smoke test (placeholder):
  ```bash
  python -m remadom.cli.mapref --cfg configs/examples/mapref_rna.yaml
  ```
- Check mapping metrics/logs in run dir.

## Resume & schedules
- Train a short run, then resume:
  ```bash
  python -m remadom.cli.train --cfg configs/examples/mock_paired.yaml --plot-latent --plot-umap
  python -m remadom.cli.train --cfg configs/examples/mock_paired.yaml --resume runs/mock/mock_paired/checkpoint.last.pt --plot-latent
  ```
- Verify:
  - Schedules/weights are restored (no jumps in loss traces).
  - New metrics appended; checkpoints updated.

## Visual artefacts
- For any run with `--plot-umap/--plot-tsne/--plot-latent`, ensure:
  - `latent_umap.png`, `latent_tsne.png`, `latent_pca.png` exist (or skipped with dependency message).
  - `evaluation_plots.png` present when evaluation enabled.

## Performance sanity
- (Optional) Large-batch/CPU fallback smoke:
  ```bash
  python -m remadom.cli.train --cfg configs/examples/mock_mosaic.yaml --force-cpu --metrics-only
  ```
- Ensure no autocast/scaler errors on CPU.
