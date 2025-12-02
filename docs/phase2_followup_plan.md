# Phase 2 Follow-up Plan

## Vision

- Close the remaining Phase 2 gaps by expanding bridge providers, strengthening alignment diagnostics, and enriching CLI artefacts so that every mock scenario (paired → hierarchical) runs with Phase 2 features enabled and produces actionable outputs.
- Establish validation tooling (metrics, plots, tests) that makes bridge/alignment behaviour observable and reproducible from the CLI.

## Scope & Guiding Principles

1. **Provider completeness:** bridge integration should support multiple strategies (MNN, seeded anchors, dictionary mapping) with consistent configuration and diagnostics.
2. **Observable alignment:** every alignment/bridge decision should surface quantitative signals (loss trends, batch variance, cohort edge stats) in CLI outputs.
3. **Robust scheduling:** schedules must cover additional patterns (piecewise, per-step) and persist state so runs are restartable.
4. **Documentation-first:** configs, docs, and tests must demonstrate the Phase 2 feature set across problem types.

## Deliverables

### 1. Bridge provider expansion

- Implement providers in `remadom/bridges`:
  - `SeededBridge` for seed-based matching;
  - `DictionaryBridge` (ridge regression);
  - batched `BridgeProvider.build()` to handle large cohorts.
- Register providers via `build_bridge_provider`.
- Update `BridgeHead` to accept `normalize`, `max_edges`, cohort filters.
- Add unit tests ensuring each provider builds edges for balanced/unbalanced cohorts and respects masks.

### 2. Alignment diagnostics & metrics

- Add `remadom/eval/alignment_metrics.py`:
  - batch variance in latent space,
  - silhouette score,
  - pairwise cross-modality distances,
  - optional trustworthiness.
- Extend `Trainer` to optionally compute diagnostics each epoch (config toggle).
- Emit diagnostics into `metrics.final.json` and per-epoch summaries.
- Provide utilities (e.g., `remadom/eval/plots.py`) for quick scatter/heatmap plots.

### 3. Scheduling & optimisation enhancements

- Expand `remadom/utils/schedules` with:
  - piecewise linear,
  - stepped decay,
  - cosine restarts,
  - step-based schedules (`mode: epoch|step`).
- Allow per-modality gradient clipping; expose config key `optim.grad_clip_modality`.
- Extend checkpoint payload to include:
  - head schedule state (current epsilon/weight),
  - modality schedule progress,
  - bridge configuration snapshot.
- Implement resume support restoring schedule counters.

### 4. CLI & configuration ecosystem

- Provide Phase 2 configs for every mock problem type (paired, unpaired, bridge, mosaic, prediction, hierarchical):
  - enable relevant heads,
  - configure schedules,
  - set bridge parameters where applicable.
- Enhance CLI output:
  - per-head loss trace printouts,
  - optional `--no-plot`, `--force-cpu`, `--metrics-only` flags,
  - logging of bridge/alignment metrics every `log_interval`.
- Generate artefacts under each run dir:
  - `bridge_edges.png`, `bridge_degree_hist.png`,
  - `alignment_metrics.json`,
  - optional latent UMAP/TSNE plots.
- Update `scripts/run_all_examples.sh` to:
  - accept overrides (`--epochs=50`, etc.),
  - produce aggregated comparison table (`runs/mock/summary.txt`).

### 5. Documentation & tests

- Documentation:
  - Expand `docs/bridge_mnn.md` with diagrams, provider comparison table, config snippets.
  - Add `docs/checklists/phase2_validation.md` capturing QA steps (commands, expected outputs).
  - Update `docs/PLAN.md`, `docs/problem_types.md`, `docs/network_architecture.md` to reference Phase 2 features, link to new configs.
  - Add CLI quickstart section describing new flags/artefacts.
- Tests:
  - New integration tests:
    - `test_bridge_training.py`: ensures bridge loss decreases, metrics files exist.
    - `test_alignment_metrics.py`: checks diagnostics output values for mock datasets.
    - `test_cli_phase2_artifacts.py`: runs CLI on bridge config and validates generated files.
  - Update unit tests for schedules and bridge providers.
  - Ensure `pytest tests` passes without needing manual monkeypatch workarounds.

## Case-specific implementation plan (Outstanding Phase 2 follow-ups)

### mock_paired (Problem Type 1)

- **Goal:** Evaluate paired recon/imputation fidelity.
- **Tasks:**
  1. Extend `remadom/eval/metrics.py` with per-modality RMSE/Pearson helpers.
  2. Add CLI hook (`--eval-paired`) that runs inference with each modality dropped in turn, logs metrics to `paired_eval.json`, and plots scatter overlays.
  3. Integrate into `mock_paired.yaml` (set `logging.eval_enabled: true`).
  4. Add regression test comparing metrics to mock ground truth tolerances.
- **Acceptance:** `python -m remadom.cli.train --cfg configs/examples/mock_paired.yaml eval.paired=true` emits non-zero metrics + plot; new test passes.

### mock_unpaired (Problem Type 2)

- **Goal:** Demonstrate latent alignment without shared cells.
- **Tasks:**
  1. Enable `alignment.mmd` in the example config, optionally gating weight schedules.
  2. Add evaluation routine that computes RNA↔ATAC imputation error by decoding missing modality for each cohort and comparing against mock truth saved in `obsm`.
  3. Report latent alignment diagnostics (batch variance, silhouette) in `alignment_metrics.json`.
  4. Create integration test that ensures metrics fall below defined thresholds when MMD is on.
- **Acceptance:** CLI run writes `alignment_metrics.json` with finite variance, imputation MAE < tolerance, and test verifies drop vs baseline.

### mock_bridge (Problem Type 3)

- **Goal:** Validate bridge head impact on single-modality cohorts.
- **Tasks:**
  1. Expand `BridgeHead` logging to capture per-cohort reconstruction/imputation changes.
  2. Add script `scripts/eval_bridge_imputation.py` that compares pre/post-bridge checkpoints on held-out cells.
  3. Ensure metrics + plots (edge histogram, imputation boxplots) land in run dir.
  4. Integration test: run shortened config with `bridge.method=mnn`, assert non-zero edges + imputation improvement.
- **Acceptance:** Example run reports positive bridge edges and improved imputation; test confirms.

### mock_mosaic (Problem Type 4)

- **Goal:** Stress-test arbitrary missingness.
- **Tasks:**
  1. Update mock generator to emit masks at varying rates plus ground-truth full modalities.
  2. Add scheduler that randomly drops modalities per epoch to simulate mosaic data.
  3. Implement evaluation that samples cells, imputes missing modalities, and reports MAE vs known truth; include visual plot of loss vs missing-rate.
  4. Add integration test verifying evaluation artefacts exist.
- **Acceptance:** CLI run with `mock_mosaic.yaml` writes `mosaic_eval.json` + plot; missing-rate sweep script runs headless test.

### mock_prediction

- **Goal:** Quantify predictive accuracy when only one modality is available at inference.
- **Tasks:**
  1. Provide CLI mode `python -m remadom.cli.impute --cfg ... --predict-target atac`.
  2. Export predicted modality matrix to `runs/.../predicted_<mod>.h5ad`.
  3. Add evaluation comparing predicted vs true modality on held-out cells, logging metrics + histogram plot.
  4. Integration test ensures CLI command produces file and metrics JSON.
- **Acceptance:** Example run outputs prediction artefacts with metrics below thresholds; test ensures reproducibility.

### mock_hierarchical

- **Goal:** Handle multi-study batch effects with bridge/alignment combos.
- **Tasks:**
  1. Introduce hierarchical head (e.g., CORAL/temporal) targeting nested `obs["study"]` and `obs["batch"]`.
  2. Extend config with dataset-specific bridge weights and graph regularization.
  3. Add evaluation computing per-study batch mixing score (e.g., ASW) and UMAP overlays saved as PNG.
  4. Integration test (short epochs) verifying metrics JSON + plot generation.
- **Acceptance:** CLI run writes `hierarchical_eval.json` showing reduced batch variance; test confirms file presence and metric bounds.

## Milestones

1. **M2-F1 – Bridge provider suite**
   - Seeded & dictionary bridges implemented with tests.
   - BridgeHead exposes extra knobs (filters, normalization).

2. **M2-F2 – Diagnostics & schedules**
   - Alignment metrics recorded in CLI outputs.
   - New schedule types and per-modality clipping integrated.
   - Checkpoints capture schedule state.

3. **M2-F3 – CLI & docs complete**
   - Phase 2 configs run end-to-end (run script green).
   - Documentation refreshed; validation checklist published.
   - Artefact plots (bridge/metrics) generated for mock scenarios.

## Acceptance Criteria

- `./scripts/run_all_examples.sh` succeeds, producing metrics + plots for all mock cases.
- `python -m pytest tests` passes with new bridge/alignment diagnostics tests.
- Config docs clearly explain how to toggle providers and schedules.
- CLI outputs include bridge/alignment summaries and artefacts by default (with opt-outs).

## Open Questions & Risks

- Computational cost of additional diagnostics on large datasets—may need sampling options.
- Scalability of complex bridge providers (dictionary, seeded) for high-dimensional data.
- Visualisation dependencies (matplotlib) might need headless support in CI environments.
- Decision on default schedule behaviour (per-step vs per-epoch) across modalities.
