# Phase 2: Alignment & Bridge Integration

## Vision

- Elevate the Phase 1 Mosaic backbone into a practical multi-cohort integrator that handles unpaired, bridge, and hierarchical settings with explicit alignment heads and bridge graph regularisers.
- Deliver protective tooling (configs, diagnostics, docs) so Phase 2 experiments can be run end-to-end from the CLI, producing reproducible artefacts and sanity plots.

## Scope & Guiding Principles

1. **Bridge-aware training:** connect the `remadom/bridges` prototypes (MNN, seeded, dictionary, etc.) to the Trainer so bridge constraints participate in the loss and can be scheduled or toggled via config.
2. **Alignment maturity:** harden existing heads (MMD, Sinkhorn/OT, GW/Fused GW) by supporting warmups, cohort-aware weighting, and diagnostics; ensure masked ELBO and alignment terms coexist without numerical brittleness.
3. **Batch + dataset corrections:** expose convenient knobs to mitigate batch effects across both paired and mosaic data (problem types 2–6) by using alignment heads, bridge edges, and modality-specific scaling.
4. **Usability:** ship ready-to-run configs, CLI helpers, and visualisations (loss curves, bridge edge stats) so that switching between problem types is a single command.

## Deliverables

### 1. Bridge module integration

- Implement `remadom/adapters/bridge_head.py` so configs can instantiate bridge heads (e.g. MNNBridge → BridgeHead → Trainer).
- Extend `remadom/config/schema.py` with `BridgeConfig` entries (method, k, weight, schedules) and wire them through `config/factories.py::build_heads`.
- Add bridge sampling utilities in `remadom/bridges` (shared base classes, batched `build()`), plus tests covering:
  - mutual nearest matching (MNN)
  - seeded bridges
  - tolerance of variable cohort sizes / missing modality masks.
- Update Trainer to accumulate bridge losses, support warmup schedules, and log bridge metrics (edge counts, mean distances).

### 2. Alignment polish

- Enrich `MMDHead`, `SinkhornHead`, `GWHead` with:
  - scheduleable weights/epsilons (per `alignment.*.schedule`);
  - optional group/dataset filters (operate only on selected batches);
  - gradient-stable implementations (clamp epsilons, normalise kernels).
- Implement composite alignment options (e.g., run MMD + GW simultaneously) and ensure loss aggregation is consistent and configurable.
- Add evaluation helpers for alignment success (latent variance by batch, silhouette score, pairwise distances).

### 3. Advanced scheduling & optimisation

- Provide scheduler presets (linear/cosine/piecewise) for KL beta, modality weights, alignment/bridge weights.
- Introduce `grad_clip_per_modality` or similar control if large bridge weights destabilise training.
- Extend checkpoint payloads to include schedule state / bridge configuration for reproducibility.

### 4. CLI & configuration ecosystem

- Create config templates for each problem type with Phase 2 features enabled:
  - `mock_unpaired` with MMD/GW warmups;
  - `mock_bridge` with MNN bridge head;
  - `mock_hierarchical` with combined alignment + bridge.
- Update `scripts/run_all_examples.sh` (or add a companion script) to run Phase 2 suites and optionally compare metrics.
- Enhance CLI output:
  - print alignment/bridge loss traces;
  - emit bridge edge histograms and saved plots (`runs/<cfg>/bridge_stats.json`, `bridge_edges.png`).

- Provide optional CLI flags to skip plotting or force CPU execution when GPU unavailable.

### 5. Diagnostics & documentation

- Extend docs with:
  - `docs/bridge_mnn.md` augmentation (pipeline diagrams, config breakdown);
  - new `docs/checklists/phase2_validation.md` capturing QA steps;
  - updates to `docs/problem_types.md` linking to Phase 2 configs.
- Add integration smoke tests (`tests/integration/test_bridge_training.py`, etc.) covering:
  - training on bridge scenario with bridge loss decreasing;
  - alignment head schedule behaviour (beta warmup hitting target values).
- Update MkDocs navigation if necessary (Phase 2 section, CLI usage page).

## Milestones

1. **M2.1 – Bridge head wired up**
   - BridgeHead implemented, config hook added, unit tests for edge construction.
   - Trainer logging bridge loss, no regressions in Phase 1 tests.

2. **M2.2 – Alignment upgrade**
   - Schedules + diagnostics for MMD/Sinkhorn/GW validated on mock datasets.
   - CLI emits expanded metrics (loss JSON + plots).

3. **M2.3 – Full problem-type coverage**
   - Paired/unpaired/bridge/mosaic/hierarchical configs run end-to-end with Phase 2 features.
   - Documentation & checklists merged, `scripts/run_all_examples.sh` (or variant) covers the suite.

## Acceptance Criteria

- All targeted problem types train with bridge/alignment activated; integration tests capture bridge and alignment improvements.
- CLI runs produce checkpoints, metrics JSON, and loss/bridge plots; README/MkDocs document how to interpret them.
- Docs clearly explain new knobs, scheduling semantics, and troubleshooting tips.
- No regressions in Phase 1 functionality (unit/integration suite green).

## Open Questions & Risks

- Numerical stability for OT/GW losses when modalities are extremely sparse; may require slicing or entropic regularisation.
- Performance of bridge edge computation on large batches (consider approximate KNN / ANN backends).
- Whether to introduce latent disentanglement (z_bio vs z_tech) during Phase 2 or defer to Phase 3.
- Standardising evaluation metrics (SCIB, LISI) vs. custom quick diagnostics.
