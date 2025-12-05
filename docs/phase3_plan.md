# Phase 3: Graph/Temporal Regularisation, Adapters, and Evaluation Harness

## Goal

Deliver the advanced alignment and scalability features that sit on top of the Phase 1 backbone and Phase 2 bridge/alignment heads. Phase 3 focuses on structural priors (graphs/temporal), reference adapters, robust evaluation (SCIB-style + visualisations), and operational polish (resume, schedules, performance).

## Scope & Deliverables

### 1) Structural alignment (graphs & temporal)
- Implement/finish Laplacian regulariser and temporal heads:
  - GraphRegularizer: kNN/ANN backend, symmetric Laplacian, sparse penalties on z_bio.
  - Temporal head: enforces ordering (time_key) via OT/contrastive loss.
- Config plumbing:
  - `structure.graph` block (backend, k, metric, lambda, normalized).
  - `alignment.temporal` block with weight/schedule.
- Tests:
  - Unit test on synthetic manifold preserving neighbourhoods.
  - Integration smoke with `temporal_hybrid.yaml`.

### 2) Adapters & reference mapping
- Expose `remadom/adapters` via CLI (mapref):
  - Config schema for adapters (arches, mixtures, mappers).
  - CLI entry to load reference embeddings and train adapters.
- Integration test:
  - Small mock reference/target; ensure adapter reduces batch variance and improves mapping metrics.
- Docs:
  - Quickstart on reference mapping; update `PLAN.md` and `problem_types.md`.

### 3) Evaluation harness
- SCIB/batch metrics:
  - Make ilisi/kBET computation first-class; add CLI flag `--scib` to write `scib_metrics.json` and merge into `metrics.final.json`.
  - Optional sampling controls to keep runtime bounded.
- Visualisations:
  - UMAP/t-SNE exports gated by flags; ensure graceful skip when deps missing.
  - Bridge/latent plots auto-saved; document expected artefacts.
- Notebooks/scripts:
  - Add `scripts/eval_scib.py` or notebook template to run SCIB on existing runs.
  - Update `docs/checklists/phase3_validation.md` with commands and expected outputs.

### 4) Schedules & resume polish
- Persist and restore schedule state (beta, head weights, modality weights) and step counters on resume.
- Extend scheduler options (cosine_restart, piecewise, per-step) with tests.
- CLI: `--resume` should restore schedules and continue logging seamlessly.

### 5) Performance & robustness
- Mixed precision polish: use `torch.amp.autocast`/`GradScaler` modern APIs; CPU fallback paths.
- Large-N handling:
  - Datalaoder/cache knobs for backed AnnData.
  - ANN backend selection hints in docs for >1e6 cells.
- Profiling hooks (optional timers) for data/forward/backward.

### 6) Documentation
- Add `phase3_plan.md` (this file) to `docs/`; link from `docs/PLAN.md`.
- Update `docs/bridge_mnn.md`, `docs/mock_data.md`, `docs/network_architecture.md` with new heads/graphs/adapters.
- Publish validation checklist `docs/checklists/phase3_validation.md` (commands for SCIB, UMAP/TSNE, adapter mapping).

## Milestones
- **M3-A** Structural heads: graph/temporal implemented, configs live, unit/integration tests pass.
- **M3-B** Adapters & mapping: CLI supports adapters; mock mapping test green.
- **M3-C** Evaluation harness: SCIB/visual exports integrated; Phase 3 checklist green; resume/schedule persistence verified.

## Acceptance Criteria
- `./scripts/run_all_examples.sh` (or relevant configs) produce: metrics.final.json with SCIB (when deps present), alignment/bridge artefacts, latent UMAP/TSNE if flags passed.
- `python -m remadom.cli.train --resume <ckpt>` continues schedules and writes coherent history.
- Graph/temporal regularisers show measurable effect on synthetic tests (neighbourhood preservation/ordering).
- Docs/checklists updated; unit/integration tests pass in CI.
