# Phase 1: Core Mosaic Backbone

## Goal

- Deliver a working Mosaic-first VAE backbone that trains end to end on paired or mosaic RNA/ATAC/ADT datasets using masked ELBO, AMP, and basic checkpointing.
- Establish the `remadom` namespace so later phases (alignment heads, evaluation, CLI) can plug into a stable package layout.

## Code base starting point

- Existing modules now live under `remadom/` (migrated from the former `mosaix/`) with skeleton implementations for encoders, decoders, PoE fusion, data loaders, and the trainer.
- Documentation (`docs/Background.md`, `docs/PLAN.md`, `docs/program_structure.md`) already captures the architectural intent; Phase 1 implements the minimal functional subset.

## Deliverables

### 1. Rename and plumbing

- Migrate the legacy `mosaix/` package to `remadom/`, update imports, tests, configs, and CLI entry points in `pyproject.toml` and `scripts/`.
- Replace documentation references and package metadata so `import remadom` works everywhere.

### 2. MosaicVAE assembly (`remadom/core`)

- Implement the constructor, `encode`, `fuse_posteriors`, `decode_all`, `_mod_nll`, `embed`, `impute`, and `set_beta` in `remadom/core/vae.py`, wiring in `ProductOfExperts` and NLL helpers.
- Extend `encoders.py` and `decoders.py` with modality-specific subclasses (RNA/ATAC/ADT) and registration hooks that configs can target by name.
- Ensure masked ELBO handles per-modality availability flags and optional modality weight schedules.

### 3. Data ingestion (`remadom/data`)

- Harden `loaders.py` into `build_dataloaders(cfg)` that reads AnnData, aligns to registries, tracks modality masks, attaches library sizes, and yields `Batch` objects on CPU or GPU.
- Populate preprocessing hooks (`preprocess.py`) with optional RNA normalization, ATAC TF-IDF/LSI, and ADT scaling (graceful no-op when dependencies are missing).
- Add lightweight dataset/datamodule helpers so matrices stay cached between epochs and metadata (batch, dataset, time) is preserved.

### 4. Training loop core (`remadom/train`)

- Add AMP support using `torch.cuda.amp.autocast` and `GradScaler`, integrate beta and modality weight schedules, and combine alignment losses when provided.
- Implement checkpoint utilities in `state.py` and `callbacks.py` that persist model, optimizer, scheduler, and scaler; expose resume options via config.
- Provide minimal validation hooks that compute masked ELBO on a held-out loader and surface metrics for logging.

### 5. Config and factories

- Define typed config models covering Phase 1 needs (data sources, model dims, optimization hyperparameters, schedules).
- Build factories that instantiate encoders, decoders, VAE, optimizer, scheduler, and trainer from config, powering `python -m remadom.cli.train --config config.yaml`.

### 6. Validation and documentation

- Add unit tests for `ProductOfExperts`, modality masks in `MosaicVAE.elbo`, and decoder NLLs; provide an integration smoke test that trains a few steps on toy RNA+ATAC data with AMP enabled.
- Document Phase 1 usage and expected outputs (loss curves, checkpoint artifacts) in `docs/PLAN.md` and keep this file updated with status.

### 7. Mock data generation (for tests and demos)

- Implement a reusable mock-data utility (`remadom/data/mock.py` + CLI wrapper `scripts/make_mock_multimodal.py`) that produces synthetic AnnData objects with controllable latent structure, modality coverage, and batch composition.
- Base generator:
  - Sample latent “cell states” from 3–5 Gaussian clusters; assign each cluster to a biological label for evaluation.
  - Define modality-specific linear / nonlinear decoders (e.g., RNA counts via NB with cluster-specific means, ATAC binary via logistic, ADT Gaussian/Poisson mixture) plus optional noise and library-size scalers.
  - Provide knobs for batch effects (additive/ multiplicative offsets) and for per-modality missingness masks.
- Scenario presets (one output per problem type, stored under `examples/mock/` with matching configs):
  1. **Paired**: all cells have RNA+ADT (CITE-seq style); no missingness; two batches with mild shifts. → `mock_paired_cite.h5ad`
  2. **Unpaired**: disjoint RNA-only and ATAC-only cohorts sharing latent clusters; no overlapping cells. → `mock_unpaired_rna_atac.h5ad`
  3. **Bridge**: large RNA-only and ATAC-only sets plus a 5 % paired bridge subset sampled from both; allows testing bridge heads. → `mock_bridge_rna_atac.h5ad`
  4. **Mosaic**: single dataset where each cell randomly observes any subset of {RNA, ATAC, ADT} with controllable coverage per modality (default 60 % RNA, 50 % ATAC, 40 % ADT). → `mock_mosaic_multiome.h5ad`
  5. **Prediction**: training split with paired RNA+ATAC; evaluation split with RNA-only cells to test imputation. → `mock_prediction_rna_to_atac.h5ad`
  6. **Hierarchical**: three batches representing studies with different modality mixes and batch effects (e.g., Study A: RNA+ATAC, Study B: RNA+ADT, Study C: ATAC only) plus dataset labels. → `mock_hierarchical_multistudy.h5ad`
- Provide an in-memory factory (`generate_mock_dataset(problem_type: str, *, n_cells: int, seed: int, …) -> AnnData`) so unit tests can bypass disk IO and directly validate dataloaders/masks.
- Document CLI usage (`python scripts/make_mock_multimodal.py --problem bridge --out examples/mock/mock_bridge_rna_atac.h5ad --config-out configs/examples/mock_bridge.yaml`) and ensure default configs in `configs/examples/` reference the generated files.

## Milestones & acceptance

- Milestone 1: `remadom` import path compiles, unit tests covering encoders/decoders/PoE pass.
- Milestone 2: `python -m remadom.cli.train --config configs/minimal.yaml` runs end to end, produces checkpoints, and reports decreasing ELBO on toy data.
- Acceptance: Code passes lint/unit suite, documentation updated, smoke test instructions validated by a fresh checkout.

## Open questions

- Library-size handling: confirm whether to scale RNA decoder by observed libsize or by a learned offset when library sizes are missing.
- Registry persistence: decide if vocabularies should cache to disk for multi-run consistency during Phase 1 or defer to later phases.
- AMP defaults: choose between `bf16` and `fp16` as the default autocast dtype and document GPU requirements.
