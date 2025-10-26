Below is a detailed, production-ready program structure for the project. It lays out directories, key modules, public APIs, and what each file is responsible for. It’s organized to support the preservation-first backbone, bridge-based and distribution-based alignment, evaluators, and CLI. You can scaffold this tree and progressively fill in components per milestone.

Implementation phases

- Phase 0 — Foundation (complete): finalize project naming (remadom), background docs, and high-level architecture; capture requirements in `docs/background.md` and `docs/PLAN.md`; agree on YAML-first configuration strategy.
- Phase 1 — Core mosaic backbone (current focus): implement `remadom/core` MosaicVAE with encoders, decoders, fusion, masked ELBO; build minimal data loaders and preprocessing to feed RNA/ATAC/ADT modalities; deliver a basic training loop with checkpoints and AMP.
- Phase 2 — Alignment and bridges (next): ship `remadom/align` heads (MMD, Sinkhorn, GW) with schedules; add `remadom/bridges` providers and `remadom/graph` Laplacian regularizers; expose controls via config and ensure loss logging.
- Phase 3 — Evaluation and CLI (upcoming): finish evaluator metrics, autotuner, and bridge diagnostics; provide CLI entry points for train/embed/impute/eval; package example configs and notebooks.
- Phase 4 — Hardening and extensions (future): expand adapters (reference mapping), add temporal heads, polish docs (mkdocs or sphinx), introduce CI tests, and prepare release artifacts.

Project layout

- README.md
- pyproject.toml or setup.cfg + setup.py
- requirements.txt
- .gitignore
- .pre-commit-config.yaml
- mkdocs.yml or docs/ Sphinx config
- CHANGELOG.md
- CODEOWNERS
- CONTRIBUTING.md

Top-level packages

- remadom/                # Python package
- scripts/               # Convenience scripts (thin wrappers around CLI)
- configs/               # YAML configs (profiles, examples)
- examples/              # Notebooks and small datasets adapters
- tests/                 # Unit, integration, regression tests
- docs/                  # Documentation sources (mkdocs or sphinx)
- data_registry/         # Optional: small metadata, vocabularies

remadom/ package structure

- remadom/__init__.py
- remadom/version.py
- remadom/typing.py                # Common type aliases, Protocols
- remadom/utils/                   # Utilities and helpers
  - __init__.py
  - logging.py                    # JSONL logger, W&B/MLflow adapters
  - seed.py                       # Reproducible seeds
  - timers.py                     # Scoped timers and epoch budgets
  - metrics_utils.py              # Shared helpers for metrics
  - numpy_torch.py                # Conversions, device moves
  - checkpoint.py                 # Save/load checkpoints, artifact mgmt
  - registry.py                   # Component factories, entry-point registry
  - schedules.py                  # KL warmup, ε schedules, cosine anneal
  - validation.py                 # Guardrails for config and inputs
  - serialization.py              # Config and artifact I/O
  - math.py                       # Stable math (logsumexp, etc.)
  - sparse.py                     # Sparse ops helpers (Laplacian)
- remadom/config/                  # Typed configs and factories
  - __init__.py
  - schema.py                     # Pydantic models for all sections
  - defaults.py                   # Default values and profiles
  - resolve.py                    # Merge, override, validate; frozen config
  - factories.py                  # Build model/heads/bridges/optim from cfg
- remadom/data/                    # Data loading, preprocessing, registries
  - __init__.py
  - loaders.py                    # AnnData loaders, minibatch iterators
  - preprocess.py                 # RNA/ATAC/ADT preprocessing pipelines
  - registries.py                 # Feature registries (genes, peaks, proteins)
  - splits.py                     # Train/val splits, stratification
  - batching.py                   # Collate, mask creation, padding
  - adapters.py                   # AnnData adapters and shims
- remadom/core/                    # Backbone, encoders/decoders, fusion
  - __init__.py
  - vae.py                        # MosaicVAE backbone
  - encoders.py                   # Modality encoders
  - decoders.py                   # Modality decoders and likelihoods
  - fusion.py                     # Product-of-Experts (PoE), alternatives
  - losses.py                     # ELBO components, NLLs, KLs
  - disentangle.py                # z_bio/z_tech regularizers
  - inference.py                  # embed, embed_np, impute utilities
- remadom/align/                   # Alignment heads and solvers
  - __init__.py
  - base.py                       # AlignmentHead base class
  - mmd.py                        # MMDHead
  - sinkhorn.py                   # SinkhornHead (GeomLoss)
  - ot.py                         # OTHead (sliced/unbalanced/partial)
  - gw.py                         # GWHead (sliced FGW/GW)
  - crossgraph.py                 # CrossGraphHead (bridge Laplacian)
  - temporal.py                   # TemporalHead (time OT + ordering)
  - solvers/                      # Thin wrappers to deps
    - __init__.py
    - ann.py                      # AnnIndex (FAISS/HNSW)
    - ot_solver.py                # POT/GeomLoss wrappers
    - gw_solver.py                # Sliced GW/FGW utilities
- remadom/graph/                   # Neighborhood graphs, Laplacians
  - __init__.py
  - builder.py                    # GraphBuilder (kNN, spatial)
  - laplacian.py                  # GraphRegularizer
  - spatial.py                    # Spatial graph helpers (Squidpy integration)
- remadom/bridges/                 # Bridge providers and utilities
  - __init__.py
  - base.py                       # BridgeProvider, BridgeEdges
  - mnn.py                        # MNNBridge
  - anchors.py                    # AnchorBridge
  - seeded.py                     # SeededBridge
  - utils.py                      # Bridge masks, diagnostics, degree caps
- remadom/adapters/                # Reference mapping mechanisms
  - __init__.py
  - arches.py                     # Residual Adapter (scArches-like)
  - mixtures.py                   # MixtureAdapter (Symphony-like)
  - bridge_head.py                # Low-rank bridge basis penalty
- remadom/train/                   # Training orchestration
  - __init__.py
  - trainer.py                    # Trainer, schedules, budgets
  - loop.py                       # Epoch/step loops, callbacks
  - callbacks.py                  # KL warmup, ε schedule, early stop, checkpoints
  - optim.py                      # Optimizer and schedulers creation
  - state.py                      # Training state, RNG snapshots
- remadom/eval/                    # Metrics, evaluator, autotuner
  - __init__.py
  - metrics.py                    # iLISI, kBET, trustworthiness, purity, FOSCTTM
  - batch_cls.py                  # Batch classifier AUC
  - align_costs.py                # MMD/Sinkhorn/OT/GW cost trackers
  - bridge_metrics.py             # Coverage, symmetry, degree stats
  - evaluator.py                  # Evaluator: compute_metrics, diagnose, suggest
  - autotune.py                   # AutoTuner (bounded adjustments)
  - plots.py                      # Optional plots (if needed)
  - harness.py                    # Reproducible eval runs
- remadom/cli/                     # Command-line tools
  - __init__.py
  - train.py                      # remadom-train
  - embed.py                      # remadom-embed
  - impute.py                     # remadom-impute
  - mapref.py                     # remadom-mapref
  - eval.py                       # remadom-eval
  - autotune.py                   # remadom-autotune

Key modules: responsibilities and public APIs

remadom/config

- schema.py
  - DataConfig, ModelConfig, FusionConfig, StructureConfig, AlignmentConfig, BridgeConfig, OptimConfig, EvalConfig, AutoTuneConfig, LoggingConfig, ExperimentConfig
- defaults.py
  - Profiles: conservation_safe, balanced, aggressive_align
- resolve.py
  - resolve_config(overrides: List[str]) -> ExperimentConfig
- factories.py
  - build_model(cfg) -> MosaicVAE
  - build_heads(cfg) -> List[AlignmentHead]
  - build_bridge(cfg) -> Optional[BridgeProvider]
  - build_graph(cfg) -> Optional[GraphBuilder], Optional[GraphRegularizer]
  - build_optimizer(cfg, model) -> Optimizer, Scheduler

remadom/data

- loaders.py
  - load_anndata(path) -> AnnData
  - dataloader(adata, cfg) -> Iterable[Batch]
- preprocess.py
  - preprocess_rna(adata, cfg) -> AnnData
  - preprocess_atac(adata, cfg) -> AnnData
  - preprocess_adt(adata, cfg) -> AnnData
- registries.py
  - save_registry(...), load_registry(...)
- batching.py
  - collate(batch) -> Batch

remadom/core

- vae.py
  - class MosaicVAE(nn.Module)
    - encode, fuse_posteriors, reparameterize
    - decode_all, elbo
    - embed, embed_np, impute
- encoders.py
  - RNAEncoder, ATACEncoder (LSI-in), ADTEncoder
  - TechEncoder variants
- decoders.py
  - RNADecoderZINB/NB, ADTMixtureDecoder, ATACDecoderBernoulli/Poisson
  - nll implementations
- fusion.py
  - ProductOfExperts, MeanOfExperts (optional)
- losses.py
  - kl_gaussian, nll_nb, nll_zinb, nll_bernoulli, nll_poisson
- disentangle.py
  - mmd_zbio_batch, info_bottleneck_ztech
- inference.py
  - embed_dataset(model, loader) -> arrays
  - impute_dataset(...)

remadom/bridges

- base.py
  - class BridgeProvider
  - @dataclass BridgeEdges
- mnn.py
  - class MNNBridge(BridgeProvider)
- anchors.py
  - class AnchorBridge(BridgeProvider)
- seeded.py
  - class SeededBridge(BridgeProvider)
- utils.py
  - bridge_to_mask(bridges, groups) -> np.ndarray
  - diagnostics(BridgeEdges, N) -> dict

remadom/align

- base.py
  - class AlignmentHead(nn.Module)
- mmd.py
  - class MMDHead(AlignmentHead)
- sinkhorn.py
  - class SinkhornHead(AlignmentHead)
- ot.py
  - class OTHead(AlignmentHead)
- gw.py
  - class GWHead(AlignmentHead)
- crossgraph.py
  - class CrossGraphHead(AlignmentHead)
- temporal.py
  - class TemporalHead(AlignmentHead)
- solvers/ann.py
  - class AnnIndex
- solvers/ot_solver.py
  - class OtSolver
- solvers/gw_solver.py
  - class GwSolver

remadom/graph

- builder.py
  - class GraphBuilder
- laplacian.py
  - class GraphRegularizer
- spatial.py
  - spatial_knn, spatial_metrics helpers

remadom/adapters

- arches.py
  - class Adapter(nn.Module)
- mixtures.py
  - class MixtureAdapter(nn.Module)
- bridge_head.py
  - class BridgeHead(nn.Module)

remadom/train

- trainer.py
  - class Trainer
    - _maybe_build_bridges(...)
    - train_step(...)
    - evaluate_step(...)
    - fit(...)
- loop.py
  - epoch loops, gradient steps, hooks
- callbacks.py
  - KL warmup, epsilon schedule, early stopping, checkpoint saver
- optim.py
  - make_optimizer, make_scheduler
- state.py
  - TrainingState dataclass

remadom/eval

- metrics.py
  - ilis i/kBET wrappers, trustworthiness, continuity, purity, FOSCTTM, neighborhood overlap
- batch_cls.py
  - train_batch_classifier(z_bio, batches) -> AUC
- align_costs.py
  - compute_mmd_cost, sinkhorn_cost, ot_cost, gw_cost, coupling_entropy
- bridge_metrics.py
  - coverage, symmetry, degree stats
- evaluator.py
  - class Evaluator
  - class EvalReport
- autotune.py
  - class AutoTuner
- plots.py
  - optional plotting utilities
- harness.py
  - standardized evaluation runner

remadom/cli

- train.py
  - cli_train(cfg_path, overrides)
- embed.py
  - cli_embed(cfg_path, checkpoint, output)
- impute.py
  - cli_impute(cfg_path, checkpoint, modalities, output, nsamples)
- mapref.py
  - cli_mapref(cfg_path, checkpoint_ref, checkpoint_query, output)
- eval.py
  - cli_eval(cfg_path, checkpoint, baseline, report_out)
- autotune.py
  - cli_autotune(cfg_path, budget_epochs)

configs/

- profiles/
  - conservation_safe.yaml
  - balanced.yaml
  - aggressive_align.yaml
- examples/
  - rna_multibatch_bridges_only.yaml
  - rna_multibatch_hybrid.yaml
  - rna_atac_unpaired_fgwsliced.yaml
  - reference_mapping_rna.yaml
  - temporal_hybrid.yaml

examples/

- 01_rna_multibatch_bridges_only.ipynb
- 02_rna_multibatch_hybrid_sinkhorn.ipynb
- 03_unpaired_rna_atac_fgwsliced.ipynb
- 04_reference_mapping_adapters.ipynb
- 05_temporal_alignment.ipynb

tests/ (indicative)

- unit/
  - test_losses.py
  - test_poe.py
  - test_mmd.py
  - test_sinkhorn.py
  - test_ot.py
  - test_gw.py
  - test_laplacian.py
  - test_mnn_bridge.py
  - test_crossgraph.py
  - test_enc_dec_shapes.py
- integration/
  - test_rna_multibatch_bridges.py
  - test_rna_multibatch_hybrid.py
  - test_unpaired_rna_atac_fgwsliced.py
  - test_reference_mapping.py
- regression/
  - test_regression_metrics.py

docs/

- index.md
- concepts/
  - preservation_first.md
  - bridges_vs_distribution.md
  - z_bio_contracts.md
- guides/
  - quickstart.md
  - configuration.md
  - evaluation_and_autotuning.md
  - performance.md
  - troubleshooting.md
  - safety_and_risk.md
- api/
  - auto-generated from docstrings

Public API surface (import paths)

- from remadom.config import resolve_config, factories
- from remadom.core.vae import MosaicVAE
- from remadom.align import MMDHead, SinkhornHead, OTHead, GWHead, CrossGraphHead, TemporalHead
- from remadom.bridges import MNNBridge, AnchorBridge, SeededBridge
- from remadom.graph import GraphBuilder, GraphRegularizer
- from remadom.train.trainer import Trainer
- from remadom.eval.evaluator import Evaluator, EvalReport
- from remadom.eval.autotune import AutoTuner
- from remadom.cli.train import cli_train
- from remadom.cli.embed import cli_embed
- from remadom.cli.eval import cli_eval

Execution flow overview

- CLI/Script
  - Parse config -> resolve_config -> factories.build_* -> Trainer.fit
- Training loop
  - ELBO -> Graph build -> Bridge refresh (K epochs) -> Heads losses -> Sum -> Step -> Log
- Evaluation
  - embed_np -> Evaluator.compute_metrics -> diagnose/suggest -> reporting
- AutoTuning
  - Warm-up -> Evaluate -> Update cfg within bounds -> Short retrain -> Repeat

Dependency strategy

- Hard deps: torch, numpy, scipy, anndata/scanpy, pydantic, omegaconf
- Optional accel: faiss or hnswlib; geomloss (+keops if available); pot; squidpy
- Capability checks in factories; graceful fallbacks; warnings when disabling features

Telemetry and artifacts

- Run directory structure:
  - runs/<timestamp_or_runid>/
    - config.resolved.yaml
    - checkpoints/*.pt
    - logs/*.jsonl
    - metrics/*.json
    - bridges/*.npz (edges, weights, diagnostics)
    - graphs/*.npz
    - eval/report.json
    - model_card.md

Security and robustness

- Config guardrails: disallow aggressive profiles without explicit flag
- Numerical guards: ε floors, NaN checks, automatic retries
- Reproducibility: seed logging, deterministic settings option, checkpoint versioning

This structure gives you a maintainable, testable, and extensible codebase that treats bridges and distribution-based alignment as first-class, composable components, while keeping the preservation-first backbone central and robust.
