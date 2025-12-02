# remadom: Multi-omic alignment and generative modeling

A self-contained guide to the remadom library: concepts, components, configuration, and workflows.

Version: M8.1

---

## 1) Executive summary

- What it is: remadom is a PyTorch library for learning shared embeddings across single-cell modalities (RNA, ATAC, ADT). It combines alignment losses (MMD, Sinkhorn OT, GW), graph regularization, and modality-aware decoders (RNA ZINB, ATAC Bernoulli, ADT Poisson mixture).
- Why it matters: Integrating heterogeneous datasets across batches and technologies requires scalable alignment with principled likelihoods. remadom delivers a modular, performant pipeline with AMP training, checkpointing, and reproducibility.
- Who it’s for: Computational biologists and ML researchers building multi-omic integration pipelines, benchmarking alignment methods, or training end-to-end generative models for imputation and downstream tasks.

---

## 2) Core ideas

- Shared latent representation: Each modality has an encoder producing a Gaussian posterior q(z|x). A Product-of-Experts fusion combines them into a shared biological latent z_bio.
- Alignment objectives: Optional heads impose distributional alignment across groups (batch/dataset/time) using MMD, Sinkhorn OT (differentiable), GW, cross-graph, and temporal order losses.
- Structural prior: A kNN graph (built via FAISS/HNSW/NumPy) yields a Laplacian penalty to preserve local geometry.
- Bridging signals: Mutual nearest neighbor (MNN) edges across batches/datasets inject sparse correspondences to stabilize alignment.
- Modality-appropriate decoders:
  - RNA: ZINB with library-size conditioning, flexible dispersion (gene/batch/cell).
  - ATAC: Bernoulli (binary peaks) decoder; or encoder-only with TF-IDF/LSI.
  - ADT: Poisson background/foreground mixture.

---

## 3) Architecture and components

- Data
  - preprocess_rna: counts normalization, optional HVG selection (Scanpy or variance heuristic).
  - preprocess_atac_tfidf_lsi: TF-IDF → LSI with scikit-learn; stores obsm["X_lsi"] and metadata for dimensions.
  - dataloader: selects matrices by key (X, layers:name, obsm:name); computes libsize_rna for RNA ZINB; attaches obs keys (batch/dataset/time).
- Modeling (PyTorch)
  - Encoders: simple MLPs per modality that output mu and logvar for q(z|x).
  - Fusion: ProductOfExperts with optional temperatures per modality (list or dict keyed by modality name).
  - Decoders:
    - RNADecoderZINB: predicts mu, theta (dispersion), and pi (zero-inflation), supporting library-size scaling and batch-conditional offsets/dispersion.
    - ATACDecoderBernoulli: predicts logits/probabilities for binary peak matrix; optional low-rank head for memory.
    - ADTMixtureDecoder: Poisson mixture (foreground/background) with learned mixing alpha.
  - VAE: MosaicVAE encapsulates encoders → fusion → decoders, with ELBO-style loss (reconstruction + KL).
- Alignment
  - MMDHead: kernel MMD across groups (e.g., batch).
  - Sinkhorn/OT: differentiable Sinkhorn divergence; optional bridge masking and schedules.
  - GW/Gromov–Wasserstein: aligns structures across modalities/datasets.
  - Temporal: couples OT with ordering/temporal constraints.
- Graph and bridges
  - AnnIndex: unified FAISS/HNSW/NumPy kNN backend.
  - GraphBuilder: builds kNN graph; symmetrizes and computes Laplacian.
  - GraphRegularizer: sparse Laplacian penalty; caches sparse tensors on device.
  - MNNBridge: mutual nearest neighbors across exactly two batches; ANN-backed and batched.
- Training
  - Trainer: AMP (fp16/bf16), early stopping, schedulers (cosine/step/plateau), gradient clipping, checkpointing (model/optimizer/scheduler), resuming.
  - Logging: CSV logs (optionally TensorBoard-ready).
  - Validation utilities: check dimension consistency before training.

---

## 4) Installation

- Base (editable install)
  - pip install -e .
- Recommended optional dependencies
  - Speed/ANN: pip install faiss-cpu hnswlib
  - OT on GPU: pip install geomloss pykeops
  - ATAC LSI: pip install scikit-learn
  - RNA HVG via Scanpy: pip install scanpy
- PyTorch with CUDA is recommended for GPU; AMP supports fp16 and bf16.

---

## 5) Quickstart

- Minimal YAML config:
  - Defines data path, preprocessing, model dims, alignment heads, structure regularizer, and optimization settings.
- Train from CLI:
  - python -m remadom.cli.train --config config.yaml
- Mock data workflow: see `docs/mock_data.md` for generator logic, CLI usage, and smoke-test datasets.
- Network design overview: refer to `docs/network_architecture.md` for a full description of modules, data flow, modalities, and loss structure.
- Problem type cheat sheet: `docs/problem_types.md` summarises masks, goals, and configuration hints for paired/unpaired/bridge/mosaic/prediction/hierarchical scenarios.

### 5.0) Phase status

- **Phase 1 – Core Mosaic Backbone:** ✅ complete (encoders/decoders, masked ELBO, dataloaders, trainer, mock data, CLI).
- **Phase 2 – Alignment & Bridge Integration:** ✅ complete (MNN/Seeded/Dictionary/Linear map bridges, schedules/diagnostics, CLI artefacts, docs/tests).

### 5.1) Outstanding Phase 2 follow-ups (per mock case)

- **mock_paired (Problem Type 1):** add paired reconstruction + cross-modal prediction metrics (per-modality RMSE/Pearson, imputation scatter), emit into `metrics.final.json`/plots, and document baseline expectations.
- **mock_unpaired (Problem Type 2):** enable at least one alignment head (MMD/GW) + evaluation step that checks RNA↔ATAC imputation or latent alignment quality (e.g., silhouette/batch variance).
- **mock_bridge (Problem Type 3):** extend bridge diagnostics with downstream quality checks—e.g., compute imputation accuracy on single-modality cohorts before/after bridge loss, assert non-zero edge counts across providers.
- **mock_mosaic (Problem Type 4):** implement mosaic masking stress tests (varying missing-rate schedules) and measure reconstruction/imputation against known ground truth; add batch-effect metrics when multiple cohorts exist.
- **mock_prediction:** create a held-out-target evaluation loop that predicts the missing modality and reports correlation/NRMSE; surface CLI command to export predicted matrices for inspection.
- **mock_hierarchical:** integrate hierarchical alignment components (graph or multi-level heads) plus quantitative batch-effect diagnostics that confirm per-study harmonisation.

### 5.2) Phase 3 outlook

- **Graph-regularised & temporal heads:** finish Laplacian/temporal adapters originally sketched for Phase 2 and wire them into configs/tests.
- **Evaluation harness:** ship reusable scripts/notebooks that compute SCIB-like metrics, cross-modal imputation scores, and bridge diagnostics for any run.
- **Adapters & reference mapping:** expose `remadom/adapters` via CLI (mapref, arches, mixtures) so external references can be injected, with smoke tests and docs.
- **Scalability polish:** upgrade AMP (torch.amp APIs), add CPU fallbacks, and profile dataloaders/checkpointing on >1e6 cells.
- **Phase 3 plan:** draft a dedicated roadmap once the outstanding Phase 2 follow-ups land to avoid scope creep.

### 5.3) Phase 1 mock workflow

- Generate synthetic data covering any problem type with the helper script:
  - `python scripts/make_mock_multimodal.py --problem mosaic --out examples/mock/mock_mosaic_multiome.h5ad --config-out configs/examples/mock_mosaic.yaml`
- Review the emitted config (or start from `configs/examples/mock_mosaic.yaml`) and adjust latent dimension, epochs, or modality-specific hidden sizes as needed.
- Launch training end-to-end:
  - `python -m remadom.cli.train --cfg configs/examples/mock_mosaic.yaml`
- Inspect outputs under `runs/<run_dir>`:
  - `checkpoint.last.pt` / `checkpoint.best.pt` (if validation enabled)
  - `train_history.json` with per-epoch train/val losses, reconstruction, KL, and beta
  - `config.resolved.yaml` capturing the exact experiment setup
- Run smoke tests locally (after installing dev dependencies):
  - `python -m pytest tests/unit/test_mock_generator.py`
  - `python -m pytest tests/integration/test_phase1_training.py`
- Outputs in runs/<run_dir>:
  - best.pt, last.pt (checkpoints with optimizer and scheduler)
  - train_log.csv (metrics per epoch/interval)
  - embeddings.npz (z_bio, mu, logvar)
  - obs_arrays.npz (optional batches/labels used for quick eval)

Example config:

- data.path: path/to.h5ad
- data.preprocess.rna.hvg: 2000
- data.preprocess.atac.lsi.enabled: true
- model.encoders.rna.in_dim: 2000
- model.decoders.rna.out_dim: 2000
- alignment.mmd.enabled: true
- structure.laplacian_enabled: true
- optim.epochs: 20, amp.enabled: true

---

## 6) Configuration reference

Use YAML or JSON. Key sections and fields:

- data
  - path: string (AnnData .h5ad)
  - batch_key: obs column for batches (default: batch)
  - dataset_key: obs column for dataset IDs (default: dataset)
  - time_key: obs column for temporal ordering (default: time)
  - preprocess:
    - rna:
      - hvg: int (e.g., 2000)
      - flavor: seurat_v3 (if using Scanpy)
    - atac.lsi:
      - enabled: bool
      - n_components: int (default: 50)
      - obsm_key: string (default: X_lsi)
- model
  - latent_bio: int (e.g., 16)
  - latent_tech: int (optional, default 0)
  - encoders:
    - rna.in_dim: int (e.g., HVG count)
    - atac.in_dim: int (if using LSI; inferred if preprocess.lsi.enabled)
    - adt.in_dim: int
  - decoders:
    - rna:
      - out_dim: int (must match RNA features)
      - dispersion: gene | batch | cell
      - conditional_batches: int (required for dispersion=batch; auto-inferred from obs if not set)
      - library: bool (use libsize scaling)
      - hidden: int
    - atac:
      - out_dim: int (if decoding peaks directly)
      - hidden: int
      - lowrank: int or null
    - adt:
      - out_dim: int
      - hidden: int
  - fusion_temps: list of floats or dict {modality: temp}
- alignment
  - mmd, sinkhorn, ot, gw, crossgraph, temporal:
    - enabled: bool
    - params: method-specific fields (e.g., epsilon, weight, group_key, etc.)
- structure
  - laplacian_enabled: bool
  - laplacian_lambda: float
  - laplacian_normalized: bool
  - graph_backend: faiss | hnsw | numpy
  - graph_metric: euclidean | cosine
- bridge
  - provider: mnn
  - refresh: int (epochs between rebuilds)
  - mnn: { k, metric, ann_backend }
- optim
  - epochs: int
  - batch_size: int
  - lr: float
  - weight_decay: float
  - grad_clip: float
  - detect_anomaly: bool
  - seed: int
  - scheduler:
    - name: none | cosine | step | plateau
    - step_size, gamma, T_max, patience, min_lr
  - amp:
    - enabled: bool
    - dtype: fp16 | bf16
  - early_stopping:
    - enabled: bool
    - monitor: metric key (e.g., elbo.loss or val.elbo.loss)
    - mode: min | max
    - patience: int
    - min_delta: float
- logging
  - run_dir: path for outputs
  - log_interval: int steps per CSV write
  - tensorboard: bool

---

## 7) Data expectations

- AnnData (.h5ad)
  - RNA
    - counts in layers["counts"] (preferred) or X; HVGs may be selected, updating var and X.
  - ATAC
    - If decoding peaks: binarized peak matrix in X or layers.
    - If encoder-only LSI: TF-IDF + LSI stored in obsm["X_lsi"] (or custom key), and encoders.atac.in_dim matches n_components.
  - ADT
    - counts in X or layers.
- obs
  - Columns for batch_key, dataset_key, time_key if used.
  - Batch categories are used to infer conditional_batches for dispersion=batch.

---

## 8) Workflows

- Alignment-only embedding
  - Enable MMD or Sinkhorn heads; optionally add graph regularization; disable decoders not needed.
- Full generative modeling
  - Enable RNA ZINB decoder (and others as needed); consider graph regularization and alignment heads for robust z_bio.
- ATAC LSI path
  - Enable data.preprocess.atac.lsi.enabled: true; set encoders.atac.in_dim to n_components (or let factory infer from metadata).
- Bridges and large N
  - Use FAISS/HNSW backends for kNN and MNN bridges; set refresh to rebuild bridges periodically.

---

## 9) Training features

- AMP: mixed precision with GradScaler; supports fp16 and bf16 (if hardware supports).
- Early stopping: patience and min_delta on a monitored key; prefers val.* metrics if a validation loader is provided.
- Schedulers: cosine, step, plateau (with ReduceLROnPlateau monitoring).
- Checkpointing: per-epoch last.pt and best.pt; stores model, optimizer, and scheduler; resume supported via CLI.
- Logging: train_log.csv with periodic aggregates; includes LR.

---

## 10) Validation and troubleshooting

- Dimension validation
  - validate_dims checks encoder/decoder dims against dataset matrices before training.
- Common issues
  - RNA dims mismatch after HVG: Ensure encoders.rna.in_dim and decoders.rna.out_dim equal the post-HVG gene count.
  - dispersion=batch with unknown batches: Provide model.decoders.rna.conditional_batches or ensure obs[batch_key] is categorical with correct labels.
  - ATAC LSI not found: Check preprocess settings; ensure scikit-learn installed; confirm obsm key exists; set encoders.atac.in_dim accordingly.
  - ADT data not counts: Current decoder expects nonnegative counts. For arcsinh-normalized data, use or add a Gaussian mixture decoder.
- Numerical stability
  - ZINB clamps mu/theta/pi internally and uses stabilized log terms.
  - Sinkhorn uses log-domain updates; GeomLoss back-end is available on GPU.

---

## 11) API overview

- Modeling
  - MosaicVAE
    - encode(batch) -> modality posteriors
    - fuse(enc_outs) -> z_bio, mu, logvar
    - decode(z, batch) -> modality params
    - elbo(batch) -> dict with loss, recon, kl
    - embed(batch) -> z_bio, mu, logvar
  - ProductOfExperts
    - forward(mus, logvars, modality_names=...) -> fused mu, logvar
  - Decoders
    - RNADecoderZINB.forward(z, libsize, batch_labels) -> mu, theta, pi
    - reconstruction_loss(x, params) -> scalar
    - ATACDecoderBernoulli.forward(z) -> logits, p
    - ADTMixtureDecoder.forward(z) -> mu_b, mu_f, alpha
- Alignment and structure
  - OtSolver.sinkhorn_divergence(X, Y, epsilon, ...) -> loss, info
  - GraphBuilder.build(emb, ...) -> SparseGraph {A, L, degrees}
  - GraphRegularizer.forward(z_bio, graph) -> loss, logs
  - MNNBridge.build(z_bio, batches, ...) -> BridgeEdges
- Training
  - Trainer.fit(train_loader, val_loader=None)
  - Checkpoints: utils.checkpoint.save_checkpoint/load_checkpoint
- Data
  - preprocess_rna, preprocess_atac_tfidf_lsi, preprocess_atac
  - dataloader(adata, cfg, x_keys=..., ...) -> Iterable[Batch]

---

## 12) Extending remadom

- New modality
  - Implement Encoder producing {"mu", "logvar"}.
  - Implement Decoder with forward(z) -> params and reconstruction_loss(x, params).
  - Register in factories and expose in config schema.
- Custom alignment head
  - Create a module with forward(z_bio, groups, graph, aux) -> (loss, logs).
  - Add a build_X_head function and configuration block.
- Alternative likelihoods
  - Swap in Negative Binomial without zero-inflation, zero-truncated variants, or continuous distributions as needed.

---

## 13) Reproducibility

- Seeding: cfg.optim.seed sets global seeds; DataLoaderLike uses NumPy shuffle for deterministic batches.
- Checkpoints: model/optimizer/scheduler state is saved; resume with --resume path.
- Logs: CSV logs at fixed intervals capture metrics, enabling reproducible plots.

---

## 14) Example end-to-end config

Minimal multi-omic run with RNA HVG and ATAC LSI:

- data:
  - path: my_data.h5ad
  - batch_key: batch
  - preprocess:
    - rna: { hvg: 2000, flavor: seurat_v3 }
    - atac:
      - lsi:
        - enabled: true
        - n_components: 50
        - obsm_key: X_lsi
- model:
  - latent_bio: 16
  - encoders:
    - rna: { in_dim: 2000 }
    - atac: { in_dim: 50 }
  - decoders:
    - rna: { out_dim: 2000, dispersion: batch, library: true }  # conditional_batches inferred
    - atac: { out_dim: 60000, lowrank: 64 }                     # if decoding peaks (optional)
  - fusion_temps:
    - rna: 1.0
    - atac: 1.5
- alignment:
  - mmd: { enabled: true, params: { weight: 0.1, group_key: batch } }
  - sinkhorn: { enabled: true, params: { epsilon: 0.05, weight: 0.05 } }
- structure:
  - laplacian_enabled: true
  - laplacian_lambda: 0.001
  - graph_backend: faiss
- bridge:
  - provider: mnn
  - refresh: 5
  - mnn: { k: 20, metric: euclidean, ann_backend: faiss }
- optim:
  - epochs: 20
  - batch_size: 512
  - lr: 0.001
  - scheduler: { name: cosine, T_max: 20, min_lr: 1.0e-6 }
  - amp: { enabled: true, dtype: bf16 }
  - early_stopping: { enabled: true, monitor: elbo.loss, mode: min, patience: 5, min_delta: 1.0e-4 }
- logging:
  - run_dir: runs/m8_demo
  - log_interval: 100

Train:

- python -m remadom.cli.train --config config.yaml

---

## 15) Performance notes

- ANN backends:
  - Prefer FAISS for large datasets; HNSW is a solid CPU alternative. remadom falls back to NumPy if neither is installed.
- Sparse ops:
  - Laplacian regularizer caches sparse tensors for repeated use on GPU.
- OT/Sinkhorn:
  - GeomLoss accelerates large problems; pykeops can further scale with GPU kernels.
- Mixed precision:
  - bf16 is often more numerically stable than fp16 on Ampere and newer GPUs.

---

## 16) Security and data handling

- remadom operates on local files and in-memory tensors. It does not fetch remote resources unless your code does.
- Ensure your AnnData does not contain sensitive metadata if sharing outputs (obs arrays may contain identifiers).

---

## 17) Roadmap

- M9: Evaluation and visualization suite: integration with scanpy plots, UMAP/TSNE exports, and HTML reports.
- M10: Unit tests and CI; coverage for decoders, alignment heads, and data loaders.
- M11: Advanced decoders (e.g., ZINB with gene-wise covariates; Gaussian for normalized ADT), partial OT, and domain-adversarial alternatives.
- M12: Distributed training and large-batch scalability.

---

## 18) FAQ

- Can I use remadom for RNA-only integration?
  - Yes. Enable only the RNA encoder/decoder if desired; alignment heads can still operate across batches.
- Do I need counts for RNA?
  - For ZINB, yes—use layers["counts"]. The dataloader computes library-size from inputs; non-integer inputs may degrade performance.
- How do I add a new alignment loss?
  - Implement AlignmentHead.forward, add a factory registration, and expose it via config.
- Does Sinkhorn backpropagate?
  - Yes, by default. Wrap calls in torch.no_grad if you need non-differentiable diagnostics.

---

## 19) Acknowledgments

- FAISS, hnswlib for ANN.
- GeomLoss and KeOps for OT on GPU.
- Scanpy and scikit-learn for preprocessing utilities.

---

If you’d like, I can also:

- Generate a shorter “Quickstart” README focused on installation and a single working example.
- Provide a tutorial notebook outline (cells for preprocessing, config, training, and evaluation).
- Add a rendered HTML guide with collapsible sections for configs and code snippets.
