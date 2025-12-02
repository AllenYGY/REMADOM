# REMADOM Network Architecture

This note summarises the Phase 1 architecture: data flow, core modules, losses, and how the framework adapts to different multi-omic scenarios.

## 1. High-level pipeline

1. **Configuration**  
   - All experiments are driven by YAML. `remadom/config/schema.py` defines the schema; `factories.py` translates configs into models, alignment heads, optimisers, schedules, data loaders, etc.

2. **Data ingestion**  
   - `remadom/data/loaders.build_dataloaders` opens AnnData files, aligns features via registries, and yields `Batch` objects.  
   - Each `Batch` carries per-modality tensors (`x_rna`, `x_atac`, `x_adt`), boolean masks (`has_<mod>`), library sizes, batch/dataset labels, and optional metadata.

3. **Core model (MosaicVAE)**  
   - Encoders: modality-specific MLPs (`RNAEncoder`, `ATACEncoder`, `ADTEncoder`) map inputs to Gaussian posterior parameters (`mu`, `logvar`).  
   - Fusion: a Product-of-Experts combines available posteriors into a shared latent `z_bio`. Missing modalities are skipped via `has_<mod>` masks.  
   - Reparameterisation: draws latent samples, or uses posterior means at evaluation.  
   - Decoders: modality-specific heads reconstruct observations with appropriate likelihoods—ZINB (RNA), Bernoulli (ATAC), Gaussian/Log-normal (ADT).  
   - Masked ELBO: only modalities present in the batch contribute to the reconstruction loss (weighted averages). KL uses the fused posterior vs. unit Gaussian.

4. **Alignment / regularisation heads (optional)**  
   - Modules in `remadom/align` impose extra constraints on `z_bio`: MMD, Sinkhorn OT, GW (Gromov-Wasserstein), cross-graph, temporal, etc.  
   - Heads are configurable per run and can have schedules (e.g., ramp up Sinkhorn epsilon).

5. **Bridges & graph regularisers (optional)**  
   - `remadom/bridges` provides mutual-nearest-neighbour and other bridge sampling utilities. Phase 2 adds the `BridgeHead`, which turns those edges into latent penalties and supports scheduled weights via the `bridge` config block.  
   - CLI runs emit bridge diagnostics (`bridge_metrics.json`) capturing edge counts and degree statistics for sanity checks.  
   - `remadom/graph` builds kNN graphs and applies Laplacian penalties to preserve local structure.

6. **Training loop**  
   - `remadom/train/trainer.py` orchestrates optimisation: mixed precision (fp16/bf16), GradScaler, gradient clipping, β/weight/head schedules, per-head metrics, logging, checkpointing.  
   - CLI entry point (`remadom.cli.train`) wires configs, loaders, model, trainer, writes history plus `metrics.final.json`, optional bridge diagnostics, and saves outputs under `runs/<run_dir>`.

7. **Evaluation / utilities (Phase 2+)**  
   - Stubs exist for adapters, evaluators, autotuning; Phase 1 focuses on the backbone but keeps module boundaries ready for expansion.

## 2. Data modalities & expectations

| Modality | Encoder | Decoder | Input format | Reconstruction likelihood |
|----------|---------|---------|--------------|---------------------------|
| RNA      | `RNAEncoder` (`remadom/core/encoders.py`) | `RNADecoderZINB` (`remadom/core/decoders.py`) | Non-negative continuous values (typically UMI counts or normalised floats) | Zero-inflated Negative Binomial (ZINB) with optional library-size scaling |
| ATAC     | `ATACEncoder` | `ATACDecoderBernoulli` | Binary peak matrix or continuous LSI projection | Bernoulli (binary cross entropy with logits) |
| ADT      | `ADTEncoder` | `ADTMixtureDecoder` | Continuous non-negative values (protein abundances) | Gaussian-style NLL (mean + log variance) |

Masks (`has_rna`, `has_atac`, `has_adt`) indicate whether a cell possesses a modality; losses only accumulate over `True` entries. Missing modalities neither add zeros nor gradients—they’re simply omitted.

## 3. Loss landscape

Per batch, the trainer aggregates:

1. **Masked reconstruction loss**  
   - For each modality, compute negative log-likelihood using the decoder’s distribution.  
   - Apply user-configurable weights per modality (`cfg.model.decoders[mod].weight`).  
   - Average over the sum of weights for observed modalities.

2. **KL divergence**  
   - Analytical KL between `N(mu, logvar)` and the unit Gaussian prior.  
   - Scaled by β (either fixed or scheduled).

3. **Alignment / regularisation terms** (optional)  
   - Alignment heads (MMD, Sinkhorn, GW, etc.) act on `z_bio` with their own weights/schedules.  
   - Graph Laplacian penalties, bridge losses, or other heads add to the total loss when enabled.

Pseudo-formula:

```
L_total = Σ_mod w_mod * NLL_mod(has_mod ⊙ x_mod, decoder_mod(z)) / Σ_mod w_mod  +  β * KL(q(z|x) || p(z))  +  Σ_heads λ_head * L_head(z, meta)
```

## 4. Problem type adaptation

The architecture doesn’t branch per problem type; instead, data masks and configuration govern behaviour:

- **Paired**: All `has_<mod>` ≈ True, minimal alignment heads.  
- **Unpaired**: Cohorts lack complementary modalities; alignment heads ensure shared latent structure.  
- **Bridge**: Most cells single-modality, a small paired “bridge” subset; leverages PoE + bridge heads.  
- **Mosaic**: Random modality availability per cell; masked ELBO handles arbitrary combinations.  
- **Prediction**: Train on paired data, predict by masking target modality (`has_target=False`) at inference.  
- **Hierarchical**: Multiple studies/batches with different modality mixes; combine alignment heads, graph regularisers, batch metadata.

Mock datasets in `examples/mock/` mimic each scenario for testing and benchmarking.

## 5. Adapters & future extensions

`remadom/adapters` hosts preliminary components for reference mapping / residual adaptation (e.g., scArches-like adapters, mixture mappers). While Phase 1 doesn’t require them, the package layout anticipates future work on transferring new datasets into pre-trained reference spaces.

## 6. Key takeaways

- **Flexible backbone**: Mosaic-first VAE with masked ELBO and PoE fusion handles any combination of modalities.  
- **Config-driven**: Alignment, bridges, graph regularisers, and schedules are toggled via YAML—no code edits required per scenario.  
- **Modary-aware**: RNA, ATAC, ADT get specialised likelihoods, ensuring realistic reconstruction and imputation.  
- **Extensible**: Adapters, evaluation, autotuning modules are ready for Phase 2+, making the system future-proof for reference mapping and automated benchmarking.
