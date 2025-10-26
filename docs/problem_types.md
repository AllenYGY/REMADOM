# REMADOM Problem Types and Training Strategies

This note revisits the six canonical scenarios, summarising data assumptions, training goals, and configuration hints for each. All cases share the same MosaicVAE backbone—only the data masks and optional alignment heads differ.

## 1. Paired

- **Definition:** every cell is observed in multiple modalities (e.g., CITE-seq, 10x Multiome). Masks `has_rna`, `has_atac`, `has_adt` are mostly `True`.
- **Goal:** learn a joint latent embedding; denoise and reconstruct each modality, enable cross-modal imputation.
- **Training setup:**
  - Data: single AnnData where all modalities are present per cell.
  - Config: disable alignment heads unless batches need correction; typical YAML already works.
  - Loss: masked ELBO reduces to the usual multivariate VAE objective.
- **Mock data:** `problem=paired` via `scripts/make_mock_multimodal.py`.

## 2. Unpaired

- **Definition:** different cell cohorts measure different modalities (e.g., RNA-only dataset A, ATAC-only dataset B). No shared cells.
- **Goal:** align the latent spaces so RNA and ATAC embeddings become comparable; optionally infer cross-modal mappings.
- **Training setup:**
  - Data: combine cohorts in one AnnData; set `has_rna`/`has_atac` masks accordingly.
  - Config: enable alignment heads (MMD, Sinkhorn/GW) keyed on `batch` or `dataset` labels to force global agreement.
  - Loss: masked recon terms handle each modality separately; alignment head penalties bridge the gap.
- **Mock data:** `problem=unpaired`.

## 3. Bridge

- **Definition:** large single-modality datasets plus a small paired “bridge” subset. Only some cells have both modalities.
- **Goal:** use the bridge to transfer information between single-modality cohorts and recover missing modalities.
- **Training setup:**
  - Data: same AnnData with masks; bridge cells have multiple `has_<mod>` set to `True`.
  - Config: optionally enable bridge-specific heads (e.g., MNN-based) and alignment terms to reinforce the bridge constraints (see `docs/bridge_mnn.md`).
- **Mock data:** `problem=bridge`.

## 4. Mosaic

- **Definition:** within one dataset, each cell may have any combination of modalities (irregular missingness).
- **Goal:** learn a robust shared latent space and impute missing modalities while respecting uncertainty.
- **Training setup:**
  - Data: `has_<mod>` masks vary per cell.
  - Config: alignment heads optional; masked ELBO naturally handles arbitrary patterns.
- **Mock data:** `problem=mosaic`.

## 5. Prediction

- **Definition:** training data is paired, but at inference we only observe source modality (e.g., predict ATAC from RNA).
- **Goal:** train as paired; during inference feed only the available modality and decode the missing one via `impute()`.
- **Training setup:**
  - Train identically to `paired`.
  - For evaluation/inference, set the target modality mask to `False` and call `model.impute`.
- **Mock data:** `problem=prediction`.

## 6. Hierarchical

- **Definition:** multiple studies/batches with different modality mixes and technical effects.
- **Goal:** unify datasets across studies, correcting both modality gaps and batch effects.
- **Training setup:**
  - Data: include `batch`/`dataset` labels; masks reflect available modalities per study.
  - Config: combine alignment heads, graph/bridge regularisers, possibly adapters; consider `z_bio`/`z_tech` disentangling if adding later modules.
- **Mock data:** `problem=hierarchical`.

## Cross-cutting notes

- **Masked ELBO:** handles all missing-data patterns automatically; only observed modalities contribute to reconstruction terms.
- **Alignment heads:** (MMD, Sinkhorn, GW, etc.) can be toggled in YAML for any problem type to remove batch effects or improve cross-domain alignment.
- **Adapters/bridges:** advanced modules (`remadom/adapters`, `remadom/bridges`) extend Phase 1 for reference mapping or explicit bridge constraints.
- **Mock generator:** see `docs/mock_data.md` for CLI examples and dataset generation scripts.
