# Bridge Integration and MNN Edges

## Recap: Bridge scenario
- Two large cohorts, each measured in a single modality (e.g. RNA-only vs ATAC-only).
- A small “bridge” subset with paired modalities.
- Goal: transfer information across cohorts, align latent spaces, and enable cross-modal imputation.

## Role of MNN bridge edges
1. **Identify correspondence**: in latent space (`z_bio`), find mutual nearest neighbours between cohorts via `remadom/bridges/mnn.py`. This captures which single-modality cells are closest to bridge samples.
2. **Supply constraints**: the resulting `BridgeEdges` (pairs + optional weights) are fed to alignment modules (e.g., a bridge-specific head or graph regulariser) so the optimiser explicitly pulls these pairs together.
3. **Enhance knowledge transfer**: with edges, the model knows which cells from cohort A should align with cohort B, beyond what standard reconstruction provides.

## How it fits into training
- MosaicVAE still handles masking and reconstruction for available modalities.
- When the bridge provider is enabled (YAML configuration), the trainer calls `bridge.build(z, groups)` each epoch.
- The returned edges become additional penalties/loss terms (e.g., L2 distances, Laplacian constraints) that encourage latent overlap for matched pairs.
- Without edges, the model relies solely on PoE/masked ELBO; with edges, alignment is more explicit and often converges faster, especially with scarce bridge cells.

## Config snippet (example)
```yaml
bridge:
  provider: mnn
  mnn:
    k: 20
    metric: euclidean
alignment:
  mmd: { enabled: true, weight: 0.1, group_key: batch }
```
- This enables MNN bridge edges and MMD head simultaneously.

## Takeaways
- Bridge edges provide explicit cross-cohort correspondences.
- They improve the robustness of bridge integration, especially when the paired subset is tiny.
- Combined with alignment heads, they help achieve consistent latents and better cross-modal completion.
