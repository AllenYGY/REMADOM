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

## Bridge providers overview

### MNNBridge

- **输入**：整个 batch 的 latent `Z` 和 cohort 标签（通常是 `dataset`）。  
- **算法**：对不同 cohort 各取一组细胞，计算 pairwise 距离； A→B 和 B→A 分别选出 k 个最近邻，只保留“互为最近邻”的配对。  
- **输出**：这些互为近邻的配对组成 `BridgeEdges`，后续 `BridgeHead` 计算距离损失。  
- **使用场景**：没有任何先验配对信息时的默认选择，完全依赖当前 latent 的相似度。

### SeededBridge

- **输入**：`params.seed_pairs`（例如 `[[0,0],[1,3]]`）表示已知的 A/B 索引配对，`radius` 可选。  
- **算法**：先把种子对直接当作桥接边；若 `radius>0`，还会在种子周围找到最近邻，形成邻域内的全部配对并去重。  
- **输出**：结合种子与邻域扩散后的索引对。  
- **使用场景**：有桥接子集或人工标注的配对时，用来把先验关系扩散到更大范围。

### DictionaryBridgeProvider

- **输入**：隐含的桥接子集（默认取两个 cohort 中样本数最小的部分，也可通过 `bridge_size` 限定）以及参数 `lam`。  
- **算法**：在桥接子集上拟合线性映射 \( W = (A^\top A + \lambda I)^{-1} A^\top B \)，再把 cohort A 所有 latent 映射到 B 空间，对每个映射后的向量找最近的 B，生成配对并去重。  
- **输出**：映射 + 最近邻得到的 `BridgeEdges`。  
- **使用场景**：相信两域之间存在线性关系，且有桥接子集可用来拟合映射。

### LinearMapBridge

- **输入**：同 DictionaryBridge，但接口更精简（`params.bridge_size`、`params.lam`）。  
- **算法**：同样拟合线性映射，但实现更轻量；直接用映射后的最近邻作为边。  
- **区别**：与 DictionaryBridge 在效果上类似，只是参数和代码路径更直接，适合想快速试验线性映射的情况。

所有 provider 返回的都是 `BridgeEdges`（`src_idx`、`dst_idx`、可选 `weight`），`BridgeHead` 会据此计算 latent 对之间的距离、写入诊断（边数、度数、桥接对等）。

## Configuration example

```yaml
bridge:
  enabled: true
  method: mnn         # or seeded/dictionary/linmap
  group_key: dataset
  weight: 0.5
  params:
    k: 15             # (mnn) neighbors
    seed_pairs: [[0,0],[5,12]]  # (seeded)
    radius: 3         # (seeded) neighbor expansion
    lam: 0.01         # (dictionary/linmap) ridge penalty
  schedule:
    kind: linear
    start: 0.0
    end: 0.5
  pairs: [[0,1]]      # optional cohort pairs
  normalize: false
  max_edges: 500
```
