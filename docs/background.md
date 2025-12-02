# REMADOM 简介

REMADOM 是一个面向单细胞多模态整合的统一框架，支持以下典型场景：

- **配对 (Paired)**：同一批细胞同时测量多种模态，如 CITE-seq、Multiome。
- **非配对 (Unpaired)**：不同细胞群分别测量不同模态，如 RNA-only vs. ATAC-only。
- **桥接 (Bridge)**：少量“桥梁”细胞拥有多模态，用来连接两个大规模的单模态数据集。
- **马赛克 (Mosaic)**：同一数据集中，每个细胞拥有的模态组合随意缺失。
- **预测 (Prediction)**：用一种模态推断另一种难测模态，如 RNA→ATAC 预测。
- **分层 (Hierarchical)**：跨多个研究/实验/组织的异构数据集，需要分层对齐。

|**Problem Type**|**Core idea**|**Typical scenario**|**Input**|**Goal**|**Challenges / Methods**|
|---|---|---|---|---|---|
|Paired|Same cells have multiple modalities|CITE-seq, Multiome|Single dataset with all modalities per cell|Joint embedding, denoising, imputation|totalVI, MultiVI, MOFA+, LIGER, WNN|
|Unpaired|Different cohorts measure different modalities|RNA-only vs ATAC-only cohorts|Separate datasets per modality|Aligned embedding, match function|Optimal transport (MOSAIC), graph methods (GLUE), adversarial alignment (bindSC)|
|Bridge|Few multi-modal “bridge” cells connect large single-modal cohorts|Small paired subset, large single-modal sets|Dataset A, dataset B, plus small A+B bridge|Unified embedding, cross-modal completion|Use bridge subset to learn mapping (Seurat v5, UINMF) and generalize|
|Mosaic|Each cell may have arbitrary modality combinations|Large atlas with missing modalities|Single dataset with irregular masks|Joint embedding, completion with uncertainty|Must handle arbitrary missingness (MultiVI, MIDAS, StabMap)|
|Prediction|One modality available, predict another|Predict ATAC from RNA|Paired training set, single-modality inference set|Cross-modal prediction|Accuracy + interpretability (sciPENN, BABEL)|
|Hierarchical|Heterogeneous studies, labs, tissues|Large atlas spanning multiple studies|Multiple datasets with different modality mixes|Global embedding + structured alignment|Hierarchical factorization (LIGER) or clustering (scMerge2)|

`remadom` 是一个统一、模块化、易扩展的多组学整合框架，主要特点包括：

1. **Mosaic-first 骨干**：核心模型 `REMADOM` 通过 Masked ELBO 支持任意模态缺失，PoE 融合多模态潜向量。
2. **模块化组件**：不同模态有专属 Encoder/Decoder；Alignment/Bridge/Graph 等“头”可按需启用，增加额外约束或损失。
3. **配置驱动**：所有行为由单一 YAML 控制，切换桥接/对齐策略无需改代码。
4. **高效 I/O 与训练**：兼容 AnnData/NPY/CSR/Zarr，多进程加载、Manifest 管理；训练支持 AMP、梯度裁剪、调度、早停、断点续训。

### 统一视角：用 Mosaic 视角覆盖所有 Problem Types

| Problem | 视作的 Mosaic 场景 | 配置要点 |
|---------|-------------------|----------|
| Paired  | 无缺失，全模态 | Masked ELBO=常规 ELBO，PoE 融合所有模态 |
| Unpaired| 模态互斥的多 cohort | 启用 MMD/Sinkhorn/GW 等对齐头对齐 $z_{bio}$ |
| Bridge  | 部分配对 + 大量单模态 | Masked ELBO + PoE 处理混合，BridgeHead 明确拉近 |
| Mosaic  | 任意缺失 | Masked ELBO 原生支持，可按需调权重 |
| Prediction| 训练配对、预测缺失 | 训练照常；预测时只编码可用模态再解码 |
| Hierarchical | 多模态 + 多来源 | 分离 $z_{bio}/z_{tech}$，多 dataset 对齐 + Masked ELBO |

总结：REMADOM 以一个灵活的 Mosaic 骨干为基，辅以可插拔的对齐/桥接组件，就能把不同整合需求都映射到同一训练管线，既减少了重复造轮子，也为未来更复杂的整合任务留下了扩展空间。

- **对齐头 (Alignment Heads)**：这是一系列可插拔的模块，作为额外的损失函数，用于在$z_{bio}$空间中对齐不同批次或不同模态的细胞分布。我们已经规划了多种“头”，包括：
- **基于分布的**：`MMDHead`、`SinkhornHead`（最优传输）、`GWHead`（Gromov-Wasserstein）。
- **基于桥接的**：`CrossGraphHead`，利用 MNN（相互最近邻）等桥接信号构建的图进行对齐。
- **基于结构的**：`GraphRegularizer`（图拉普拉斯正则化）和 `TemporalHead`（时序约束）。

3. **中心化的配置驱动 (Configuration-Driven)**：

    - 整个程序的行为由一个**单一的 YAML 配置文件**驱动。用户通过开关和调整配置项（如启用哪个对齐头、设置权重、选择学习率调度策略等）来解决不同的整合问题，而无需修改代码。

### 统一配置：把不同 Problem Types 映射成 Mosaic 特例

| Problem | 视作的 Mosaic 场景 | 配置思路 |
|---------|-------------------|----------|
| Paired  | 无缺失 | Masked ELBO=常规 ELBO，PoE 融合所有模态 |
| Unpaired| 模态互斥 | 启用 MMD/Sinkhorn/GW 等对齐头对齐 $z_{bio}$ |
| Bridge  | 少量配对 + 单模态 | Masked ELBO+PoE 处理混合，BridgeHead/bridge providers 显式拉近 |
| Mosaic  | 任意缺失 | Masked ELBO 原生支持，可设置模态权重 |
| Prediction| 训练配对、推理缺失 | 训练照常；推理时只编码可用模态再解码 |
| Hierarchical | 多模态 + 多来源 | 解耦 $z_{bio}/z_{tech}$，多 dataset 对齐，Masked ELBO 负责各自缺失 |

凭借这种“马赛克优先”的骨干，REMADOM 可以通过配置把看似不同的任务汇聚到同一训练管线，减少重复开发，也为更复杂的多模态整合留出扩展空间。
