# REMADOM: Multi-omic alignment and generative modeling

remadom is a PyTorch library for learning shared embeddings across single-cell modalities (RNA, ATAC, ADT). It combines alignment losses (MMD, Sinkhorn OT, GW), graph regularization, and modality-aware decoders (RNA ZINB, ATAC Bernoulli, ADT mixture). It supports AMP training, checkpointing, and reproducibility.

Links

- Docs: <https://your-org.github.io/remadom/>
- Repo: <https://github.com/your-org/remadom>

## Installation

Base (CPU):

- pip install remadom

Editable dev install:

- git clone <https://github.com/your-org/remadom>
- cd remadom
- pip install -e .[dev]

Optional extras:

- Zarr/AnnData: pip install "remadom[zarr]"
- ANN (FAISS/HNSW): pip install "remadom[ann]"
- OT (GeomLoss/KeOps): pip install "remadom[ot]"
- Scanpy preprocessing: pip install "remadom[scanpy]"
- Plots (UMAP/Matplotlib): pip install "remadom[plots]"
- All: pip install "remadom[all]"

## Quickstart (single-file config + CLI)

1) Create config.yaml (example)

```yaml
data:
  path: path/to/data.h5ad
  batch_key: batch
  preprocess:
    rna: { hvg: 2000, flavor: seurat_v3 }
    atac:
      lsi: { enabled: true, n_components: 50, obsm_key: X_lsi }

model:
  latent_bio: 16
  encoders:
    rna: { in_dim: 2000 }
    atac: { in_dim: 50 }
  decoders:
    rna: { out_dim: 2000, dispersion: batch, library: true }
  fusion_temps: { rna: 1.0, atac: 1.5 }

alignment:
  mmd: { enabled: true, params: { weight: 0.1, group_key: batch } }
  sinkhorn: { enabled: false }

structure:
  laplacian_enabled: true
  laplacian_lambda: 0.001
  graph_backend: faiss

bridge:
  provider: mnn
  refresh: 5
  mnn: { k: 20, metric: euclidean, ann_backend: faiss }

optim:
  epochs: 20
  batch_size: 512
  lr: 1.0e-3
  scheduler: { name: cosine, T_max: 20, min_lr: 1.0e-6 }
  amp: { enabled: true, dtype: bf16 }
  early_stopping: { enabled: true, monitor: elbo.loss, mode: min, patience: 5, min_delta: 1.0e-4 }

logging:
  run_dir: runs/m8_demo
  log_interval: 100
