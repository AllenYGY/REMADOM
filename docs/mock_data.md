# Mock Data Generation in REMADOM

## Why synthetic data?

REMADOM’s Phase 1 targets rapid iteration on the core Mosaic VAE without depending on large public datasets. A configurable synthetic data generator lets you:

- Begin training/debugging immediately after cloning the repo.
- Stress-test masking logic across all integration scenarios (paired, unpaired, bridge, mosaic, prediction, hierarchical).
- Produce perfectly reproducible datasets via fixed seeds.
- Unify unit/integration tests with the same artefacts used for manual smoke runs.

## Core logic

1. **Latent backbone**
   - Sample 3–5 Gaussian cluster centres in a low-dimensional latent space.
   - Draw one latent vector `z` per cell; record cluster IDs to emulate biological labels.
   - Optionally add batch/study offsets (used in bridge/hierarchical presets).
   - This latent `z` is a synthetic “true biology” representation shared by all modalities.

2. **Full-modality synthesis**
   - Convert `z` into RNA counts via log-linear Negative Binomial parameters.
   - Convert `z` into ATAC binary peaks through logistic Bernoulli probabilities.
   - Convert `z` into ADT intensities using exponentiated Gaussian (log-normal-like) values.
   - At this stage every cell has all three modalities—this is the “ground truth”.
   - These choices mirror the reconstruction terms used in the VAE (ZINB, Bernoulli, Gaussian).

3. **Scenario masking**
   - For each `problem_type`, compute `has_<mod>` masks describing which modalities are *observed*:
     - `paired`: keep RNA+ADT for all cells (CITE-like).
     - `unpaired`: split RNA-only vs ATAC-only cohorts, no overlap.
     - `bridge`: large single-modality cohorts with ~5 % paired “bridge” cells.
     - `mosaic`: each cell randomly retains any subset of {RNA, ATAC, ADT}.
     - `prediction`: training cells are paired; evaluation cells hide target modality.
     - `hierarchical`: three studies with different modality mixes and batch shifts.
   - Masks are stored in `adata.obs["has_<mod>"]`; missing entries in the data matrices are zeroed out.
   - Downstream loaders read these masks to skip missing modalities during training.

4. **AnnData packaging**
   - `adata.X` / `layers["counts"]` hold RNA counts; ATAC/ADT arrays go to `adata.obsm`.
   - `adata.obs` includes columns for `batch`, `dataset`, `cluster`, `split`, and modality masks.
   - `adata.uns["adt_names"]` captures ADT vocabulary when present.
   - A `keys` dictionary is returned alongside the AnnData object for configuration wiring.

## Usage modes

### Python API

```python
from remadom.data.mock import generate_mock_dataset

adata, keys = generate_mock_dataset(
    "bridge",
    n_cells=600,
    n_genes=2000,
    n_peaks=60000,
    n_proteins=30,
    seed=42,
)
adata.write_h5ad("examples/mock/mock_bridge_rna_atac.h5ad")
print(keys)
```

### CLI helper

```bash
conda activate REMADOM
python scripts/make_mock_multimodal.py \
  --problem mosaic \
  --out examples/mock/mock_mosaic_multiome.h5ad \
  --config-out configs/examples/mock_mosaic.yaml \
  --cells 600 --genes 2000 --peaks 60000 --proteins 30 --seed 0
```

### Batch generator

```bash
chmod +x scripts/generate_all_mock_data.sh
./scripts/generate_all_mock_data.sh
```
Outputs land in `examples/mock/*.h5ad` and `configs/examples/mock_*.yaml`.

## Validation hooks

- Unit test: `tests/unit/test_mock_generator.py` checks masks, shapes, and scenario coverage.
- Integration test: `tests/integration/test_phase1_training.py` trains on bridge/mosaic datasets.

## Customisation tips

- Adjust cell/feature counts via CLI flags.
- Modify scenario ratios（桥接占比、随机缺失比例等）通过编辑 `remadom/data/mock.py`。
- 若想保留“完整配对真值”，可先生成 `problem_type="paired"` 的数据，再在内存中手动打掩码。
