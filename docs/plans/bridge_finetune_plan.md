# Bridge-only Fine-tune Plan

## Motivation

Comparing bridge providers (MNN, Seeded, Dictionary, LinearMap) is noisy when each run learns the entire MosaicVAE from scratch. We want a clean protocol that:

1. Pretrains the core MosaicVAE on the bridge dataset without any bridge head, ensuring reconstruction/masked ELBO is converged.
2. Fine-tunes the same backbone with different bridge providers, freezing or lightly updating the main networks, so differences come purely from the bridge loss.
3. Automates evaluation (MAE, plots, bridge diagnostics) and outputs a concise summary for each method.

## Baseline Pretrain

- Config: `configs/examples/mock_bridge.yaml` with `bridge.enabled: false`.
- Train long enough (e.g., 200 epochs) to stabilise ELBO.
- Save checkpoint to `runs/mock/mock_bridge_pretrain/checkpoint.last.pt`.

## Fine-tune Config

Create `configs/examples/mock_bridge_ft.yaml`:

```yaml
data: same as baseline
model: identical (latent dims, encoders/decoders)
bridge:
  enabled: true
  method: mnn  # overridden per run
  weight: 0.5
  normalize: true
optim:
  epochs: 50
  lr: 1e-4
  batch_size: 256
  amp: {...}
  freeze_backbone: true   # new flag to freeze encoders/decoders/fusion
logging:
  run_dir: runs/bridge_ft
evaluation:
  enabled: true
  save_predictions: true
  tasks: [bridge_imputation]
```

Notes:
- `freeze_backbone` requires Trainer support (set `requires_grad=False` for encoders/decoders; only bridge head parameters update).
- Optionally expose `optim.lr_bridge` to use a larger LR for bridge head while backbone is frozen.

## CLI Workflow

1. **Pretrain**
   ```bash
   python -m remadom.cli.train \
     --cfg configs/examples/mock_bridge.yaml \
     logging.run_dir=runs/mock/mock_bridge_pretrain \
     optim.epochs=200
   ```

2. **Fine-tune per provider**
   ```bash
   for method in mnn seeded dictionary linmap; do
     python -m remadom.cli.train \
       --cfg configs/examples/mock_bridge_ft.yaml \
       --resume runs/mock/mock_bridge_pretrain/checkpoint.last.pt \
       bridge.method=${method} \
       logging.run_dir=runs/bridge_ft/${method}
   done
   ```

3. Each run writes:
   - `metrics.final.json` (includes `evaluation.bridge_imputation` MAE).
   - `bridge_metrics.json`, `bridge_edges.png`.
   - `evaluation_samples.npz`, `evaluation_plots.png`.

## Automation Script

- Extend `scripts/compare_bridge_strategies.sh` or add `scripts/compare_bridge_finetune.sh` to:
  - Accept `--pretrain-run` path or auto-trigger pretrain when missing.
  - Loop over methods with `--resume`.
  - Append summary table with train loss, KL, bridge loss, and evaluation MAE.
  - Optional flag `--no-freeze` to allow backbone updates (slow but flexible).

## Acceptance Criteria

- `python -m remadom.cli.train --cfg configs/examples/mock_bridge_ft.yaml ...` runs successfully with `--resume`.
- Each method produces distinct `evaluation.bridge_imputation` metrics and plots in `runs/bridge_ft/<method>/`.
- Summary script emits `runs/bridge_ft/summary.txt` comparing methods.
- Documentation (`docs/bridge_mnn.md` or this plan) explains the two-stage protocol and how to reproduce the comparison.
