import torch
from remadom.train.trainer import Trainer

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self._beta_current = 1.0
    def set_beta(self, b: float): self._beta_current = b
    def embed(self, batch): return {"z": torch.zeros((batch.x.shape[0], 2))}
    def forward(self, x): return x
    def elbo(self, batch, beta=None, mod_weights=None):
        # val loader will pass batches with increasing loss; train just returns const
        loss = torch.tensor(float(batch.loss_val) if hasattr(batch, "loss_val") else 1.0)
        return {"recon": loss, "kl": torch.tensor(0.0), "total": loss}

class B:
    def __init__(self, n, loss_val=1.0):
        self.x = torch.zeros(n,1)
        self.loss_val = loss_val

def test_early_stopping_logic():
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, opt, None, heads=[], cfg=type("C", (), {"optim": type("O", (), {"epochs": 50, "clip": type("CL", (), {"enabled": False})(), "early_stopping": type("ES", (), {"enabled": True, "patience": 2, "min_delta": 0.1})()})()}))
    # Train loader of 3 steps
    train_loader = [B(4) for _ in range(3)]
    # Validation: loss improves twice, then plateaus
    val_losses = [1.0, 0.8, 0.7, 0.61, 0.60, 0.60, 0.60]
    class VL:
        def __iter__(self):
            for v in val_losses:
                b = B(4, loss_val=v)
                yield b
    # Monkeypatch validate to consume a single value per epoch
    def validate_once(_):
        # pop from val_losses, return value
        v = val_losses.pop(0) if val_losses else 0.60
        return v
    trainer.validate = validate_once  # type: ignore
    logs = trainer.fit(train_loader, VL())
    # Should stop early when patience exceeded (after improvements stop)
    assert trainer._best_metric <= 0.60 + 1e-6