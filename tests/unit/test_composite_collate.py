import torch
from remadom.data.composite_dataset import composite_collate

def test_composite_collate_basic():
    items = [
        {"x_rna": torch.ones(3), "has_rna": torch.tensor(True), "batch_labels": torch.tensor(0)},
        {"x_rna": torch.zeros(3), "has_rna": torch.tensor(True), "batch_labels": torch.tensor(1)},
    ]
    b = composite_collate(items)
    assert b["x_rna"].shape == (2,3)
    assert b["batch_labels"].shape == (2,)