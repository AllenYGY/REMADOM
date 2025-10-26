import torch
from remadom.typing import Batch

def make_synthetic_mosaic(n=600, d=16, p_rna=200, p_atac=180, p_adt=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    # Linear maps
    A_rna = torch.randn(d, p_rna, generator=g) * 0.3
    A_atac = torch.randn(d, p_atac, generator=g) * 0.3
    A_adt = torch.randn(d, p_adt, generator=g) * 0.3
    # Create mosaic: thirds have different modality availability
    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2
    z1, z2, z3 = z[:n1], z[n1:n1+n2], z[n1+n2:]
    X_rna = torch.cat([torch.relu(z1 @ A_rna) + 0.1, torch.zeros(n2, p_rna), torch.relu(z3 @ A_rna) + 0.1], 0)
    X_atac = torch.cat([torch.zeros(n1, p_atac), torch.sigmoid(z2 @ A_atac), torch.sigmoid(z3 @ A_atac)], 0)
    X_adt = torch.cat([torch.sigmoid(z1 @ A_adt), torch.zeros(n2, p_adt), torch.sigmoid(z3 @ A_adt)], 0)
    has_rna = torch.cat([torch.ones(n1, dtype=torch.bool), torch.zeros(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    has_atac = torch.cat([torch.zeros(n1, dtype=torch.bool), torch.ones(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    has_adt = torch.cat([torch.ones(n1, dtype=torch.bool), torch.zeros(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    batch_labels = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long), 2*torch.ones(n3, dtype=torch.long)], 0)
    lib = X_rna.sum(1)
    return Batch(
        x_rna=X_rna, x_atac=X_atac, x_adt=X_adt,
        has_rna=has_rna, has_atac=has_atac, has_adt=has_adt,
        libsize_rna=lib, batch_labels=batch_labels
    )

if __name__ == "__main__":
    b = make_synthetic_mosaic()
    print("Shapes:", b.x_rna.shape, b.x_atac.shape, b.x_adt.shape)