import numpy as np
import pytest
try:
    import scipy.sparse as sp
except Exception:
    sp = None

from remadom.data.csr_dataset import CSRDenseDataset

@pytest.mark.skipif(sp is None, reason="scipy not available")
def test_csr_dataset(tmp_path):
    n, p = 7, 4
    X = sp.csr_matrix(np.arange(n*p).reshape(n, p))
    path = tmp_path / "atac_part0_7.npz"
    sp.save_npz(path, X)
    batches_p = tmp_path / "batches.npy"
    np.save(batches_p, np.zeros(n, dtype="int64"))
    ds = CSRDenseDataset({"atac": {"shards": [str(path)], "shape": [n, p]}}, batches_path=str(batches_p))
    item = ds[3]
    assert item["x_atac"].numel() == p