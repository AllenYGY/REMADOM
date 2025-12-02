from __future__ import annotations

import torch

from remadom.eval.alignment_metrics import compute_alignment_metrics


def test_compute_alignment_metrics_basic():
    z = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    groups = torch.tensor([0, 0, 1, 1])
    out = compute_alignment_metrics(z, groups)
    assert set(out.keys()) == {"within_dispersion", "between_centroid_dist", "centroid_variance", "silhouette_proxy"}
    assert out["within_dispersion"] >= 0.0
    assert out["between_centroid_dist"] > 0.0
    assert out["centroid_variance"] > 0.0
    assert -1.0 <= out["silhouette_proxy"] <= 1.0
