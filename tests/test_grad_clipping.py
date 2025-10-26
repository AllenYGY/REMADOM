import torch

def test_grad_clipping_applies_norm():
    lin = torch.nn.Linear(10, 10)
    x = torch.randn(64, 10)
    y = torch.randn(64, 10)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    crit = torch.nn.MSELoss()
    # Make large gradients
    out = lin(x) * 1000.0
    loss = crit(out, y)
    loss.backward()
    # Measure grad norm before clipping
    total_norm = torch.sqrt(sum((p.grad.data**2).sum() for p in lin.parameters()))
    torch.nn.utils.clip_grad_norm_(lin.parameters(), max_norm=1.0, norm_type=2.0)
    clipped_norm = torch.sqrt(sum((p.grad.data**2).sum() for p in lin.parameters()))
    assert clipped_norm <= total_norm
    assert clipped_norm <= 1.0 + 1e-4