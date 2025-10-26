import torch
from remadom.bridges.dictionary import BridgeDictionary

def main():
    torch.manual_seed(0)
    n, p_a, p_b = 200, 64, 80
    W_true = torch.randn(p_a, p_b) * 0.2
    X_a = torch.randn(n, p_a)
    X_b = X_a @ W_true + 0.05 * torch.randn(n, p_b)
    # split
    idx = torch.randperm(n)
    bridge = idx[:100]
    test = idx[100:]
    bd = BridgeDictionary(rank=32, lam=1e-3)
    bd.fit(X_a[bridge], X_b[bridge])
    X_b_pred = bd.map_a_to_b(X_a[test])
    mse = ((X_b_pred - X_b[test]) ** 2).mean().item()
    print("Bridge dictionary MSE (lower is better):", mse)

if __name__ == "__main__":
    main()