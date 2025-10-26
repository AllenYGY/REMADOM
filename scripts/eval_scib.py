import os
import numpy as np
import torch
from remadom.eval.evaluator import Evaluator
from remadom.eval.scib_wrapper import adata_from_numpy, compute_ilisi, compute_kbet

def main():
    # Expect X.npy, Z.npy, batches.npy in current directory
    X = np.load("X.npy")
    Z = np.load("Z.npy")
    batches = np.load("batches.npy")
    ev = Evaluator(["trustworthiness", "ilisi", "kbet", "batch_cls_acc"])
    # Compute trustworthiness and batch classifier directly from tensors
    rep = ev.compute_metrics(Z=torch.tensor(Z, dtype=torch.float32), batches=torch.tensor(batches, dtype=torch.long), aux={"X": torch.tensor(X, dtype=torch.float32), "X_np": X, "Z_np": Z, "batch_key": "batch"})
    print(rep.to_json())

    # Alternatively, construct AnnData and call scIB directly
    A = adata_from_numpy(X, Z, batches)
    if A is not None:
        print("iLISI:", compute_ilisi(A))
        print("kBET:", compute_kbet(A))

if __name__ == "__main__":
    main()