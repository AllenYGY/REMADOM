from __future__ import annotations
import argparse
import json
import numpy as np
import torch

from ..eval.evaluator import Evaluator, EvalReport

def cli_eval(cfg_path: str, checkpoint: str, baseline: str | None, report_out: str) -> int:
    # Minimal: evaluate trustworthiness if user provides raw X and embedded Z (npz/npy outside scope)
    # Here we assume user prepared npy: X.npy and Z.npy in the same folder as report_out
    base_dir = report_out if report_out.endswith(".json") else (report_out + ".json")
    X_path = "X.npy"
    Z_path = "Z.npy"
    if not (os.path.exists(X_path) and os.path.exists(Z_path)):  # type: ignore[name-defined]
        print("Provide X.npy and Z.npy in cwd to run evaluator (minimal stub).")
        return 0
    X = torch.tensor(np.load(X_path), dtype=torch.float32)
    Z = torch.tensor(np.load(Z_path), dtype=torch.float32)
    ev = Evaluator(metric_names=["trustworthiness"])
    rep: EvalReport = ev.compute_metrics(Z, aux={"X": X})
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(rep.to_json())
    print("Saved report to", report_out)
    return 0

def main():
    ap = argparse.ArgumentParser("remadom-eval")
    ap.add_argument("--cfg", type=str, required=False)
    ap.add_argument("--checkpoint", type=str, required=False)
    ap.add_argument("--baseline", type=str, required=False)
    ap.add_argument("--report", type=str, required=True)
    args = ap.parse_args()
    return cli_eval(args.cfg or "", args.checkpoint or "", args.baseline, args.report)