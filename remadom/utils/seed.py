from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
