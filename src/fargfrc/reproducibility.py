import os
import random

import numpy as np
import torch


def set_global_reproducibility_seed(seed):
    """
    Set reproducible random states for Python, NumPy, and PyTorch.

    Notes
    -----
    - Must be called before model initialization and DataLoader creation.
    - Deterministic settings may reduce speed but improve reproducibility.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
