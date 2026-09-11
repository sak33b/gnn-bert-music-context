"""
utils.py — shared helpers used across every other file in src/.

Not called out as its own file in the assignment's folder diagram, but every
script needs config loading / seeding / device selection, so it lives here
instead of being copy-pasted six times.
"""
import random
import yaml
import numpy as np
import torch


def load_config(path: str = "config.yaml") -> dict:
    """Load the single project config.yaml into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Make a run reproducible across numpy / torch / python's random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Fall back to CPU silently if CUDA was requested but isn't available."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
