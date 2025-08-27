
import os, random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
