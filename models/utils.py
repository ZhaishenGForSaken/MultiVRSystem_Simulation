# models/utils.py
import os
import numpy as np

def clamp(x, lo, hi): return max(lo, min(hi, x))
def lowpass(prev, new, alpha): return alpha * new + (1 - alpha) * prev
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_heterogeneity(rng):
    return {
        "p_base_mul": float(np.clip(rng.normal(1.0, 0.1), 0.8, 1.3)),
        "p_dyn_mul": float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.3)),
        "l_base_mul": float(np.clip(rng.normal(1.0, 0.1), 0.8, 1.3)),
        "l_util_mul": float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.4)),
        "l_task_mul": float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.4)),
        "util_bias": float(np.clip(rng.normal(0.1, 0.05), 0.02, 0.25)),
    }
