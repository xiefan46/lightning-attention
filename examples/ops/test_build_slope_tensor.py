import torch

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

if __name__ == "__main__":
    for h in [1, 5, 12, 16, 64, 100]:
        s = _build_slope_tensor(h)
        print("h: {}, s shape: {}".format(h, s.shape))

