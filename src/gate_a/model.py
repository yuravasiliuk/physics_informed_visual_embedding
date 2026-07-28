"""GATE A feasibility model — a *trainable* tiny clip encoder + per-object
dynamics heads (IMPLEMENTATION.md P0.5a).

Deliberately NOT V-JEPA: GATE A asks "is dynamics learnable from perfect RGB at
all", so the encoder must be trained from pixels. A few Conv3d blocks are enough
to overfit a handful of clips — capacity here is a feature, not a worry (the
overfit verdict is read on `v`/`w`, which are genuinely visible; see
train_overfit.py).

Flow per object:
    frames (B,3,T,H,W) --encoder--> feat (B,C,T',H',W')
    object mask (B,T,H,W) --adaptive_avg_pool3d--> weights (B,1,T',H',W')
    masked mean over (T',H',W')  --> pooled (B,C)
    per-target MLP head          --> {v,w,com_offset: (B,3)}

Mask pooling resamples the mask to whatever grid the encoder emits at runtime
(adaptive pool), so the conv stack can change without touching the data path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_DIMS = {"v": 3, "w": 3, "com_offset": 3}


class TinyClipEncoder(nn.Module):
    """Small 3D-CNN: (B,3,T,H,W) -> (B,C,T',H',W'). At T=16,H=W=128 the default
    stack yields (C,4,8,8) — enough spatial grid for per-object masked pooling."""

    def __init__(self, width=64, out_dim=128):
        super().__init__()
        def blk(ci, co, st):
            return nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=3, stride=st, padding=1, bias=False),
                nn.GroupNorm(8, co), nn.GELU(),
            )
        self.net = nn.Sequential(
            blk(3, width, (1, 2, 2)),          # 16,64,64
            blk(width, width, (2, 2, 2)),      # 8,32,32
            blk(width, out_dim, (2, 2, 2)),    # 4,16,16
            blk(out_dim, out_dim, (1, 2, 2)),  # 4,8,8
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class DynamicsModel(nn.Module):
    """Encoder + per-object masked pool + one MLP head per active target."""

    def __init__(self, targets, width=64, feat_dim=128, hidden=256):
        super().__init__()
        self.encoder = TinyClipEncoder(width=width, out_dim=feat_dim)
        self.targets = list(targets)
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(),
                             nn.Linear(hidden, TARGET_DIMS[t]))
            for t in self.targets
        })

    def pool(self, feat, masks):
        """feat (B,C,T',H',W'); masks (B,T,H,W) float -> pooled (B,C)."""
        w = F.adaptive_avg_pool3d(masks.unsqueeze(1), feat.shape[2:])  # (B,1,T',H',W')
        num = (feat * w).flatten(2).sum(-1)                            # (B,C)
        den = w.flatten(2).sum(-1).clamp_min(1e-6)                     # (B,1)
        return num / den

    def forward(self, frames, masks):
        pooled = self.pool(self.encoder(frames), masks)
        return {t: self.heads[t](pooled) for t in self.targets}
