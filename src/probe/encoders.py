"""Frozen V-JEPA 2 wrapper with per-block taps (gate-b-probe.md §4.1).

v0.1 replaces the HF `facebook/vjepa2-vitl-fpc64-256` checkpoint with the
repo-native ViT-g/384 path validated in the user's notebook
(vjepa2/notebooks/Untitled.ipynb): a facebookresearch/vjepa2 checkout +
`weights/vitg-384.pt`, the repo's own eval video transform (resize short side
to 256/224·img_size, center-crop img_size², ImageNet-normalize), and the
model's native `out_layers` hook — forward then returns EVERY block's tokens
with the final LayerNorm applied, which is exactly the per-layer tap the
GATE-B curve needs (no hidden-state surgery, no HF dependency).

Per-object pooling happens here (at the token level) because the token-grid
geometry (tubelet 2 × patch 16) is an encoder detail: each body's pixel mask
is pushed through the SAME resize-short-side + center-crop arithmetic as the
pixels (mirrored from the repo's resize_clip/get_resize_sizes/CenterCrop),
soft-resized to the (T', H', W') token grid, and used as weights for a
weighted token mean per block. Soft weights instead of a hard threshold so a
small body covering fractions of patches still pools sensibly.
"""

import os
import sys

import numpy as np

import cv2
import torch

DEFAULT_REPO = "~/vis_emb/vjepa2"
DEFAULT_WEIGHTS_REL = "weights/vitg-384.pt"   # under the repo checkout
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VJEPA2Encoder:
    """Frozen repo-native V-JEPA 2 ViT-g; encode_pooled() → (n_layers, D) per body."""

    def __init__(self, repo=DEFAULT_REPO, weights=None, img_size=384, device=None):
        repo = os.path.expanduser(repo)
        weights = (os.path.join(repo, DEFAULT_WEIGHTS_REL) if weights is None
                   else os.path.expanduser(weights))
        # The vjepa2 checkout's package root is also named `src` — insert the
        # repo at the FRONT of sys.path so its `src` wins over this project's
        # src/ if both are importable.
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import src.datasets.utils.video.transforms as video_transforms
        import src.datasets.utils.video.volume_transforms as volume_transforms
        from src.models.vision_transformer import vit_giant_xformers_rope

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on GPU: probe features feed a ridge solve, fp16 noise is far
        # below the fit's regularization scale; ViT-g in fp32 would double the
        # ~2 GB weight footprint for nothing.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.patch, self.tubelet = 16, 2          # ViT-g/16, tubelet 2 (paper)
        self.crop = int(img_size)

        # num_frames only seeds the RoPE grid default; with use_rope the
        # forward re-derives T from the input, so shorter/longer clips are fine
        # (the paper evaluates with varied frame counts for the same reason).
        self.model = vit_giant_xformers_rope(
            img_size=(self.crop, self.crop), num_frames=32,
        )
        # Tap every block: `out_layers` is a plain attribute the repo forward
        # checks — when set, it returns [norm(block_i(x)) for i in out_layers].
        self.model.out_layers = list(range(len(self.model.blocks)))
        self.n_layers = len(self.model.blocks)
        # Notebook's load recipe: take the "encoder" entry, strip DDP/backbone
        # prefixes, strict=False (predictor/target keys are absent by design).
        state = torch.load(weights, weights_only=True, map_location="cpu")["encoder"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        msg = self.model.load_state_dict(state, strict=False)
        print(f"loaded {weights}: {msg}")
        self.model.to(self.device, self.dtype).eval()
        for p in self.model.parameters():          # frozen, heads only (§2)
            p.requires_grad_(False)

        # Repo eval transform, exactly as the notebook's build_pt_video_transform.
        self.short_side = int(256.0 / 224 * self.crop)
        self.transform = video_transforms.Compose([
            video_transforms.Resize(self.short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(self.crop, self.crop)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                       std=IMAGENET_DEFAULT_STD),
        ])
        # Cache provenance string (replaces the old HF model name in the .npz).
        self.model_id = f"vjepa2-pt-vitg-{self.crop}:{os.path.basename(weights)}"

    def _mask_to_token_weights(self, m, t_grid, s_grid):
        """(T, h, w) float mask → (t_grid·s_grid·s_grid,) token weights.

        Mirrors the pixel path's geometry exactly: short side → self.short_side
        (get_resize_sizes uses int() truncation), then a round()-centered crop
        of crop² (transforms.CenterCrop), then INTER_AREA down to the token
        grid (fractional patch coverage) and a mean over each tubelet's frames.
        """
        T, h, w = m.shape
        if w < h:
            nw, nh = self.short_side, int(self.short_side * h / w)
        else:
            nh, nw = self.short_side, int(self.short_side * w / h)
        x1 = int(round((nw - self.crop) / 2.0))
        y1 = int(round((nh - self.crop) / 2.0))
        grid = np.empty((T, s_grid, s_grid), dtype=np.float32)
        for t in range(T):
            r = cv2.resize(m[t], (nw, nh), interpolation=cv2.INTER_AREA)
            r = r[y1:y1 + self.crop, x1:x1 + self.crop]
            grid[t] = cv2.resize(r, (s_grid, s_grid), interpolation=cv2.INTER_AREA)
        # (T, s, s) → (T', s, s): average the tubelet's frames into its token.
        return grid.reshape(t_grid, self.tubelet, s_grid, s_grid).mean(axis=1).reshape(-1)

    @torch.no_grad()
    def encode_pooled(self, frames, masks_per_body):
        """frames (T, h, w, 3) uint8 RGB; masks_per_body list of (T, h, w) bool.

        Returns (features, weights):
            features (n_bodies, n_layers, D) float32 — weighted-mean token
                features per block (layer axis = block index, every block
                tapped through the final LayerNorm; the per-layer report
                indexes this axis).
            weights  (n_bodies,) float32 — total token weight (≈ visible token
                area) per body; 0 ⇒ vanished under downsampling, caller skips.
        """
        T = frames.shape[0] - (frames.shape[0] % self.tubelet)  # tubelet-align
        frames = frames[:T]
        # Notebook path verbatim: (T,H,W,C) uint8 → (T,C,H,W) → transform →
        # (C,T,H,W) → batch dim. The transform owns resize/crop/normalize.
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        x = self.transform(video).unsqueeze(0).to(self.device, self.dtype)
        outs = self.model(x)                         # n_layers × (1, N, D)
        hs = torch.stack([o[0] for o in outs])       # (L, N, D), model dtype

        t_grid, s_grid = T // self.tubelet, self.crop // self.patch
        n_tokens = t_grid * s_grid * s_grid
        # Hard guard on token-order assumptions: the 3D patch embed flattens
        # (T', H', W') row-major. If N mismatches, the mask→token mapping
        # below would silently pool garbage — fail loudly instead.
        assert hs.shape[1] == n_tokens, (
            f"token count {hs.shape[1]} != grid {t_grid}x{s_grid}x{s_grid}; "
            "patch/tubelet assumptions broken for this checkpoint"
        )

        feats, weights = [], []
        for masks in masks_per_body:
            wv = self._mask_to_token_weights(masks[:T].astype(np.float32),
                                             t_grid, s_grid)
            total = float(wv.sum())
            if total <= 0.0:
                feats.append(torch.zeros(self.n_layers, hs.shape[2], device=hs.device))
                weights.append(0.0)
                continue
            w = torch.from_numpy(wv).to(hs.device)
            # Per-layer fp32 pooling: a fp16 sum over ~9k tokens loses
            # precision and risks overflow; one (N, D) fp32 slice at a time
            # keeps the transient small next to the fp16 (L, N, D) stack.
            pooled = torch.stack([
                (hs[l].float() * w[:, None]).sum(dim=0) / total
                for l in range(self.n_layers)
            ])
            feats.append(pooled)
            weights.append(total)
        return (
            torch.stack(feats).cpu().numpy().astype(np.float32),
            np.asarray(weights, dtype=np.float32),
        )
