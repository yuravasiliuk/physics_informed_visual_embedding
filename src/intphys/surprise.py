"""Option-A intuitive-physics surprise from a frozen V-JEPA 2 world model.

This is the *action-free* recipe (NOT the energy-landscape notebook, which uses
the action-conditioned vjepa2-AC predictor for robot planning). It reproduces
V-JEPA 2's own pretraining objective at inference time as a violation-of-
expectation signal, exactly as app/vjepa/train.py computes its loss:

    h = target_encoder(clip)                      # layer-normed target reps
    z = predictor(encoder(clip, masks_enc),       # predict masked tokens from
                  masks_enc, masks_pred)          #   the context tokens
    surprise = mean |z - h[masks_pred]|           # L1 in representation space

For intuitive physics the masks are *temporal*: context = the past token-frames
of a short window, pred = the next token-frame(s). A physically impossible event
is poorly predicted from its past, so the L1 spikes there. We slide the window
across the video and aggregate (max localizes the violation moment; mean is the
smoother global score).

Loaded from the LOCAL full pretraining checkpoint (weights/vitg-384.pt holds
encoder + predictor + target_encoder), so nothing is downloaded. Construction
mirrors src.hub.backbones._make_vjepa2_model(model_name="vit_giant_384") with
pretrained=False, then loads our three sub-state-dicts.

GPU required in practice (ViT-g); this module is import-safe on CPU but a real
run wants cuda + fp16. The token-index conventions below match apply_masks
([B, K] index lists) and must be validated against a real forward on GPU before
trusting absolute numbers — see plans/probe/intphys2-surprise.md.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_REPO = "~/vis_emb/vjepa2"
DEFAULT_CKPT_REL = "weights/vitg-384.pt"


def _clean_backbone_key(sd):
    """module./backbone. prefix strip — copy of hub backbones._clean_backbone_key
    (re-implemented to avoid importing the hub's download-on-pretrained path)."""
    out = {}
    for k, v in sd.items():
        out[k.replace("module.", "").replace("backbone.", "")] = v
    return out


class VJEPA2Surprise:
    """Frozen V-JEPA 2 world model → per-video intuitive-physics surprise."""

    def __init__(self, repo=DEFAULT_REPO, ckpt=None, img_size=384, num_frames=64,
                 device=None):
        repo = os.path.expanduser(repo)
        ckpt = (os.path.join(repo, DEFAULT_CKPT_REL) if ckpt is None
                else os.path.expanduser(ckpt))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # Repo-native constructors (the same ones the hub factory calls).
        from src.models import predictor as vit_predictor
        from src.models import vision_transformer as vit_encoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.patch, self.tubelet = 16, 2
        self.crop = int(img_size)
        self.s_grid = self.crop // self.patch          # 24 at 384
        self.tokens_per_tframe = self.s_grid * self.s_grid

        enc_kwargs = dict(
            patch_size=self.patch, img_size=(self.crop, self.crop),
            num_frames=num_frames, tubelet_size=self.tubelet,
            use_sdpa=True, use_SiLU=False, wide_SiLU=True,
            uniform_power=False, use_rope=True,
        )
        self.encoder = vit_encoder.__dict__["vit_giant_xformers"](**enc_kwargs)
        self.target_encoder = vit_encoder.__dict__["vit_giant_xformers"](**enc_kwargs)
        self.predictor = vit_predictor.__dict__["vit_predictor"](
            img_size=(self.crop, self.crop), patch_size=self.patch,
            use_mask_tokens=True, embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=384, out_embed_dim=None,
            num_frames=num_frames, tubelet_size=self.tubelet,
            depth=12, num_heads=12, num_mask_tokens=10,
            use_rope=True, uniform_power=False, use_sdpa=True,
            use_silu=False, wide_silu=True,
        )

        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(_clean_backbone_key(sd["encoder"]), strict=False)
        self.target_encoder.load_state_dict(
            _clean_backbone_key(sd["target_encoder"]), strict=False)
        self.predictor.load_state_dict(_clean_backbone_key(sd["predictor"]), strict=False)

        for m in (self.encoder, self.target_encoder, self.predictor):
            m.to(self.device, self.dtype).eval()
            for p in m.parameters():
                p.requires_grad_(False)

        # ImageNet normalization (same as the encoder notebook).
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1, 1)
        self.model_id = f"vjepa2-pt-vitg-{self.crop}-surprise"

    # -- preprocessing --------------------------------------------------------
    def _to_clip(self, frames):
        """(T, h, w, 3) uint8 RGB → (1, 3, T, crop, crop) normalized on device.

        Center-square resize to crop² (cv2). T is tubelet-aligned by the caller
        via window construction; here we only resize+normalize."""
        import cv2
        sized = np.stack([
            cv2.resize(f, (self.crop, self.crop), interpolation=cv2.INTER_AREA)
            for f in frames
        ]).astype(np.float32) / 255.0                     # (T, crop, crop, 3)
        x = torch.from_numpy(sized).to(self.device).permute(3, 0, 1, 2).unsqueeze(0)
        return ((x - self._mean) / self._std).to(self.dtype)  # (1,3,T,crop,crop)

    # -- masks ----------------------------------------------------------------
    def _temporal_masks(self, t_ctx, t_pred, t_total):
        """Context = first t_ctx token-frames, pred = next t_pred token-frames.

        Returns ([B,Kx] enc indices], [[B,Ky] pred indices]) as the [B,K] index
        lists apply_masks expects (B=1). Tokens are flattened (T', H', W')
        row-major, so token-frame f owns indices [f·S, (f+1)·S) with S=576."""
        S = self.tokens_per_tframe
        ctx = torch.arange(0, t_ctx * S, device=self.device)
        pred = torch.arange(t_ctx * S, (t_ctx + t_pred) * S, device=self.device)
        assert (t_ctx + t_pred) <= t_total, "window exceeds clip token-frames"
        return [ctx.unsqueeze(0)], [pred.unsqueeze(0)]

    # -- core -----------------------------------------------------------------
    @torch.no_grad()
    def _window_surprise(self, clip, t_ctx, t_pred):
        """L1 prediction error over the predicted token-frames of one window.
        clip: (1,3,T,crop,crop), T tubelet-aligned. Returns float."""
        from src.masks.utils import apply_masks
        t_total = clip.shape[2] // self.tubelet
        masks_enc, masks_pred = self._temporal_masks(t_ctx, t_pred, t_total)

        h = self.target_encoder(clip)                       # (1, N, D)
        h = F.layer_norm(h, (h.size(-1),))
        h_pred = apply_masks(h, masks_pred, concat=False)[0]  # (1, Ky, D)

        z = self.encoder(clip, masks_enc)                   # context tokens
        z = self.predictor(z, masks_enc, masks_pred)        # (1, Ky, D) predicted
        return float(torch.mean(torch.abs(z - h_pred)).item())

    @torch.no_grad()
    def video_surprise(self, frames, window=16, t_ctx=4, t_pred=4, stride=8):
        """Slide a (context→future) window over the video; aggregate L1 error.

        frames : (T, h, w, 3) uint8 RGB, the whole clip (caller subsamples fps).
        window : raw frames per window (tubelet-aligned internally).
        t_ctx  : context token-frames (= 2·t_ctx raw frames of conditioning).
        t_pred : predicted token-frames per window.
        stride : raw-frame hop between windows.

        Returns {max, mean, per_window:[...]}. `max` localizes the violation
        (the headline VoE score); `mean` is the smoother global number.
        """
        window -= window % self.tubelet
        T = len(frames)
        if T < window:                       # pad short clips by edge-repeat
            frames = np.concatenate([frames, np.repeat(frames[-1:], window - T, 0)])
            T = window
        scores = []
        for start in range(0, T - window + 1, stride):
            clip = self._to_clip(frames[start:start + window])
            scores.append(self._window_surprise(clip, t_ctx, t_pred))
        if not scores:                       # window == T exactly, no slide
            scores = [self._window_surprise(self._to_clip(frames[:window]), t_ctx, t_pred)]
        return {"max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "per_window": scores}
