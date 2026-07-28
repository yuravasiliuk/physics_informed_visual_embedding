"""GATE A overfit harness — can a trainable encoder learn per-object dynamics
from perfect RGB *at all*? (IMPLEMENTATION.md P0.5a.)

This is the minimal feasibility test: take a handful of clips, train and
evaluate on the SAME set, and drive train R² up. Reading:

    v  (control) MUST reach R² ~1. v is rigid translation — directly visible.
       If it doesn't overfit, the harness/data/labels are broken; fix that
       before believing anything else (same role `v` plays in GATE B).
    w  should also overfit (visible rotation).
    com_offset overfitting proves little — a high-capacity net can memorize it
       by clip identity though it has no appearance cue (R8). Don't over-read it.

Not a generalization test (train==eval here by design); that's the later split
gate. CPU-runnable at tiny --res/--clips/--steps for a smoke; GPU for real.

    python src/gate_a/train_overfit.py --data ~/data/sim_physics \\
        --clips 8 --targets v --steps 2000 --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_a.model import DynamicsModel, TARGET_DIMS
from probe.dataset import find_clip_dirs, load_clip_samples

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_samples(clip_dirs, n_frames, res, max_objects):
    """Load clips → flat per-object tensors. Returns (frames, masks, targets)
    where frames (N,3,T,res,res) f32, masks (N,T,res,res) f32, targets dict of
    (N,3) f64. Each object carries its clip's frames (re-encoded per object —
    fine at the overfit scale)."""
    import cv2
    F_, M_, T_ = [], [], {k: [] for k in TARGET_DIMS}
    meta = []
    for cd in clip_dirs:
        frames, samples = load_clip_samples(cd, n_frames=n_frames)
        if not samples:
            continue
        T = frames.shape[0]
        # Resize the shared clip once; reuse for each object.
        small = np.stack([cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA)
                          for f in frames]).astype(np.float32) / 255.0
        small = (small - IMAGENET_MEAN) / IMAGENET_STD            # (T,res,res,3)
        clip_t = torch.from_numpy(small).permute(3, 0, 1, 2)      # (3,T,res,res)
        for s in samples[:max_objects]:
            m = np.stack([cv2.resize(mf.astype(np.float32), (res, res),
                                     interpolation=cv2.INTER_AREA)
                          for mf in s["masks"]])                   # (T,res,res)
            F_.append(clip_t)
            M_.append(torch.from_numpy(m))
            for k in TARGET_DIMS:
                T_[k].append(np.asarray(s[k], np.float64))
            meta.append((os.path.basename(cd), s["name"], s["scenario"]))
    if not F_:
        sys.exit("no usable objects in the selected clips")
    frames = torch.stack(F_)
    masks = torch.stack(M_)
    targets = {k: torch.tensor(np.stack(v)) for k, v in T_.items()}
    return frames, masks, targets, meta


def r2_per_target(pred, true, true_mean):
    """Aggregate R² over dims (1 - SSE/SST), SST vs the set mean."""
    sse = ((true - pred) ** 2).sum().item()
    sst = ((true - true_mean) ** 2).sum().item()
    return 1.0 - sse / sst if sst > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/sim_physics")
    p.add_argument("--clips", type=int, default=8, help="how many clips to overfit")
    p.add_argument("--targets", nargs="+", default=["v"],
                   choices=list(TARGET_DIMS) + ["all"])
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--max_objects", type=int, default=4, help="cap objects/clip")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    targets = list(TARGET_DIMS) if "all" in args.targets else args.targets
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clip_dirs = find_clip_dirs(os.path.expanduser(args.data))[:args.clips]
    if not clip_dirs:
        sys.exit(f"no clip dirs under {args.data}")
    frames, masks, tgt_raw, meta = build_samples(
        clip_dirs, args.frames, args.res, args.max_objects)
    N = frames.shape[0]
    print(f"overfit set: {N} objects from {len(clip_dirs)} clips | targets={targets} "
          f"| res={args.res} | device={device}")

    frames, masks = frames.to(device), masks.to(device)
    # Standardize each target (per-dim, over the set) so multi-target loss is
    # balanced; R² is scale-free so we report it in standardized space (identical
    # to original-space R² up to per-dim weighting).
    tgt, tmean, tstd = {}, {}, {}
    for k in targets:
        y = tgt_raw[k].to(device).float()
        mu, sd = y.mean(0, keepdim=True), y.std(0, keepdim=True).clamp_min(1e-6)
        tgt[k] = (y - mu) / sd
        tmean[k] = tgt[k].mean(0, keepdim=True)

    model = DynamicsModel(targets, width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    for step in range(1, args.steps + 1):
        model.train()
        preds = model(frames, masks)
        loss = sum(F.mse_loss(preds[k], tgt[k]) for k in targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                preds = model(frames, masks)
                r2 = {k: r2_per_target(preds[k], tgt[k], tmean[k]) for k in targets}
            r2s = " ".join(f"{k} R²={r2[k]:.3f}" for k in targets)
            print(f"  step {step:5d} | loss {loss.item():.4f} | {r2s}")

    print("\nverdict: v R²~1 => dynamics learnable from perfect RGB (GATE A path "
          "open). v stuck low => harness/data/labels broken, fix before scaling.")


if __name__ == "__main__":
    main()
