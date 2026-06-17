"""Run the Option-A V-JEPA 2 surprise pipeline over IntPhys2 and score it.

Two outputs:
  1. A predictions CSV (filename, response, surprise_max, surprise_mean) that
     drops straight into the official ~/IntPhys2/ComputeScoreIntPhys2.ipynb
     scorer (it reads `filename` + `response`, target from `type`).
  2. The **relative pairwise accuracy** printed here directly — the metric
     V-JEPA 2 reports for world models: within each matched (possible,
     impossible) pair, correct iff surprise(impossible) > surprise(possible).
     Threshold-free, no label leakage. This is the honest headline number.

`response` for the CSV is a *count-based within-SceneIndex* assignment (rank by
surprise, label the k highest as Impossible where k = #impossible in that scene)
— the bridge to the VLM-oriented absolute scorer. Prefer the printed relative
number; the CSV is for parity with the official notebook.

GPU + the local full checkpoint (weights/vitg-384.pt) required for a real run:

    python src/intphys/run_intphys2.py --intphys_root ~/data/IntPhys2 \\
        --split Debug --out_csv out/intphys/debug_vjepa2.csv --fps 15
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_video(path, fps, max_frames=256):
    """Decode an mp4, subsample toward `fps`, return (T, h, w, 3) uint8 RGB.

    cv2 (present in every env) rather than decord — one less dependency. cv2
    decodes BGR; we flip to RGB to match the encoder's ImageNet normalization."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / fps)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame[..., ::-1])           # BGR -> RGB
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.ascontiguousarray(np.stack(frames))


def relative_pairwise_accuracy(rows):
    """Canonical IntPhys relative metric. `type` is like '1_Possible' /
    '2_Impossible'; (SceneIndex, leading pair-id) identifies a matched pair.
    Correct iff the impossible twin is more surprising than the possible one.
    Returns (acc, n_pairs) over score keys 'max' and 'mean'."""
    from collections import defaultdict
    out = {}
    for key in ("max", "mean"):
        pairs = defaultdict(dict)  # (scene, pair_id) -> {Possible: s, Impossible: s}
        for r in rows:
            t = r["type"]
            pair_id = t.split("_")[0]
            polarity = "Impossible" if "Impossible" in t else "Possible"
            pairs[(r["SceneIndex"], pair_id)][polarity] = r[f"surprise_{key}"]
        correct = total = 0
        for d in pairs.values():
            if "Possible" in d and "Impossible" in d:
                total += 1
                correct += int(d["Impossible"] > d["Possible"])
        out[key] = (correct / total if total else float("nan"), total)
    return out


def assign_response_within_scene(rows):
    """Count-based within-SceneIndex labels for the official scorer CSV: rank by
    surprise_max, mark the k highest as Impossible (response=0), rest Possible
    (response=1), where k = #impossible in that scene (from `type`)."""
    from collections import defaultdict
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["SceneIndex"]].append(r)
    for scene_rows in by_scene.values():
        k = sum("Impossible" in r["type"] for r in scene_rows)
        order = sorted(scene_rows, key=lambda r: r["surprise_max"], reverse=True)
        for i, r in enumerate(order):
            r["response"] = 0 if i < k else 1   # 0=Impossible/no, 1=Possible/yes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intphys_root", required=True)
    p.add_argument("--split", default="Debug", choices=["Debug", "Main", "HeldOut"])
    p.add_argument("--out_csv", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max_videos", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--repo", default="~/vis_emb/vjepa2")
    p.add_argument("--ckpt", default=None, help="default: <repo>/weights/vitg-384.pt")
    p.add_argument("--device", default=None)
    # Surprise window knobs (see surprise.video_surprise).
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--t_ctx", type=int, default=4)
    p.add_argument("--t_pred", type=int, default=4)
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()

    import pandas as pd
    from intphys.surprise import VJEPA2Surprise

    root = os.path.expanduser(args.intphys_root)
    meta = pd.read_csv(os.path.join(root, args.split, "metadata.csv"))
    if args.max_videos:
        meta = meta.iloc[:args.max_videos]
    has_labels = "type" in meta.columns and meta["type"].notna().any()

    model = VJEPA2Surprise(repo=args.repo, ckpt=args.ckpt, device=args.device)
    print(f"{len(meta)} videos | split={args.split} | model={model.model_id} "
          f"| device={model.device}")

    rows = []
    for n, rec in enumerate(meta.itertuples(index=False)):
        rec = rec._asdict()
        vid_path = os.path.join(root, args.split, rec["file_name"])
        try:
            frames = load_video(vid_path, args.fps)
            s = model.video_surprise(
                frames, window=args.window, t_ctx=args.t_ctx,
                t_pred=args.t_pred, stride=args.stride,
            )
        except Exception as e:                       # never lose a whole run to one bad file
            print(f"  [skip] {rec['name']}: {e}")
            continue
        rows.append({
            "filename": rec["name"], "SceneIndex": rec.get("SceneIndex"),
            "type": rec.get("type", ""), "env": rec.get("env", ""),
            "surprise_max": s["max"], "surprise_mean": s["mean"],
        })
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(meta)} videos")

    if not rows:
        sys.exit("no videos scored")
    if has_labels:
        assign_response_within_scene(rows)
        rel = relative_pairwise_accuracy(rows)
        print("\nrelative pairwise accuracy (impossible more surprising than possible):")
        for key, (acc, n) in rel.items():
            print(f"  surprise_{key}: {acc:.3f}  over {n} matched pairs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    import csv
    cols = ["filename", "response", "surprise_max", "surprise_mean",
            "SceneIndex", "type", "env"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
