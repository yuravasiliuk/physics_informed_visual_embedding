"""Variant supervision targets over the SAME cached features (E2.1).

gate_b_v2 (755 groups) broke the v1 plateau (v test −0.02 → +0.31 @ layer 32)
but the v control still misses the 0.5 bar. With n_train ≈ 2800 ≫ D = 1408 and
tight fold variance, overfit is no longer the default suspect; the top suspect
is target↔pooling-window mismatch: extract pools features over ALL sampled
frames, while the Tier-1 targets are INSTANTANEOUS center-frame state
(dataset.py). This builds window-aligned / motion-gated / size-normalized
target variants, row-aligned to an existing feature cache, so fit_matrix.py
can A/B them on identical features and identical group folds
(EXPERIMENTS.md ACTIVE node; probe-walkthrough.md §2; probe-corpus-spec.md §4).

    v_clipmean / w_clipmean   mean state over the SAME frames the encoder saw
                              (window-aligned with the pooled features)
    v_peak / w_peak           state at the max-|v| (resp. max-|w|) sampled
                              frame — a whole-clip statistic, matched to
                              whole-clip pooling; also gives settled-at-center
                              bodies (25% of samples) a live target
    com_norm                  com_offset / (bbox_diag/2) — dimensionless
                              offset; kills the +0.44 |CoM|~bbox size route
                              (coverage_v2; R7/R8)

CPU-only: rereads state.json + meta.json, never video, never the GPU.

    python src/probe/build_targets_v2.py \\
        --features results/probe/probe_corpus_feats.npz \\
        --data ~/data/probe_corpus --result_path results/probe/targets_v2.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sample_idx(frame_count, n_frames):
    """Bit-identical to dataset.load_clip_samples — the frames extract saw."""
    return np.unique(np.linspace(0, frame_count - 1, n_frames).round().astype(int))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/probe_corpus_feats.npz")
    p.add_argument("--data", default="~/data/probe_corpus")
    p.add_argument("--frames", type=int, default=32,
                   help="must match the extract --frames the cache was built with")
    p.add_argument("--result_path", default="results/probe/targets_v2.npz")
    args = p.parse_args()
    root = os.path.expanduser(args.data)

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    clips = d["clip"]
    names = d["name"]
    n = len(clips)
    v_cm = np.full((n, 3), np.nan)
    w_cm = np.full((n, 3), np.nan)
    v_pk = np.full((n, 3), np.nan)
    w_pk = np.full((n, 3), np.nan)

    rows_by_clip = {}
    for i, c in enumerate(clips):
        rows_by_clip.setdefault(str(c), []).append(i)

    missing = 0
    for c, idxs in rows_by_clip.items():
        with open(os.path.join(root, c, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(root, c, "state.json")) as f:
            state = json.load(f)
        frames = set(_sample_idx(int(meta["frame_count"]), args.frames).tolist())
        recs = {}
        for r in state:
            if r["frame"] in frames:
                recs.setdefault(r["body"], []).append(r)
        for i in idxs:
            rs = recs.get(str(names[i]))
            if not rs:
                missing += 1
                continue
            v = np.array([r["lin_vel_camera"] for r in rs], float)
            w = np.array([r["ang_vel_camera"] for r in rs], float)
            v_cm[i], w_cm[i] = v.mean(0), w.mean(0)
            v_pk[i] = v[np.linalg.norm(v, axis=1).argmax()]
            w_pk[i] = w[np.linalg.norm(w, axis=1).argmax()]

    com_norm = d["com_offset"].astype(np.float64) / (d["bbox_diag"][:, None] / 2.0)

    out = os.path.expanduser(args.result_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        clip=clips, name=names,               # alignment keys for fit_matrix
        v_clipmean=v_cm.astype(np.float32), w_clipmean=w_cm.astype(np.float32),
        v_peak=v_pk.astype(np.float32), w_peak=w_pk.astype(np.float32),
        com_norm=com_norm.astype(np.float32),
        frames=np.asarray(args.frames),
    )
    vmag_i = np.linalg.norm(d["v"], axis=1)
    vmag_c = np.linalg.norm(v_cm, axis=1)
    ok = np.isfinite(vmag_c)
    print(f"{n} rows | {missing} without state records (NaN, dropped at fit) | "
          f"median |v| instant {np.median(vmag_i):.3f} vs clip-mean "
          f"{np.median(vmag_c[ok]):.3f} m/s")
    print(f"targets_v2 -> {out}")


if __name__ == "__main__":
    main()
