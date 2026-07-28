"""Standalone, deterministic replication of the GATE-B arm-2a per-layer probe
(Deliverable A). One entry point, fixed config, reproducible numbers.

Pipeline:
  1. ensure the feature cache exists (else call the *validated* extractor in
     run_probe.py — the encoder path proven bit-identical to the reference
     notebook; not re-implemented here);
  2. **group k-fold** ridge per (layer, target) — the single-test-group problem
     of a one-shot split is why a lone `fit` is not reportable; k-fold over the
     rollout groups turns "+0.42 on one rollout" into "mean ± std over folds";
  3. emit: per-layer report JSON, a per-layer plot, and a **split manifest**
     (groups per fold, scenario counts) — the leakage-proof-split evidence the
     supervisor instructions require.

Reuses probe.metrics.evaluate_layer verbatim, so the per-fold math is identical
to run_probe.py's single-split path. CPU-only (the GPU pass is the cached
extract); deterministic given --seed.

    python src/probe/replicate_gate_b.py --data ~/data/sim_physics \\
        --features results/probe/sim_physics_feats.npz \\
        --result_dir results/probe/gate_b --folds 5
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import metrics
from probe.run_probe import MIN_TOKEN_WEIGHT, TARGETS


def group_kfold(groups, k, seed):
    """Deterministic partition of the *unique* rollout groups into k folds
    (round-robin over a seeded permutation), so siblings never split across
    train/test and every group is tested exactly once."""
    uniq = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(uniq))
    return [uniq[order[i::k]] for i in range(k)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/sim_physics",
                   help="clip corpus (only needed if the cache must be built)")
    p.add_argument("--features", default="results/probe/sim_physics_feats.npz")
    p.add_argument("--result_dir", default="results/probe/gate_b", help="output dir")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    feats = os.path.expanduser(args.features)
    # 1. Ensure cache — defer to the validated extractor, do not re-implement it.
    if not os.path.exists(feats):
        print(f"cache {feats} missing → running validated extractor")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "run_probe.py"),
                        "extract", "--data", os.path.expanduser(args.data),
                        "--result_path", feats, "--frames", str(args.frames),
                        "--device", args.device], check=True)

    d = np.load(feats, allow_pickle=False)
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)
    targets = {k: d[k][keep].astype(np.float64) for k in TARGETS}
    groups = d["group"][keep]
    scenarios = d["scenario"][keep]
    bbox = d["bbox_diag"][keep].astype(np.float64)
    n_layers = X.shape[1]
    folds = group_kfold(groups, args.folds, args.seed)

    print(f"{keep.sum()} samples | {len(np.unique(groups))} groups → {args.folds} folds "
          f"| {n_layers} layers | model {str(d['model'])}")

    # 2. k-fold: per (fold, layer) v/w R² and com lift, via the same evaluate_layer.
    vr = np.full((args.folds, n_layers), np.nan)
    wr = np.full((args.folds, n_layers), np.nan)
    cl = np.full((args.folds, n_layers), np.nan)
    manifest = {"folds": [], "model": str(d["model"]), "seed": args.seed}
    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        t_tr = {k: v[is_tr] for k, v in targets.items()}
        t_te = {k: v[is_te] for k, v in targets.items()}
        for layer in range(n_layers):
            res = metrics.evaluate_layer(
                X[is_tr, layer], X[is_te, layer], t_tr, t_te,
                groups[is_tr], scenarios[is_te], bbox[is_te])
            vr[fi, layer] = res["v"]["r2"]
            wr[fi, layer] = res["w"]["r2"]
            cl[fi, layer] = res["com_offset"]["lift"]
        scen, cnt = np.unique(scenarios[is_te], return_counts=True)
        manifest["folds"].append({
            "fold": fi, "test_groups": [int(g) for g in test_groups],
            "n_test": int(is_te.sum()), "n_train": int(is_tr.sum()),
            "test_scenarios": dict(zip(scen.tolist(), cnt.tolist())),
        })

    # 3. aggregate, report, headline.
    def stat(a): return float(np.nanmean(a)), float(np.nanstd(a))
    layers = []
    print("\nlayer | v R2 (mean±std)   | w R2 (mean±std)   | com lift (mean±std)")
    for L in range(n_layers):
        vm, vs = stat(vr[:, L]); wm, ws = stat(wr[:, L]); cm, cs = stat(cl[:, L])
        layers.append({"layer": L, "v_r2": [vm, vs], "w_r2": [wm, ws], "com_lift": [cm, cs]})
        print(f"  {L:3d} | {vm:+.3f} ± {vs:.3f}    | {wm:+.3f} ± {ws:.3f}    | "
              f"{cm:+.4f} ± {cs:.4f}")

    bv = max(layers, key=lambda r: r["v_r2"][0])
    bw = max(layers, key=lambda r: r["w_r2"][0])
    bc = max(layers, key=lambda r: r["com_lift"][0])
    print(f"\nbest (k-fold mean): v=layer {bv['layer']} {bv['v_r2'][0]:+.3f}±{bv['v_r2'][1]:.3f}"
          f" | w=layer {bw['layer']} {bw['w_r2'][0]:+.3f}±{bw['w_r2'][1]:.3f}"
          f" | com=layer {bc['layer']} lift {bc['com_lift'][0]:+.4f}±{bc['com_lift'][1]:.4f}")
    verdict = ("HARNESS OK" if bv["v_r2"][0] > 0.5 else
               "HARNESS WEAK: v control ≤ 0.5 — interpret physics rows with caution")
    print(verdict)

    os.makedirs(os.path.expanduser(args.result_dir), exist_ok=True)
    report = {"arm": "2a", "folds": args.folds, "seed": args.seed,
              "best_layer": {"v": bv["layer"], "w": bw["layer"], "com_offset": bc["layer"]},
              "v_control_ok": bool(bv["v_r2"][0] > 0.5), "layers": layers}
    with open(os.path.join(os.path.expanduser(args.result_dir), "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(os.path.expanduser(args.result_dir), "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = np.arange(n_layers)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, arr, lab in ((a1, vr, "v R²"), (a1, wr, "ω R²")):
        m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
        ax.plot(Ls, m, "-o", ms=3, label=lab)
        ax.fill_between(Ls, m - s, m + s, alpha=0.2)
    a1.axhline(0.5, ls=":", c="green", lw=1, label="v-control bar")
    a1.axhline(0, ls="-", c="0.7", lw=0.8)
    a1.set_xlabel("layer"); a1.set_ylabel("R² (k-fold mean ± std)"); a1.legend()
    a1.set_title("v / ω recoverability by layer")
    m, s = np.nanmean(cl, 0), np.nanstd(cl, 0)
    a2.plot(Ls, m, "-o", ms=3, color="crimson"); a2.fill_between(Ls, m - s, m + s, alpha=0.2, color="crimson")
    a2.axhline(0, ls="-", c="0.7", lw=0.8)
    a2.set_xlabel("layer"); a2.set_ylabel("CoM lift over centroid (mean ± std)")
    a2.set_title("CoM-offset lift by layer (>0 = beats baseline)")
    fig.suptitle(f"GATE-B arm 2a — {args.folds}-fold, {keep.sum()} samples, {str(d['model'])}")
    fig.tight_layout()
    png = os.path.join(os.path.expanduser(args.result_dir), "per_layer.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"\nreport + manifest + {png} -> {os.path.expanduser(args.result_dir)}")


if __name__ == "__main__":
    main()
