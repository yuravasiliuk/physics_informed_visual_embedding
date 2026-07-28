"""Learning curve: probe test-R² vs number of independent rollout groups.

De-risks/sizes the overnight data-gen run. The diagnostic said the probe is
group-starved (train R² 0.90, test ~0, PCA-64 recovers +0.11); this asks the
quantitative follow-up: *how fast does test R² climb with #groups, and how many
groups to clear the v-control bar?*

Method: subsample G of the 17 rollout groups (several random draws per G), run
group k-fold PCA-64 ridge at the diagnostic's best PCA layer, record mean test
R². The slope answers "is overnight gen worth it" and roughly "how big".
Extrapolation from 17 groups is low-confidence — reported as a trend, not a
promise. CPU-only; reuses diagnose_probe / replicate_gate_b helpers.

    python src/probe/learning_curve.py --features results/probe/sim_physics_feats.npz \\
        --result_dir results/probe/learning_curve
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.diagnose_probe import _pca_train_test_r2
from probe.replicate_gate_b import group_kfold
from probe.run_probe import MIN_TOKEN_WEIGHT


def _best_pca_layer(out_dir, target, default):
    """Reuse the layer the diagnostic already found best for PCA test R²."""
    rep = os.path.join(out_dir, "..", "diagnostics", "report.json")
    try:
        with open(rep) as f:
            return json.load(f)["best"][f"{target}_pca_test"]["layer"]
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/sim_physics_feats.npz")
    p.add_argument("--result_dir", default="results/probe/learning_curve")
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--target_r2", type=float, default=0.5, help="v-control bar")
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)
    yv = d["v"][keep].astype(np.float64)
    yw = d["w"][keep].astype(np.float64)
    groups = d["group"][keep]
    uniq = np.unique(groups)
    G_total = len(uniq)
    Lv = _best_pca_layer(result_dir, "v", 20)
    Lw = _best_pca_layer(result_dir, "w", 20)
    print(f"{keep.sum()} samples | {G_total} groups | PCA-{args.pca} "
          f"| layers v={Lv} w={Lw} | {args.repeats} repeats/point")

    grid = [g for g in (5, 7, 9, 11, 13, 15, G_total) if g <= G_total]
    curve = {"v": {}, "w": {}}
    rows = []
    for G in grid:
        accv, accw = [], []
        for r in range(args.repeats):
            rng = np.random.default_rng(1000 * G + r)
            sub = rng.choice(uniq, size=G, replace=False)
            m = np.isin(groups, sub)
            gs = groups[m]
            folds = group_kfold(gs, min(5, G), seed=r)
            fv, fw = [], []
            for test_g in folds:
                te = np.isin(gs, test_g); tr = ~te
                if te.sum() == 0 or tr.sum() < 5:
                    continue
                _, rte_v, _ = _pca_train_test_r2(
                    X[m][tr, Lv], yv[m][tr], X[m][te, Lv], yv[m][te], gs[tr], args.pca)
                _, rte_w, _ = _pca_train_test_r2(
                    X[m][tr, Lw], yw[m][tr], X[m][te, Lw], yw[m][te], gs[tr], args.pca)
                fv.append(rte_v); fw.append(rte_w)
            if fv:
                accv.append(np.mean(fv)); accw.append(np.mean(fw))
        curve["v"][G] = (float(np.mean(accv)), float(np.std(accv)))
        curve["w"][G] = (float(np.mean(accw)), float(np.std(accw)))
        rows.append((G, *curve["v"][G], *curve["w"][G]))
        print(f"  G={G:2d} | v test {curve['v'][G][0]:+.3f} ± {curve['v'][G][1]:.3f} "
              f"| w test {curve['w'][G][0]:+.3f} ± {curve['w'][G][1]:.3f}")

    # --- trend + rough sizing (linear in #groups over the upper half; low confidence) ---
    Gs = np.array([r[0] for r in rows], float)
    vm = np.array([r[1] for r in rows])
    hi = Gs >= np.median(Gs)
    slope, intercept = np.polyfit(Gs[hi], vm[hi], 1) if hi.sum() >= 2 else (np.nan, np.nan)
    if np.isfinite(slope) and slope > 1e-4:
        g_target = (args.target_r2 - intercept) / slope
        sizing = (f"v test rises ≈ {slope:+.4f} R²/group over the upper range; "
                  f"linear extrapolation reaches R²={args.target_r2:.2f} near "
                  f"**~{int(round(g_target))} groups** (LOW CONFIDENCE — extrapolated "
                  f"from {G_total}).")
    else:
        sizing = ("v test is flat/declining over the upper range — more groups may "
                  "NOT clear the bar; investigate dimensionality/features before scaling.")

    verdict = ("MORE GROUPS HELP — overnight gen justified." if slope > 1e-3
               else "Gain marginal — reconsider before spending a night.")
    print(f"\n{verdict}\n{sizing}")

    # --- outputs: json, tensorboard, plot, md ---
    with open(os.path.join(result_dir,"curve.json"), "w") as f:
        json.dump({"grid": grid, "v": curve["v"], "w": curve["w"],
                   "slope_v_per_group": float(slope), "intercept_v": float(intercept),
                   "layers": {"v": int(Lv), "w": int(Lw)}}, f, indent=2)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(result_dir,"tb"))
    for G in grid:
        tb.add_scalar("v/test_vs_groups", curve["v"][G][0], G)
        tb.add_scalar("w/test_vs_groups", curve["w"][G][0], G)
    tb.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, c in (("v", "C0"), ("w", "C1")):
        m = np.array([curve[t][G][0] for G in grid])
        s = np.array([curve[t][G][1] for G in grid])
        ax.plot(grid, m, "-o", color=c, label=f"{t} test R²")
        ax.fill_between(grid, m - s, m + s, color=c, alpha=0.2)
    ax.axhline(args.target_r2, ls=":", c="green", label=f"v bar {args.target_r2}")
    ax.axhline(0, ls="-", c="0.7", lw=0.8)
    if np.isfinite(slope):
        xx = np.linspace(grid[0], max(grid[-1] * 3, 60), 50)
        ax.plot(xx, slope * xx + intercept, "--", c="C0", alpha=0.5, label="v linear extrap")
    ax.set_xlabel("# independent rollout groups (train+test)")
    ax.set_ylabel("PCA-64 test R² (k-fold mean ± std)")
    ax.set_title(f"probe learning curve — sizing the data-gen run")
    ax.legend(); fig.tight_layout()
    png = os.path.join(result_dir,"learning_curve.png")
    fig.savefig(png, dpi=120); plt.close(fig)

    md = f"""# Probe learning curve — data-gen sizing

Corpus: {keep.sum()} samples / {G_total} groups · PCA-{args.pca} · layers v={Lv}, w={Lw} · {args.repeats} draws/point.

## Verdict
{verdict}

{sizing}

## Curve (PCA-64 test R², mean ± std over draws)
| #groups | v test | w test |
|---|---|---|
""" + "".join(f"| {G} | {curve['v'][G][0]:+.3f}±{curve['v'][G][1]:.3f} "
               f"| {curve['w'][G][0]:+.3f}±{curve['w'][G][1]:.3f} |\n" for G in grid) + f"""
## Reading
- Rising curve ⇒ the probe is data-limited (more independent rollouts → higher
  test R²); flat ⇒ a deeper limit that more data won't fix.
- The extrapolation is a **trend estimate from {G_total} groups**, not a guarantee —
  treat the group target as an order-of-magnitude for the overnight run, then
  re-measure on the bigger corpus.
- Groups = independent **physics rollouts**, NOT camera views. The cheapest way
  to raise this number is fewer cameras/rollout (≈1), converting render budget
  into groups.

Plot: `learning_curve.png` · TensorBoard: `tensorboard --logdir {os.path.join(result_dir,'tb')}`.
"""
    with open(os.path.join(result_dir,"conclusions.md"), "w") as f:
        f.write(md)
    print(f"\ncurve.json + tb/ + learning_curve.png + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
