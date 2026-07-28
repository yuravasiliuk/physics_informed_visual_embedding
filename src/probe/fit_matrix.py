"""E2.1 fit matrix — variant targets on identical features and identical folds.

The discriminating experiment after gate_b_v2: v control reached +0.31 (from
−0.02) at 755 groups but not 0.5. Three explanations remain; each variant pair
isolates one (EXPERIMENTS.md ACTIVE node):

    v_instant vs v_clipmean   target↔pooling-window mismatch — the features
                              pool the whole clip, the Tier-1 target is one
                              frame; a big gap here means the plateau remnant
                              was window misalignment, not missing signal
    v_clipmean vs v_peak      motion gating — do settled-at-center bodies
                              (25%) with ~0 instantaneous targets dilute R²?
    com_offset vs com_norm    the +0.44 |CoM|~bbox size route (coverage_v2):
                              normalized offset is size-free (R7/R8)

Same ridge, same group k-fold (replicate_gate_b.group_kfold, same seed), same
metrics module as gate_b_v2 — only the y changes, so any delta is attributable
to the target formulation alone. CPU-only.

Reading (probe-walkthrough.md §2.2 discipline still applies):
  * the control is now v_clipmean (window-aligned) — if IT clears 0.5 the
    harness is fine and the aligned targets become the Tier-1 defaults;
  * if no v variant clears ~0.5, window/gating is exonerated → suspect the
    POOLING (per-temporal-token extraction next) or the linear class (arm 2b).

    python src/probe/fit_matrix.py --features results/probe/probe_corpus_feats.npz \\
        --targets results/probe/targets_v2.npz --result_dir results/probe/fit_matrix
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import metrics
from probe.replicate_gate_b import group_kfold
from probe.run_probe import MIN_TOKEN_WEIGHT

# name -> (source, kind): source "cache"/"targets"; kind "r2" | "com" (lift).
VARIANTS = {
    "v_instant":   ("cache:v", "r2"),
    "v_clipmean":  ("targets:v_clipmean", "r2"),
    "v_peak":      ("targets:v_peak", "r2"),
    "w_instant":   ("cache:w", "r2"),
    "w_clipmean":  ("targets:w_clipmean", "r2"),
    "w_peak":      ("targets:w_peak", "r2"),
    "com_offset":  ("cache:com_offset", "com"),
    "com_norm":    ("targets:com_norm", "com"),
}


def _parse_layers(spec, n_layers):
    if spec == "all":
        return list(range(n_layers))
    return sorted({int(x) for x in spec.split(",")})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/probe_corpus_feats.npz")
    p.add_argument("--targets", default="results/probe/targets_v2.npz")
    p.add_argument("--result_dir", default="results/probe/fit_matrix")
    p.add_argument("--layers", default="all",
                   help='"all" (≈2.7× a gate_b run) or e.g. "8,16,24,30,32,36"')
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    t = np.load(os.path.expanduser(args.targets), allow_pickle=False)
    if not (np.array_equal(d["clip"], t["clip"]) and np.array_equal(d["name"], t["name"])):
        sys.exit("targets file is not row-aligned with the feature cache — "
                 "rebuild with build_targets_v2.py against THIS cache")

    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)
    groups = d["group"][keep]
    bbox = d["bbox_diag"][keep].astype(np.float64)
    ys = {}
    for name, (src, kind) in VARIANTS.items():
        store, key = src.split(":")
        ys[name] = (d if store == "cache" else t)[key][keep].astype(np.float64)

    layers = _parse_layers(args.layers, X.shape[1])
    folds = group_kfold(groups, args.folds, args.seed)
    print(f"{keep.sum()} samples | {len(np.unique(groups))} groups → {args.folds} folds "
          f"| layers {layers[0]}..{layers[-1]} ({len(layers)}) | {len(VARIANTS)} variants")

    # score[variant] = (folds, layers) of R² (r2 kind) or lift (com kind).
    score = {v: np.full((args.folds, len(layers)), np.nan) for v in VARIANTS}
    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        for li, L in enumerate(layers):
            Xtr, Xte = X[is_tr, L], X[is_te, L]
            for vname, (src, kind) in VARIANTS.items():
                y = ys[vname]
                ok_tr = np.isfinite(y[is_tr]).all(axis=1)   # NaN rows: no state recs
                ok_te = np.isfinite(y[is_te]).all(axis=1)
                y_tr, y_te = y[is_tr][ok_tr], y[is_te][ok_te]
                pred, _ = metrics.fit_linear_probe(
                    Xtr[ok_tr], y_tr, Xte[ok_te], groups[is_tr][ok_tr])
                if kind == "r2":
                    score[vname][fi, li] = metrics.r2_aggregate(
                        y_te, pred, y_tr.mean(axis=0))
                else:
                    score[vname][fi, li] = metrics.com_metrics(
                        y_te, pred, bbox[is_te][ok_te])["lift"]
        print(f"  fold {fi}: done ({is_te.sum()} test)")

    def mstd(v):
        return np.nanmean(score[v], 0), np.nanstd(score[v], 0)

    report = {"folds": args.folds, "seed": args.seed, "layers": layers,
              "model": str(d["model"]), "variants": {}}
    best = {}
    print(f"\n{'variant':12s} | best layer | score (mean±std)")
    for v in VARIANTS:
        m, s = mstd(v)
        bi = int(np.nanargmax(m))
        best[v] = (layers[bi], float(m[bi]), float(s[bi]))
        report["variants"][v] = {
            "kind": VARIANTS[v][1], "best_layer": layers[bi],
            "best": [float(m[bi]), float(s[bi])],
            "per_layer": {int(L): [float(m[i]), float(s[i])]
                          for i, L in enumerate(layers)},
        }
        print(f"{v:12s} |   {layers[bi]:3d}    | {m[bi]:+.4f} ± {s[bi]:.4f}")

    # Headline deltas — the three questions this matrix exists to answer.
    dv_window = best["v_clipmean"][1] - best["v_instant"][1]
    dv_peak = best["v_peak"][1] - best["v_instant"][1]
    dcom = best["com_norm"][1] - best["com_offset"][1]  # lifts, scale differs
    control = best["v_clipmean"][1]
    verdict = [
        f"window alignment (v_clipmean − v_instant): {dv_window:+.3f}",
        f"motion gating (v_peak − v_instant): {dv_peak:+.3f}",
        f"CoM normalization (com_norm − com_offset lift; scales differ): {dcom:+.4f}",
        ("CONTROL PASSES: v_clipmean R² > 0.5 — adopt window-aligned targets "
         "as Tier-1 and read ω/CoM rows (scenario-split)." if control > 0.5 else
         "CONTROL STILL < 0.5 on every v variant — window/gating exonerated; "
         "next suspect is the whole-clip POOLING (per-temporal-token extract) "
         "or the linear class (arm 2b)." if max(best[v][1] for v in
         ("v_instant", "v_clipmean", "v_peak")) <= 0.5 else
         f"MIXED: best v variant "
         f"{max(('v_instant','v_clipmean','v_peak'), key=lambda v: best[v][1])} "
         f"at {max(best[v][1] for v in ('v_instant','v_clipmean','v_peak')):.3f}."),
    ]
    print("\n" + "\n".join(verdict))

    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fams = (("v", ("v_instant", "v_clipmean", "v_peak"), "R²"),
            ("ω", ("w_instant", "w_clipmean", "w_peak"), "R²"),
            ("CoM", ("com_offset", "com_norm"), "lift"))
    for ax, (fam, vs, ylab) in zip(axes, fams):
        for v in vs:
            m, s = mstd(v)
            ax.plot(layers, m, "-o", ms=3, label=v)
            ax.fill_between(layers, m - s, m + s, alpha=0.15)
        if ylab == "R²":
            ax.axhline(0.5, ls=":", c="green", lw=1)
        ax.axhline(0, ls="-", c="0.7", lw=0.8)
        ax.set_xlabel("layer"); ax.set_ylabel(ylab); ax.set_title(fam); ax.legend()
    fig.suptitle(f"E2.1 fit matrix — {keep.sum()} samples / "
                 f"{len(np.unique(groups))} groups / {args.folds}-fold")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "fit_matrix.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(result_dir, "conclusions.md"), "w") as f:
        f.write("# E2.1 fit matrix — conclusions\n\n"
                + f"{keep.sum()} samples / {len(np.unique(groups))} groups / "
                  f"{args.folds}-fold / layers {args.layers} / model "
                  f"`{str(d['model'])}`.\n\n"
                + "| variant | kind | best layer | score |\n|---|---|---|---|\n"
                + "".join(f"| {v} | {VARIANTS[v][1]} | {best[v][0]} | "
                          f"{best[v][1]:+.4f} ± {best[v][2]:.4f} |\n" for v in VARIANTS)
                + "\n## Verdict\n\n- " + "\n- ".join(verdict) + "\n\n"
                + "Plot: `fit_matrix.png` · raw: `report.json`. Same folds/ridge/"
                  "metrics as `gate_b_v2`; only the target formulation varies.\n")
    print(f"\nreport.json + fit_matrix.png + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
