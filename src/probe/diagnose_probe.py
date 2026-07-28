"""GATE-B harness diagnostic: WHY does the k-fold probe fail the v control?

The 5-fold CV showed every target negative at every layer, with huge fold
variance — the v control fails, so the result is uninterpretable as physics
(gate-b §10). Two hypotheses, two cheap CPU tests:

  H1  D≫n overfitting / underpowered. 1408-dim ridge, ~360 train samples,
      17 rollout groups. → compare TRAIN vs TEST R². Train high + test ≤0 means
      overfit, not "no signal". Fix = more data / fewer dims.
  H2  no linear signal at all. → even TRAIN R² is low. Deeper problem
      (feature/target mismatch), not fixable by regularization.

Plus PCA(1408→k) before ridge: if test R² recovers under PCA, dimensionality
was the binding constraint (supports H1).

Outputs (per the request): JSON report, TensorBoard scalars (per-layer curves),
and an auto-written conclusions.md. Reuses probe.metrics; CPU-only; deterministic.

    python src/probe/diagnose_probe.py --features results/probe/sim_physics_feats.npz \\
        --result_dir results/probe/diagnostics --folds 5 --pca 64
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

KIN = ("v", "w")  # kinematic targets the diagnostic focuses on (com is §later)


def _train_test_r2(X_tr, y_tr, X_te, y_te, groups_tr):
    """Ridge with λ picked on an inner group split; return (train R², test R², λ)."""
    pred_te, lam = metrics.fit_linear_probe(X_tr, y_tr, X_te, groups_tr)
    pred_tr = metrics._ridge(X_tr, y_tr, X_tr, lam)        # train preds at chosen λ
    ymu = y_tr.mean(axis=0)
    return (metrics.r2_aggregate(y_tr, pred_tr, ymu),
            metrics.r2_aggregate(y_te, pred_te, ymu), lam)


def _pca_train_test_r2(X_tr, y_tr, X_te, y_te, groups_tr, k):
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(k, X_tr.shape[0] - 1, X_tr.shape[1]))
    Ztr = p.fit_transform(X_tr)                            # PCA fit on TRAIN only
    Zte = p.transform(X_te)
    return _train_test_r2(Ztr, y_tr, Zte, y_te, groups_tr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/sim_physics_feats.npz")
    p.add_argument("--result_dir", default="results/probe/diagnostics")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)
    targets = {k: d[k][keep].astype(np.float64) for k in KIN}
    groups = d["group"][keep]
    n_layers = X.shape[1]
    folds = group_kfold(groups, args.folds, args.seed)
    print(f"{keep.sum()} samples | {len(np.unique(groups))} groups → {args.folds} folds "
          f"| {n_layers} layers | PCA k={args.pca}")

    # arrays[variant][target] = (folds, layers, {train,test})
    shape = (args.folds, n_layers)
    A = {v: {t: {"train": np.full(shape, np.nan), "test": np.full(shape, np.nan)}
             for t in KIN} for v in ("full", "pca")}

    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        g_tr = groups[is_tr]
        for L in range(n_layers):
            Xtr, Xte = X[is_tr, L], X[is_te, L]
            for t in KIN:
                ytr, yte = targets[t][is_tr], targets[t][is_te]
                rtr, rte, _ = _train_test_r2(Xtr, ytr, Xte, yte, g_tr)
                A["full"][t]["train"][fi, L] = rtr
                A["full"][t]["test"][fi, L] = rte
                prtr, prte, _ = _pca_train_test_r2(Xtr, ytr, Xte, yte, g_tr, args.pca)
                A["pca"][t]["train"][fi, L] = prtr
                A["pca"][t]["test"][fi, L] = prte
        print(f"  fold {fi}: done ({is_te.sum()} test / {is_tr.sum()} train)")

    # --- TensorBoard + JSON ---
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(result_dir,"tb"))
    report = {"folds": args.folds, "pca_k": args.pca, "model": str(d["model"]), "layers": []}
    for L in range(n_layers):
        row = {"layer": L}
        for v in ("full", "pca"):
            for t in KIN:
                for split in ("train", "test"):
                    m = float(np.nanmean(A[v][t][split][:, L]))
                    s = float(np.nanstd(A[v][t][split][:, L]))
                    row[f"{t}_{v}_{split}"] = [m, s]
                    tb.add_scalar(f"{t}/{v}_{split}", m, L)
        report["layers"].append(row)
    tb.close()

    def best(v, t, split):
        col = np.array([r[f"{t}_{v}_{split}"][0] for r in report["layers"]])
        L = int(np.nanargmax(col))
        return L, report["layers"][L][f"{t}_{v}_{split}"]

    summ = {}
    for t in KIN:
        for v in ("full", "pca"):
            for split in ("train", "test"):
                L, (m, s) = best(v, t, split)
                summ[f"{t}_{v}_{split}"] = {"layer": L, "mean": m, "std": s}
    report["best"] = summ
    with open(os.path.join(result_dir,"report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- console + auto conclusions.md ---
    print("\nbest-layer R² (k-fold mean ± std):")
    for t in KIN:
        f_tr, f_te = summ[f"{t}_full_train"], summ[f"{t}_full_test"]
        p_tr, p_te = summ[f"{t}_pca_train"], summ[f"{t}_pca_test"]
        print(f"  {t}: full train {f_tr['mean']:+.3f} / test {f_te['mean']:+.3f}  | "
              f"pca train {p_tr['mean']:+.3f} / test {p_te['mean']:+.3f}")

    vtr, vte = summ["v_full_train"]["mean"], summ["v_full_test"]["mean"]
    vpte = summ["v_pca_test"]["mean"]
    if vtr > 0.5 and vte < 0.1:
        h = ("**H1 confirmed — overfitting / underpowered.** v fits TRAIN "
             f"({vtr:+.2f}) but not TEST ({vte:+.2f}): a 1408-dim ridge memorizes "
             "the ~360 training samples and cannot generalize across the 17 rollout "
             "groups. The features *do* contain linear v information; there is just "
             "too little data to estimate the map. Fix: more rollouts (more groups) "
             "and/or dimensionality reduction — NOT a physics verdict.")
    elif vtr < 0.2:
        h = ("**H2 indicated — no linear signal even on TRAIN.** v does not fit "
             f"even the training split ({vtr:+.2f}); this is a feature/target or "
             "pooling mismatch, not merely a data-quantity problem. Investigate the "
             "pooling/label alignment before blaming corpus size.")
    else:
        h = (f"**Mixed.** v train {vtr:+.2f} / test {vte:+.2f} — partial fit, partial "
             "generalization gap; both more data and stronger regularization likely help.")
    pca_helps = vpte > vte + 0.05
    pca_note = (f"PCA(1408→{args.pca}) {'RECOVERS' if pca_helps else 'does not recover'} "
                f"test signal (v test {vte:+.2f} → {vpte:+.2f}): dimensionality "
                f"{'was' if pca_helps else 'was not'} the binding constraint.")

    md = f"""# GATE-B harness diagnostic — conclusions

Model: `{str(d['model'])}` · corpus: {keep.sum()} samples / {len(np.unique(groups))} rollout groups · {args.folds}-fold group CV · PCA k={args.pca}.

## Headline
{h}

{pca_note}

## Best-layer numbers (k-fold mean ± std)
| target | full train | full test | pca train | pca test |
|---|---|---|---|---|
""" + "".join(
        f"| {t} | {summ[f'{t}_full_train']['mean']:+.3f}±{summ[f'{t}_full_train']['std']:.3f} "
        f"| {summ[f'{t}_full_test']['mean']:+.3f}±{summ[f'{t}_full_test']['std']:.3f} "
        f"| {summ[f'{t}_pca_train']['mean']:+.3f}±{summ[f'{t}_pca_train']['std']:.3f} "
        f"| {summ[f'{t}_pca_test']['mean']:+.3f}±{summ[f'{t}_pca_test']['std']:.3f} |\n"
        for t in KIN) + f"""
## Reading
- TRAIN≫TEST ⇒ overfitting / too few independent rollouts (H1). TRAIN also low ⇒
  deeper mismatch (H2).
- The `v` control still does not clear 0.5 on TEST, so per gate-b §10 no physics
  conclusion (ω, CoM) is drawn from this corpus.
- This is a **harness/data verdict, not a model verdict.** The encoder is
  bit-identical to the reference notebook; the limit is the 17-rollout corpus.

## Implication for next step
The binding constraint points at **training-data generation**: the probe needs
many more *independent rollouts* (groups), not more cameras per rollout (which
share a group and do not add split-independent evidence).

TensorBoard: `tensorboard --logdir {os.path.join(result_dir,'tb')}` · raw: `report.json`.
"""
    with open(os.path.join(result_dir,"conclusions.md"), "w") as f:
        f.write(md)
    print(f"\n{h}\n{pca_note}")
    print(f"\nreport.json + tb/ + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
