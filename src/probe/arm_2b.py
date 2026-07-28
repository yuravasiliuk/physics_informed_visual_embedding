"""Arm 2b — small nonlinear (MLP) head on the SAME cached frozen features.

The last probe-side test of the CoM null, and the ω lift test (gate-b-probe.md
arm 2b; probe-walkthrough.md §2.2 point 3). Evidence chain that puts us here
(EXPERIMENTS.md ACTIVE node, all on the 755-group v2 corpus):

    linear (2a):        v_clipmean 0.60 @ L32 (CONTROL PASSES) · ω 0.38 @ L30
                        · CoM lift ≈ 0 at every layer, raw AND normalized
    scenario split:     the CoM null is UNIFORM (no contact-rich rescue; R6)
    register collapse:  RULED OUT — body↔frame cosine FALLS with depth
                        (0.96 → 0.56), objects are most distinct at depth

So: same features, same group k-fold, same metrics — only the hypothesis class
grows (ridge → 2-hidden-layer MLP). The 2b − 2a gap is "nonlinearly
recoverable"; a CoM null that survives 2b is no longer a probe artifact and
the remaining hypotheses are pooling granularity (per-token probe), corpus
observability at the probed frames (topple W3a; collide count), or genuine
absence — i.e. GATE B reads "kinematics yes, latent intrinsics no".

Also fits one CONCAT variant (features of several layers stacked) as the
cheapest preview of the Rung-1 multi-layer head (BRAINSTORM §2.2).

    python src/probe/arm_2b.py --features results/probe/probe_corpus_feats.npz \\
        --targets results/probe/targets_v2.npz --result_dir results/probe/arm_2b
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

TARGETS_2B = {          # name -> (source key, kind)
    "v_clipmean": ("t:v_clipmean", "r2"),   # control — must reproduce ≥ 2a
    "w_clipmean": ("t:w_clipmean", "r2"),
    "com_norm":   ("t:com_norm", "com"),
    "com_offset": ("d:com_offset", "com"),
}


def fit_mlp(X_tr, y_tr, X_te, groups_tr, seed, device, epochs=300, patience=25):
    """Standardized 2-hidden-layer MLP with inner-group-split early stopping.
    Mirrors the ridge protocol shape: model selection touches only train."""
    import torch
    torch.manual_seed(seed)
    xm, xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    ym, ysd = y_tr.mean(0), y_tr.std(0) + 1e-8
    in_tr, in_val = metrics.group_split(groups_tr, test_frac=0.2, salt="2b-val")
    if in_val.sum() == 0 or in_tr.sum() == 0:
        in_tr = in_val = np.ones(len(groups_tr), dtype=bool)

    def T(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    Xa, Xv = T((X_tr[in_tr] - xm) / xs), T((X_tr[in_val] - xm) / xs)
    ya, yv = T((y_tr[in_tr] - ym) / ysd), T((y_tr[in_val] - ym) / ysd)
    net = torch.nn.Sequential(
        torch.nn.Linear(X_tr.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 128), torch.nn.GELU(),
        torch.nn.Linear(128, y_tr.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best_err, best_state, bad = np.inf, None, 0
    idx = torch.arange(len(Xa), device=device)
    for ep in range(epochs):
        net.train()
        for b in idx[torch.randperm(len(idx), device=device)].split(512):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xa[b]), ya[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            err = float(torch.nn.functional.mse_loss(net(Xv), yv))
        if err < best_err - 1e-5:
            best_err, bad = err, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(T((X_te - xm) / xs)).cpu().numpy()
    return pred * ysd + ym


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/probe_corpus_feats.npz")
    p.add_argument("--targets", default="results/probe/targets_v2.npz")
    p.add_argument("--ref", default="results/probe/fit_matrix/report.json",
                   help="2a report for side-by-side deltas (optional)")
    p.add_argument("--result_dir", default="results/probe/arm_2b")
    p.add_argument("--layers", default="21,26,28,30,32,34")
    p.add_argument("--concat", default="21,30,32",
                   help='layers to stack for the Rung-1 preview ("" disables)')
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    t = np.load(os.path.expanduser(args.targets), allow_pickle=False)
    if not (np.array_equal(d["clip"], t["clip"]) and np.array_equal(d["name"], t["name"])):
        sys.exit("targets not row-aligned with the cache — rebuild targets_v2")
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)
    groups = d["group"][keep]
    bbox = d["bbox_diag"][keep].astype(np.float64)
    ys = {n: (d if s.split(":")[0] == "d" else t)[s.split(":")[1]][keep].astype(np.float64)
          for n, (s, _) in TARGETS_2B.items()}

    layers = sorted({int(x) for x in args.layers.split(",")})
    concat = sorted({int(x) for x in args.concat.split(",")}) if args.concat else []
    slots = [("L%d" % L, [L]) for L in layers]
    if concat:
        slots.append(("concat" + "+".join(map(str, concat)), concat))
    folds = group_kfold(groups, args.folds, args.seed)
    print(f"{keep.sum()} samples | {len(np.unique(groups))} groups → {args.folds} folds "
          f"| slots {[s for s, _ in slots]} | device {device}")

    score = {s: {n: np.full(args.folds, np.nan) for n in TARGETS_2B} for s, _ in slots}
    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        for sname, Ls in slots:
            Xtr = np.concatenate([X[is_tr, L] for L in Ls], axis=1)
            Xte = np.concatenate([X[is_te, L] for L in Ls], axis=1)
            for tname, (_, kind) in TARGETS_2B.items():
                y = ys[tname]
                ok_tr = np.isfinite(y[is_tr]).all(axis=1)
                ok_te = np.isfinite(y[is_te]).all(axis=1)
                y_tr, y_te = y[is_tr][ok_tr], y[is_te][ok_te]
                pred = fit_mlp(Xtr[ok_tr], y_tr, Xte[ok_te],
                               groups[is_tr][ok_tr],
                               seed=args.seed * 1000 + fi, device=device)
                if kind == "r2":
                    score[sname][tname][fi] = metrics.r2_aggregate(
                        y_te, pred, y_tr.mean(0))
                else:
                    score[sname][tname][fi] = metrics.com_metrics(
                        y_te, pred, bbox[is_te][ok_te])["lift"]
        print(f"  fold {fi}: done")

    ref = {}
    try:  # 2a per-layer numbers for the delta column
        rj = json.load(open(os.path.expanduser(args.ref)))
        ref = {n: {int(L): v[0] for L, v in
                   rj["variants"][n]["per_layer"].items()}
               for n in TARGETS_2B if n in rj.get("variants", {})}
    except Exception:
        pass

    report = {"folds": args.folds, "seed": args.seed, "model": str(d["model"]),
              "slots": {}}
    print(f"\n{'slot':16s} | " + " | ".join(f"{n:>11s}" for n in TARGETS_2B))
    for sname, Ls in slots:
        row = {}
        cells = []
        for tname in TARGETS_2B:
            m = float(np.nanmean(score[sname][tname]))
            s = float(np.nanstd(score[sname][tname]))
            row[tname] = [m, s]
            delta = ""
            if len(Ls) == 1 and tname in ref and Ls[0] in ref[tname]:
                delta = f" (2a {ref[tname][Ls[0]]:+.3f})"
            cells.append(f"{m:+.3f}±{s:.3f}{delta}")
        report["slots"][sname] = row
        print(f"{sname:16s} | " + " | ".join(f"{c:>11s}" for c in cells))

    best = {n: max(report["slots"].items(), key=lambda kv: kv[1][n][0])
            for n in TARGETS_2B}
    com_m, com_s = best["com_norm"][1]["com_norm"]
    w_m, _ = best["w_clipmean"][1]["w_clipmean"]
    verdict = []
    verdict.append(f"control v_clipmean best {best['v_clipmean'][1]['v_clipmean'][0]:+.3f} "
                   f"@ {best['v_clipmean'][0]} — "
                   + ("OK (≥ 2a level, harness sound)."
                      if best['v_clipmean'][1]['v_clipmean'][0] > 0.5 else
                      "BELOW 2a/bar — investigate the MLP protocol before reading rows."))
    verdict.append(f"ω (w_clipmean) best {w_m:+.3f} @ {best['w_clipmean'][0]} — "
                   + ("clears 0.5 nonlinearly → GATE-B positive for ω." if w_m > 0.5
                      else "still short of 0.5; nonlinear gain "
                           "quantifies the 2b−2a recoverability gap."))
    verdict.append(f"CoM (com_norm) best lift {com_m:+.4f}±{com_s:.4f} @ {best['com_norm'][0]} — "
                   + ("POSITIVE beyond noise → nonlinear CoM signal exists; tap depth found."
                      if com_m - com_s > 0 else
                      "null SURVIVES the nonlinear head. Probe-side excuses exhausted "
                      "(linear, normalized, scenario-split, register-collapse, MLP, "
                      "multi-layer concat). Remaining hypotheses: per-token pooling "
                      "granularity; corpus observability at probed frames (topple W3a, "
                      "collide count); genuine absence → GATE-B reads "
                      "'kinematics YES, latent intrinsics NO from clip-pooled features'."))
    print("\n" + "\n".join(verdict))

    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(result_dir, "conclusions.md"), "w") as f:
        f.write("# Arm 2b (MLP) — conclusions\n\n"
                f"{keep.sum()} samples / {len(np.unique(groups))} groups / "
                f"{args.folds}-fold / slots {[s for s, _ in slots]} / "
                f"`{str(d['model'])}`; same folds+metrics as 2a, hypothesis "
                "class ridge→MLP(512,128).\n\n"
                "| slot | " + " | ".join(TARGETS_2B) + " |\n"
                "|---|" + "---|" * len(TARGETS_2B) + "\n"
                + "".join("| " + s + " | "
                          + " | ".join(f"{report['slots'][s][n][0]:+.4f}±{report['slots'][s][n][1]:.4f}"
                                       for n in TARGETS_2B) + " |\n"
                          for s, _ in slots)
                + "\n## Verdict\n\n- " + "\n- ".join(verdict)
                + "\n\nRaw: `report.json` · 2a reference: `" + args.ref + "`.\n")
    print(f"\nreport.json + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
