"""Coverage audit of the probe corpus — is the +0.05 plateau range, confound, or groups?

The learning curve left three candidate co-limiters for the frozen-probe plateau
(out/probe/learning_curve/conclusions.md): too few independent rollout GROUPS,
RANGE-starved targets (nothing to decode), or CONFOUNDED targets (the probe can
cheat via a visible proxy like size). This audit measures the latter two directly
off the cached features so we know which lever the overnight data-gen must pull —
BEFORE settling the non-retrofittable knobs (density / CoM-offset / friction ranges).

It reports, over the SAME samples the probe sees (weight ≥ MIN_TOKEN_WEIGHT):
  1. RANGE — per-target spread, dynamic range (p95/p5), and the near-zero mass of
     |v|/|ω| (settled objects whose ~0 label inflates SST and is trivially
     "predicted" by the mean). A range-starved target cannot be decoded at any N.
  2. CONFOUND — sample-level Pearson among |v|, |ω|, |CoM|, and bbox_diag (size).
     A target strongly tied to bbox_diag means a positive probe could be decoding
     SIZE (visible) not physics (R8). These are the couplings the stratified
     sampler must break: density⊥size, ω₀⊥v₀, CoM⊥shape.
  3. COVERAGE — samples & independent groups per scenario / observability regime /
     body kind. Few groups per regime ⇒ that regime is group-starved regardless of
     total N (the probe splits on groups).

Density/mass are NOT in the feature cache (they live in clip meta.json); the
density⊥size confound therefore needs a separate meta pass — flagged, not done here.

CPU-only, deterministic, no Isaac. Emits report.md + histograms.png + audit.json.

    python src/probe/coverage_audit.py --features results/probe/sim_physics_feats.npz \\
        --result_dir results/probe/coverage
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.run_probe import MIN_TOKEN_WEIGHT

# |v|, |ω| below this (in their raw units, m/s and rad/s) count as "settled":
# near-zero labels that the mean predicts for free and so inflate SST without
# carrying decodable physics. Coarse, only used to quantify the near-zero mass.
SETTLED_V = 0.05
SETTLED_W = 0.10


def _summ(x):
    """Distribution summary for a 1-D magnitude array."""
    x = np.asarray(x, float)
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "std": float(x.std()),
        "p5": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
        "p75": float(p[3]), "p95": float(p[4]),
        # Dynamic range the probe can exploit: a flat target (p95≈p5) is
        # range-starved. Guard the divide for targets whose p5 is ~0.
        "dyn_range_p95_p5": float(p[4] / p[0]) if p[0] > 1e-6 else float("inf"),
        # Spread relative to extent: small ⇒ values bunched inside their range.
        "spread_ratio": float(x.std() / (x.max() - x.min())) if x.max() > x.min() else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="results/probe/sim_physics_feats.npz")
    p.add_argument("--result_dir", default="results/probe/coverage")
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    d = np.load(os.path.expanduser(args.features), allow_pickle=False)
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    v = d["v"][keep].astype(np.float64)
    w = d["w"][keep].astype(np.float64)
    com = d["com_offset"][keep].astype(np.float64)
    bbox = d["bbox_diag"][keep].astype(np.float64)
    scenario = d["scenario"][keep]
    kind = d["kind"][keep]
    groups = d["group"][keep]

    vmag, wmag, cmag = (np.linalg.norm(a, axis=1) for a in (v, w, com))
    n = int(keep.sum())
    n_groups = len(np.unique(groups))
    print(f"{n} samples (of {keep.size}) | {n_groups} groups | model {str(d['model'])}")

    # --- 1. RANGE ------------------------------------------------------------
    mags = {"|v| (m/s)": vmag, "|w| (rad/s)": wmag,
            "|CoM| (m)": cmag, "bbox_diag (m)": bbox}
    ranges = {k: _summ(x) for k, x in mags.items()}
    # Per-component spread too — a magnitude can look broad while components
    # are axis-locked (e.g. v almost always vertical ⇒ x/z components dead).
    comp = {}
    for nm, a in (("v", v), ("w", w), ("com", com)):
        for i, ax in enumerate("xyz"):
            comp[f"{nm}_{ax}"] = _summ(np.abs(a[:, i]))
    settled = {
        "|v|<%.2f frac" % SETTLED_V: float((vmag < SETTLED_V).mean()),
        "|w|<%.2f frac" % SETTLED_W: float((wmag < SETTLED_W).mean()),
    }

    # --- 2. CONFOUND (sample-level Pearson) ----------------------------------
    M = np.vstack([vmag, wmag, cmag, bbox])
    labels = ["|v|", "|w|", "|CoM|", "bbox"]
    corr = np.corrcoef(M)
    corr_tbl = {f"{labels[i]}~{labels[j]}": float(corr[i, j])
                for i in range(len(labels)) for j in range(i + 1, len(labels))}
    # The dangerous ones: any target tied to bbox (size is trivially visible).
    size_confounds = {k: v_ for k, v_ in corr_tbl.items() if k.endswith("~bbox")}

    # --- 3. COVERAGE (samples AND independent groups per stratum) ------------
    def by(key):
        tab = {}
        for lev in np.unique(key):
            m = key == lev
            tab[str(lev)] = {"samples": int(m.sum()),
                             "groups": int(len(np.unique(groups[m])))}
        return tab
    cov_scenario, cov_kind = by(scenario), by(kind)
    # group balance: how lopsided is samples-per-group?
    _, gcounts = np.unique(groups, return_counts=True)
    gbal = {"min": int(gcounts.min()), "max": int(gcounts.max()),
            "mean": float(gcounts.mean()), "median": float(np.median(gcounts))}

    audit = {"n_samples": n, "n_groups": n_groups, "model": str(d["model"]),
             "range": ranges, "range_components": comp, "settled": settled,
             "confound_pearson": corr_tbl, "coverage_scenario": cov_scenario,
             "coverage_kind": cov_kind, "group_balance": gbal}
    with open(os.path.join(result_dir, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # --- histograms ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (nm, x) in zip(axes.flat, mags.items()):
        ax.hist(x, bins=40, color="C0", alpha=0.85)
        ax.set_title(nm); ax.set_xlabel("value"); ax.set_ylabel("count")
        ax.axvline(np.median(x), ls="--", c="crimson", lw=1, label="median")
        ax.legend()
    # correlation heatmap
    axc = axes.flat[4]
    im = axc.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axc.set_xticks(range(len(labels))); axc.set_xticklabels(labels)
    axc.set_yticks(range(len(labels))); axc.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axc.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=9)
    axc.set_title("target ~ confound Pearson"); fig.colorbar(im, ax=axc, shrink=0.7)
    # groups-per-scenario bar
    axg = axes.flat[5]
    scs = list(cov_scenario)
    axg.bar(range(len(scs)), [cov_scenario[s]["groups"] for s in scs], color="C2")
    axg.set_xticks(range(len(scs))); axg.set_xticklabels(scs, rotation=60, ha="right", fontsize=7)
    axg.set_ylabel("# independent groups"); axg.set_title("groups per scenario")
    fig.suptitle(f"Probe corpus coverage — {n} samples / {n_groups} groups")
    fig.tight_layout()
    png = os.path.join(result_dir, "histograms.png")
    fig.savefig(png, dpi=120); plt.close(fig)

    # --- report.md (auto-read of the three questions) ------------------------
    def starved(s):
        return s["dyn_range_p95_p5"] < 3.0 or s["spread_ratio"] < 0.12
    # bbox_diag is a CONFOUND axis, not a decode target — a narrow size range is
    # fine (even helpful: less size for CoM to hide behind), so don't flag it as
    # "range-starved / widen it". Only the actual regression targets count here.
    range_flags = [k for k, s in ranges.items()
                   if starved(s) and not k.startswith("bbox")]
    confound_flags = [k for k, c in size_confounds.items() if abs(c) > 0.4]

    lines = [f"# Probe corpus coverage audit", "",
             f"Model `{str(d['model'])}` · {n} samples / **{n_groups} groups** "
             f"(weight ≥ {MIN_TOKEN_WEIGHT}).", "",
             "## 1. Range — is there anything to decode?", "",
             "| target | min | median | max | p95/p5 | spread | flag |",
             "|---|---|---|---|---|---|---|"]
    for k, s in ranges.items():
        dr = "∞" if not np.isfinite(s["dyn_range_p95_p5"]) else f"{s['dyn_range_p95_p5']:.1f}"
        lines.append(f"| {k} | {s['min']:.3f} | {s['p50']:.3f} | {s['max']:.3f} | "
                     f"{dr} | {s['spread_ratio']:.2f} | "
                     f"{'⚠ STARVED' if starved(s) else 'ok'} |")
    lines += ["",
              f"Settled (near-zero) mass: "
              + ", ".join(f"**{k} = {v_:.0%}**" for k, v_ in settled.items()) + ".",
              "A large near-zero |v|/|ω| mass means the mean predicts those samples "
              "for free; SST is inflated by motionless bodies and the probe is graded "
              "mostly on the moving minority.", "",
              "## 2. Confound — can the probe cheat via size?", "",
              "Sample-level Pearson (|target| vs bbox_diag = visible size):", ""]
    for k, c in size_confounds.items():
        lines.append(f"- `{k}` = **{c:+.2f}**"
                     + ("  ⚠ exploitable size proxy" if abs(c) > 0.4 else ""))
    lines += ["", "Among targets: "
              + ", ".join(f"`{k}`={v_:+.2f}" for k, v_ in corr_tbl.items()
                          if not k.endswith("~bbox")) + ".", "",
              "## 3. Coverage — groups per stratum", "",
              "| scenario | samples | groups |", "|---|---|---|"]
    for s, c in sorted(cov_scenario.items(), key=lambda kv: -kv[1]["groups"]):
        lines.append(f"| {s} | {c['samples']} | {c['groups']} "
                     + ("⚠" if c["groups"] < 3 else "") + " |")
    lines += ["", f"Body kind: " + ", ".join(
                  f"{k} ({c['samples']}s/{c['groups']}g)" for k, c in cov_kind.items()),
              f"  ·  group balance samples/group: min {gbal['min']}, "
              f"median {gbal['median']:.0f}, max {gbal['max']}.", "",
              "## Verdict — which lever does the overnight run pull?", ""]
    if range_flags:
        lines.append(f"- **RANGE is a limiter** for {', '.join(range_flags)}: widen "
                     "the sampling range for these (non-retrofittable — set before scale).")
    else:
        lines.append("- **Range looks adequate** — targets span a usable dynamic range.")
    if confound_flags:
        lines.append(f"- **CONFOUND present**: {', '.join(confound_flags)} tracks size; "
                     "a positive probe may be decoding silhouette, not physics. The "
                     "stratified sampler must break density⊥size / CoM⊥shape.")
    else:
        lines.append("- **No strong size confound** at the magnitude level (check "
                     "components / density separately).")
    starved_regimes = [s for s, c in cov_scenario.items() if c["groups"] < 3]
    if starved_regimes:
        lines.append(f"- **GROUP-starved regimes**: {', '.join(starved_regimes)} have "
                     "<3 independent groups — they cannot survive a group split; the "
                     "overnight run needs more *rollouts* in these, not more cameras.")
    lines += ["",
              "> Density/mass are not in the feature cache — the density⊥size confound "
              "(the core R8 latent-physics test) needs a `meta.json` pass; not covered here.",
              "", f"Plot: `histograms.png` · raw: `audit.json`."]
    with open(os.path.join(result_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- console digest ------------------------------------------------------
    print("\nRANGE:")
    for k, s in ranges.items():
        print(f"  {k:16s} med {s['p50']:.3f}  range [{s['min']:.3f},{s['max']:.3f}]"
              f"  p95/p5 {s['dyn_range_p95_p5']:.1f}  {'STARVED' if starved(s) else ''}")
    print(f"  settled: " + ", ".join(f"{k}={v_:.0%}" for k, v_ in settled.items()))
    print("CONFOUND (vs bbox):", {k: round(v_, 2) for k, v_ in size_confounds.items()})
    print("COVERAGE groups/scenario:",
          {s: c["groups"] for s, c in cov_scenario.items()})
    print(f"\nreport.md + histograms.png + audit.json -> {result_dir}")


if __name__ == "__main__":
    main()
