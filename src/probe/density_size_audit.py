"""Density ⟂ size meta audit — the R8 confound the feature cache cannot see.

The coverage audit (coverage_audit.py) found the kinematic targets clean against
size but `|CoM| ~ bbox = +0.62`. It could NOT test the core R8 question — *is mass
inferable from dynamics, not appearance?* — because density/mass live in the clip
`meta.json`, not in the feature cache. This closes that gap.

R8 / physics-diversification.md W1 *intends* density to be sampled independently of
shape so that `mass = density · volume` is NOT a deterministic function of visible
size. If density still tracks size, a probe "decoding mass" is really decoding
**volume** (trivially visible), and the latent-physics premise is confounded at the
generator. This script measures, over every dynamic body in the corpus:

  * `density ~ size`  — MUST be ≈ 0 for R8 to hold (the headline test);
  * `mass ~ size`     — high is expected (mass ∝ volume); the point is whether the
                        residual after volume is carried by *density variation*;
  * `|CoM| ~ size`    — re-checks the +0.62 confound at the body (meta) level;
  * density spread    — confirms W1 actually randomized density (non-degenerate).

Bodies are deduplicated by (rollout_seed, body id) so camera-duplicate clips don't
weight a rollout multiple times. Size proxy = volume^(1/3) (shape-agnostic length).
Effective density for composites = mass / Σ(part volume). Pearson (linear
exploitability) and Spearman (monotone) are both reported. CPU-only, no Isaac.

    python src/probe/density_size_audit.py --data ~/data/sim_physics \\
        --result_dir results/probe/density_audit
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.dataset import find_clip_dirs


def _spearman(a, b):
    """Rank-correlation = Pearson on ranks; catches monotone-nonlinear coupling
    (e.g. the cube law) that linear Pearson understates."""
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _com_mag(body):
    """CoM offset magnitude in body frame, kind-appropriate field (mirrors the
    dataset.py _COM_FIELD mapping: primitive→authored, composite→analytic)."""
    field = "com_local_authored" if body["kind"] == "primitive" else "com_local_analytic"
    c = body.get(field)
    return float(np.linalg.norm(c)) if c else 0.0


def _body_volume_density(body):
    """(volume, effective bulk density) for a body of either kind.
    primitive: body-level volume/density. composite: Σ part volume, mass/Σvolume."""
    if body["kind"] == "primitive":
        vol = float(body["volume"])
        return vol, float(body["density"])
    vol = float(sum(p["volume"] for p in body["parts"]))
    return vol, float(body["mass"]) / vol if vol > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/sim_physics")
    p.add_argument("--result_dir", default="results/probe/density_audit")
    args = p.parse_args()
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # Recursive discovery (probe-corpus layout nests clip dirs per scenario;
    # the old flat clip_*/meta.json glob finds nothing there). find_clip_dirs
    # walks, so both flat (sim_physics) and nested (probe_corpus) layouts work.
    metas = [os.path.join(d, "meta.json")
             for d in find_clip_dirs(os.path.expanduser(args.data))]
    seen = set()  # (rollout_seed, body_id) — dedup camera siblings
    rows = []     # per unique dynamic body
    for f in metas:
        m = json.load(open(f))
        rseed = m.get("multiview", {}).get("rollout_seed", m.get("seed"))
        scenario = m.get("scenario", "?")
        for b in m["bodies"]:
            key = (rseed, b["id"])
            if key in seen:
                continue
            seen.add(key)
            vol, dens = _body_volume_density(b)
            rows.append({
                "rollout_seed": rseed, "scenario": scenario, "kind": b["kind"],
                "size": vol ** (1.0 / 3.0), "volume": vol, "density": dens,
                "mass": float(b["mass"]), "com_mag": _com_mag(b),
            })

    n = len(rows)
    n_roll = len({r["rollout_seed"] for r in rows})
    size = np.array([r["size"] for r in rows])
    dens = np.array([r["density"] for r in rows])
    mass = np.array([r["mass"] for r in rows])
    com = np.array([r["com_mag"] for r in rows])
    print(f"{n} unique bodies | {n_roll} rollouts | {len(metas)} clip metas")

    def corr(a, b):
        return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": _spearman(a, b)}

    res = {
        "density~size": corr(dens, size),
        "mass~size": corr(mass, size),
        "|CoM|~size": corr(com, size),
        "density~mass": corr(dens, mass),
    }
    # log-log mass~size slope: pure mass=ρ·V with fixed ρ ⇒ slope 3 in log(size).
    # A slope < 3 with wide density range is the signature of density doing work.
    pos = (size > 0) & (mass > 0)
    slope = float(np.polyfit(np.log(size[pos]), np.log(mass[pos]), 1)[0]) if pos.sum() > 2 else float("nan")

    def stat(a):
        a = a[np.isfinite(a)]
        return {"min": float(a.min()), "max": float(a.max()), "median": float(np.median(a)),
                "ratio_max_min": float(a.max() / a.min()) if a.min() > 0 else float("inf")}
    spreads = {"density": stat(dens), "size": stat(size),
               "mass": stat(mass), "|CoM|": stat(com)}

    audit = {"n_bodies": n, "n_rollouts": n_roll, "corr": res,
             "mass_size_loglog_slope": slope, "spreads": spreads,
             "by_kind": {k: int(sum(r["kind"] == k for r in rows))
                         for k in ("primitive", "composite")}}
    with open(os.path.join(result_dir, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # --- scatter plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iscomp = np.array([r["kind"] == "composite" for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (y, lab, key) in zip(ax, ((dens, "density (kg/m³)", "density~size"),
                                     (mass, "mass (kg)", "mass~size"),
                                     (com, "|CoM offset| (m)", "|CoM|~size"))):
        a.scatter(size[~iscomp], y[~iscomp], s=14, alpha=0.6, label="primitive", color="C0")
        a.scatter(size[iscomp], y[iscomp], s=14, alpha=0.6, label="composite", color="C1")
        a.set_xlabel("size = volume^(1/3) (m)"); a.set_ylabel(lab)
        a.set_title(f"{key}: r={res[key]['pearson']:+.2f} "
                    f"(ρ_s={res[key]['spearman']:+.2f})")
        a.legend()
    fig.suptitle(f"Density / mass / CoM vs size — {n} bodies / {n_roll} rollouts")
    fig.tight_layout()
    png = os.path.join(result_dir, "scatter.png")
    fig.savefig(png, dpi=120); plt.close(fig)

    # --- report.md ---
    dps, dss = res["density~size"]["pearson"], res["density~size"]["spearman"]
    r8_ok = abs(dps) < 0.2 and abs(dss) < 0.2
    cps = res["|CoM|~size"]["pearson"]
    lines = [
        "# Density ⟂ size meta audit (R8 confound)", "",
        f"{n} unique dynamic bodies / {n_roll} rollouts (deduped across cameras) "
        f"from {len(metas)} clip metas. Size proxy = volume^(1/3).", "",
        "## Correlations (size = visible volume length scale)", "",
        "| pair | Pearson | Spearman | meaning |",
        "|---|---|---|---|",
        f"| **density ~ size** | **{dps:+.2f}** | **{dss:+.2f}** | R8 core: must be ≈0 |",
        f"| mass ~ size | {res['mass~size']['pearson']:+.2f} | {res['mass~size']['spearman']:+.2f} | high expected (mass∝volume) |",
        f"| \\|CoM\\| ~ size | {cps:+.2f} | {res['|CoM|~size']['spearman']:+.2f} | re-check of feature-cache +0.62 |",
        f"| density ~ mass | {res['density~mass']['pearson']:+.2f} | {res['density~mass']['spearman']:+.2f} | — |",
        "",
        f"log–log `mass~size` slope = **{slope:.2f}** "
        f"(3.0 = pure cube law / fixed density; lower ⇒ density variation is doing work).",
        "",
        "## Spreads", "",
        "| quantity | min | median | max | max/min |",
        "|---|---|---|---|---|",
    ] + [
        "| {} | {:.4g} | {:.4g} | {:.4g} | {} |".format(
            k, s["min"], s["median"], s["max"],
            "∞" if not np.isfinite(s["ratio_max_min"]) else f"{s['ratio_max_min']:.1f}")
        for k, s in spreads.items()
    ] + [
        "", "## Verdict", "",
        (f"- **R8 HOLDS at the generator**: density is decoupled from size "
         f"(|r|={abs(dps):.2f}, |ρ_s|={abs(dss):.2f} < 0.2). Mass is *not* a "
         f"deterministic function of visible volume — a probe that recovers mass "
         f"must be using dynamics. Density spread is "
         f"{spreads['density']['ratio_max_min']:.1f}× (W1 randomization live)."
         if r8_ok else
         f"- **R8 AT RISK**: density tracks size (Pearson {dps:+.2f}, "
         f"Spearman {dss:+.2f}). A probe could decode mass via visible volume; the "
         f"sampler must break density⊥size before scale (non-retrofittable)."),
        (f"- **CoM ~ size confirmed at meta level** ({cps:+.2f}) — consistent with "
         f"the feature-cache +0.62; offset magnitude scales with body size, so "
         f"sample absolute CoM offset decoupled from extent."
         if abs(cps) > 0.4 else
         f"- CoM ~ size at meta level is {cps:+.2f} (weaker than the feature-cache "
         f"+0.62; the cache value may be inflated by pose/camera projection)."),
        "",
        "Scatter: `scatter.png` · raw: `audit.json`. Companion: "
        "[scene-variability-rationale.md](../../plans/data-gen/scene-variability-rationale.md) §5.",
    ]
    with open(os.path.join(result_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndensity~size  Pearson {dps:+.3f}  Spearman {dss:+.3f}  "
          f"{'R8 OK' if r8_ok else 'R8 AT RISK'}")
    print(f"mass~size {res['mass~size']['pearson']:+.3f} (loglog slope {slope:.2f}) | "
          f"|CoM|~size {cps:+.3f} | density spread {spreads['density']['ratio_max_min']:.1f}x")
    print(f"report.md + scatter.png + audit.json -> {result_dir}")


if __name__ == "__main__":
    main()
